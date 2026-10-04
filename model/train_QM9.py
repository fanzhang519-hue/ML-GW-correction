from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.nn import GINEConv, global_mean_pool

REQUIRED_KEYS = ("ele", "wave", "charge", "edge_index", "edge_attr", "band_role", "system_id")


@dataclass
class TrainConfig:
    data_root: str
    train_data_root: str
    test_data_root: str
    train_molecules: int
    output_dir: str
    target_key: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    branch_dim: int
    hidden_dim: int
    gnn_layers: int
    edge_rbf_bins: int
    cutoff: float
    dropout: float
    seed: int
    train_ratio: float
    split_mode: str
    num_workers: int
    grad_clip: float
    huber_delta: float
    ema_decay: float
    lr_patience: int
    lr_factor: float
    min_lr: float
    early_stop_patience: int
    clean_labels: bool
    homo_min: float
    homo_max: float
    lumo_min: float
    lumo_max: float


def resolve_data_dir(root: Path, target_key: str) -> Path:
    candidates = [root, root / "homo_lumo_dataset", root / "dataset", root / "dataset" / "homo_lumo_dataset"]
    for cand in candidates:
        if all((cand / f"{k}.pt").exists() for k in REQUIRED_KEYS) and (cand / f"{target_key}.pt").exists():
            return cand
    raise FileNotFoundError("Could not find homo_lumo_dataset with required pt files")


class DeltaDataset(Dataset):
    def __init__(self, data_root: Path, target_key: str, cfg: TrainConfig) -> None:
        self.data_dir = resolve_data_dir(data_root, target_key)
        raw: Dict[str, List] = {}
        for key in REQUIRED_KEYS + (target_key,):
            raw[key] = torch.load(self.data_dir / f"{key}.pt", map_location="cpu")
        n = len(raw[target_key])
        keep = []
        removed = 0
        for i in range(n):
            y = float(torch.as_tensor(raw[target_key][i]).flatten()[0])
            role = int(torch.as_tensor(raw["band_role"][i]).flatten()[0])
            ok = True
            if cfg.clean_labels:
                if role == 2:
                    ok = cfg.homo_min <= y <= cfg.homo_max
                elif role == 3:
                    ok = cfg.lumo_min <= y <= cfg.lumo_max
            if ok:
                keep.append(i)
            else:
                removed += 1
        self.data: Dict[str, List] = {k: [v[i] for i in keep] for k, v in raw.items()}
        self.target_key = target_key
        self.removed = removed
        if len(keep) == 0:
            raise ValueError("No samples left after filtering")

    def __len__(self) -> int:
        return len(self.data[self.target_key])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        out = {k: self.data[k][idx] for k in REQUIRED_KEYS + (self.target_key,)}
        out["y"] = out[self.target_key]
        return out


def as_float_2d(x) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    return t


def scalar_float(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float32).flatten()[0]


def scalar_long(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.long).flatten()[0]


def collate_batch(samples: Sequence[Dict[str, Tensor]], cutoff: float) -> Dict[str, Tensor]:
    ele_parts, wave_parts, charge_parts, batch_parts = [], [], [], []
    edge_indices, edge_attrs = [], []
    node_offset = 0
    for graph_id, sample in enumerate(samples):
        ele = as_float_2d(sample["ele"])
        wave = as_float_2d(sample["wave"])
        charge = as_float_2d(sample["charge"])
        n = ele.shape[0]
        if wave.shape[0] != n or charge.shape[0] != n:
            raise ValueError("Atom count mismatch among ele/wave/charge")
        ele_parts.append(ele); wave_parts.append(wave); charge_parts.append(charge)
        batch_parts.append(torch.full((n,), graph_id, dtype=torch.long))
        edge_index = torch.as_tensor(sample["edge_index"], dtype=torch.long)
        edge_attr = as_float_2d(sample["edge_attr"])
        if edge_index.numel() > 0:
            dist = edge_attr.abs() if edge_attr.shape[-1] == 1 else edge_attr.norm(dim=-1, keepdim=True)
            mask = (dist.squeeze(-1) <= cutoff) if cutoff > 0 else torch.ones(edge_attr.shape[0], dtype=torch.bool)
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
            if edge_index.numel() > 0:
                edge_indices.append(edge_index + node_offset)
                edge_attrs.append(edge_attr)
        node_offset += n
    edge_index_batch = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr_batch = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.empty((0, 1), dtype=torch.float32)
    return {
        "ele": torch.cat(ele_parts, dim=0),
        "wave": torch.cat(wave_parts, dim=0),
        "charge": torch.cat(charge_parts, dim=0),
        "edge_index": edge_index_batch,
        "edge_attr": edge_attr_batch,
        "batch": torch.cat(batch_parts, dim=0),
        "band_role": torch.stack([scalar_long(s["band_role"]) for s in samples]),
        "y": torch.stack([scalar_float(s["y"]) for s in samples]),
    }


class RBFExpansion(nn.Module):
    def __init__(self, cutoff: float, bins: int) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, bins)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (centers[1] - centers[0]).square()

    def forward(self, edge_attr: Tensor) -> Tensor:
        if edge_attr.numel() == 0:
            return edge_attr.new_zeros((0, self.centers.numel()))
        dist = edge_attr.abs() if edge_attr.shape[-1] == 1 else edge_attr.norm(dim=-1, keepdim=True)
        return torch.exp(-self.gamma * (dist - self.centers.view(1, -1)).square())


class Branch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.LayerNorm(out_dim), nn.Dropout(dropout))
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LitDeltaGNN(nn.Module):
    def __init__(self, ele_dim: int, wave_dim: int, charge_dim: int, cfg: TrainConfig) -> None:
        super().__init__()
        self.ele = Branch(ele_dim, cfg.branch_dim, cfg.dropout)
        self.wave = Branch(wave_dim, cfg.branch_dim, cfg.dropout)
        self.charge = Branch(charge_dim, cfg.branch_dim, cfg.dropout)
        self.node_proj = nn.Sequential(nn.Linear(3 * cfg.branch_dim, cfg.hidden_dim), nn.SiLU(), nn.LayerNorm(cfg.hidden_dim))
        self.rbf = RBFExpansion(cfg.cutoff, cfg.edge_rbf_bins)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(cfg.dropout)
        for _ in range(cfg.gnn_layers):
            mlp = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=cfg.edge_rbf_bins))
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))
        self.homo_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1))
        self.lumo_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1))

    def forward(self, ele: Tensor, wave: Tensor, charge: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor, band_role: Tensor) -> Tensor:
        h = torch.cat([self.ele(ele), self.wave(wave), self.charge(charge)], dim=-1)
        h = self.node_proj(h)
        e = self.rbf(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            res = h
            h = conv(h, edge_index, e)
            h = self.drop(F.silu(norm(h + res)))
        g = global_mean_pool(h, batch)
        homo = self.homo_head(g).squeeze(-1)
        lumo = self.lumo_head(g).squeeze(-1)
        return torch.where(band_role == 2, homo, lumo)


def seed_everything(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_indices(dataset: DeltaDataset, cfg: TrainConfig):
    rng = random.Random(cfg.seed)
    if cfg.split_mode == "sample":
        idx = list(range(len(dataset))); rng.shuffle(idx)
        n_train = int(math.floor(len(idx) * cfg.train_ratio))
        return idx[:n_train], idx[n_train:]
    groups: Dict[str, List[int]] = {}
    for i, sid in enumerate(dataset.data["system_id"]):
        groups.setdefault(str(sid), []).append(i)
    items = list(groups.items()); rng.shuffle(items)
    n_train_groups = int(math.floor(len(items) * cfg.train_ratio))
    train = [i for _, ids in items[:n_train_groups] for i in ids]
    test = [i for _, ids in items[n_train_groups:] for i in ids]
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def _system_key(x):
    if isinstance(x, torch.Tensor):
        t = x.flatten()
        if t.numel() == 1:
            return str(int(t[0].item()))
        return "_".join(str(int(v.item())) for v in t)
    return str(x)


def select_molecule_indices(dataset: DeltaDataset, n_molecules: int, seed: int) -> List[int]:
    groups: Dict[str, List[int]] = {}
    for i, sid in enumerate(dataset.data["system_id"]):
        groups.setdefault(_system_key(sid), []).append(i)
    items = sorted(groups.items(), key=lambda kv: kv[0])
    if n_molecules > 0:
        if len(items) < n_molecules:
            raise ValueError(f"Requested {n_molecules} train molecules, but only {len(items)} are available")
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:n_molecules]
    idx = [i for _, ids in items for i in ids]
    random.Random(seed).shuffle(idx)
    return idx


def make_loaders(cfg: TrainConfig):
    if cfg.train_data_root and cfg.test_data_root:
        train_dataset = DeltaDataset(Path(cfg.train_data_root), cfg.target_key, cfg)
        test_dataset = DeltaDataset(Path(cfg.test_data_root), cfg.target_key, cfg)
        train_idx = select_molecule_indices(train_dataset, cfg.train_molecules, cfg.seed)
        test_idx = list(range(len(test_dataset)))
        collate = lambda samples: collate_batch(samples, cfg.cutoff)
        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
        return train_dataset, train_loader, test_loader, len(train_idx), len(test_idx)

    dataset = DeltaDataset(Path(cfg.data_root), cfg.target_key, cfg)
    train_idx, test_idx = split_indices(dataset, cfg)
    collate = lambda samples: collate_batch(samples, cfg.cutoff)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    return dataset, train_loader, test_loader, len(train_idx), len(test_idx)


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def role_stats(loader):
    ys, roles = [], []
    for b in loader:
        ys.append(b["y"]); roles.append(b["band_role"])
    y = torch.cat(ys); role = torch.cat(roles)
    stats = {}
    for r in [2, 3]:
        vals = y[role == r]
        stats[r] = (float(vals.mean()), float(vals.std(unbiased=False).clamp_min(1e-8)))
    return stats


def norm_by_role(values: Tensor, roles: Tensor, stats: Dict[int, tuple[float, float]]) -> Tensor:
    out = torch.empty_like(values)
    for r, (mean, std) in stats.items():
        m = roles == r
        if m.any():
            out[m] = (values[m] - mean) / std
    return out


def update_ema(ema, model, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()): ep.mul_(decay).add_(p, alpha=1.0-decay)
        for eb, b in zip(ema.buffers(), model.buffers()): eb.copy_(b)


def predict(model, batch):
    return model(batch["ele"], batch["wave"], batch["charge"], batch["edge_index"], batch["edge_attr"], batch["batch"], batch["band_role"])


def train_epoch(model, loader, opt, ema, cfg, device, stats):
    model.train(); total_abs = 0.0; total_n = 0
    for batch in loader:
        batch = move(batch, device)
        pred = predict(model, batch); y = batch["y"]
        pred_n = norm_by_role(pred, batch["band_role"], stats)
        y_n = norm_by_role(y, batch["band_role"], stats)
        loss = F.smooth_l1_loss(pred_n, y_n, beta=cfg.huber_delta)
        opt.zero_grad(set_to_none=True); loss.backward()
        if cfg.grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if ema is not None: update_ema(ema, model, cfg.ema_decay)
        total_abs += (pred.detach() - y).abs().sum().item(); total_n += y.numel()
    return total_abs / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); acc = {"all":[0,0.0], "homo":[0,0.0], "lumo":[0,0.0]}
    for batch in loader:
        batch = move(batch, device)
        pred = predict(model, batch); err = (pred - batch["y"]).abs()
        acc["all"][0] += err.numel(); acc["all"][1] += err.sum().item()
        for name, r in [("homo",2),("lumo",3)]:
            m = batch["band_role"] == r
            if m.any(): acc[name][0] += int(m.sum()); acc[name][1] += err[m].sum().item()
    return {k: s / max(n, 1) for k, (n, s) in acc.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/mnt/sdb/user_home/zhangf/qm9/0715-qm9/dataset")
    p.add_argument("--train-data-root", default="")
    p.add_argument("--test-data-root", default="")
    p.add_argument("--train-molecules", type=int, default=0, help="Use N molecules/systems from --train-data-root. 0 means all.")
    p.add_argument("--output-dir", default="./runs_lit")
    p.add_argument("--target-key", choices=["y", "y_abs"], default="y")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--branch-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=4)
    p.add_argument("--edge-rbf-bins", type=int, default=32)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--split-mode", choices=["system", "sample"], default="system")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--huber-delta", type=float, default=0.03)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--lr-patience", type=int, default=35)
    p.add_argument("--lr-factor", type=float, default=0.7)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stop-patience", type=int, default=200)
    p.add_argument("--clean-labels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--homo-min", type=float, default=-6.0)
    p.add_argument("--homo-max", type=float, default=-1.0)
    p.add_argument("--lumo-min", type=float, default=-1.0)
    p.add_argument("--lumo-max", type=float, default=5.0)
    return TrainConfig(**vars(p.parse_args()))


def main():
    cfg = parse_args(); seed_everything(cfg.seed)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, train_loader, test_loader, n_train, n_test = make_loaders(cfg)
    stats = role_stats(train_loader)
    first = next(iter(train_loader))
    model = LitDeltaGNN(first["ele"].shape[-1], first["wave"].shape[-1], first["charge"].shape[-1], cfg).to(device)
    ema = copy.deepcopy(model).eval() if cfg.ema_decay > 0 else None
    if ema is not None:
        for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    print(f"data_dir={dataset.data_dir} removed={dataset.removed} split={cfg.split_mode} train={n_train} test={n_test} device={device} dims={first['ele'].shape[-1]},{first['wave'].shape[-1]},{first['charge'].shape[-1]} role_stats={stats}")
    best, bad = float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        train_mae = train_epoch(model, train_loader, opt, ema, cfg, device, stats)
        raw = evaluate(model, test_loader, device)
        ema_m = evaluate(ema, test_loader, device) if ema is not None else raw
        if ema is not None and ema_m["all"] <= raw["all"]:
            eval_model, metrics, source = ema, ema_m, "ema"
        else:
            eval_model, metrics, source = model, raw, "raw"
        train_eval = evaluate(eval_model, train_loader, device)["all"]
        sched.step(metrics["all"])
        if metrics["all"] < best:
            best, bad = metrics["all"], 0
            torch.save({"epoch":epoch,"model_state_dict":eval_model.state_dict(),"raw_model_state_dict":model.state_dict(),"ema_model_state_dict":None if ema is None else ema.state_dict(),"optimizer_state_dict":opt.state_dict(),"config":asdict(cfg),"role_stats":stats,"train_mae":train_mae,"test_mae":metrics["all"],"weight_source":source}, out/"best_model.pt")
        else:
            bad += 1
        torch.save({"epoch":epoch,"model_state_dict":eval_model.state_dict(),"config":asdict(cfg),"role_stats":stats,"train_mae":train_mae,"test_mae":metrics["all"],"weight_source":source}, out/"last_model.pt")
        print(f"Epoch {epoch:04d}/{cfg.epochs} train_MAE={train_mae:.6f} train_eval_MAE={train_eval:.6f} test_MAE={metrics['all']:.6f} homo={metrics['homo']:.6f} lumo={metrics['lumo']:.6f} raw={raw['all']:.6f} ema={ema_m['all']:.6f} selected={source} best={best:.6f} lr={opt.param_groups[0]['lr']:.3e}")
        if cfg.early_stop_patience > 0 and bad >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch}"); break

if __name__ == "__main__":
    main()
