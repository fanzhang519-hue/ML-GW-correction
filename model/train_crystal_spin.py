from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.nn import GINEConv, global_mean_pool


REQUIRED_KEYS = (
    "element",
    "wave_l",
    "charge_l",
    "pos",
    "cell",
    "pbc",
    "E",
    "spin",
    "y",
)
OPTIONAL_META_KEYS = ("band", "band_abs", "occupation", "source_name", "case_name", "system", "case", "case_key", "band_name", "state_type", "kpoint")


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    energy_embedding_dim: int
    spin_embedding_dim: int
    num_spin_classes: int
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
    valence_min: float
    valence_max: float
    conduction_min: float
    conduction_max: float
    max_samples: int


def resolve_data_dir(root: Path) -> Path:
    candidates = [
        root,
        root / "hpro_ao_l_dataset",
        root / "dataset" / "hpro_ao_l_dataset",
        root / "0801" / "dataset" / "hpro_ao_l_dataset",
    ]
    for cand in candidates:
        if all((cand / f"{key}.pt").exists() for key in REQUIRED_KEYS):
            return cand
    raise FileNotFoundError(f"Could not find hpro_ao_l_dataset under {root}")


def as_float_2d(x) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    return t


def scalar_float(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float32).flatten()[0]


def scalar_long(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.long).flatten()[0]


def spin_from_sample(sample: Dict) -> Tensor:
    if "spin" not in sample:
        return torch.tensor(0, dtype=torch.long)
    return torch.tensor(int(scalar_long(sample["spin"])), dtype=torch.long)


def role_from_sample(sample: Dict) -> Tensor:
    # Prefer explicit metadata. Fall back to the VBM-referenced DFT energy E.
    state = str(sample.get("state_type", "")).lower()
    if state.startswith("unoccupied") or state.startswith("conduction"):
        return torch.tensor(1, dtype=torch.long)
    if state.startswith("occupied") or state.startswith("valence"):
        return torch.tensor(0, dtype=torch.long)
    band_name = str(sample.get("band_name", "")).lower()
    if band_name.startswith("l"):
        return torch.tensor(1, dtype=torch.long)
    if band_name.startswith("h"):
        return torch.tensor(0, dtype=torch.long)
    return (scalar_float(sample["E"]) > 0).long()


def band_offset_from_sample(sample: Dict) -> Tensor:
    return scalar_float(sample["band"]) if sample.get("band") is not None else torch.tensor(0.0)


def occupation_from_sample(sample: Dict) -> Tensor:
    if sample.get("occupation") is not None:
        return scalar_float(sample["occupation"])
    return 1.0 - role_from_sample(sample).float()


class AoDataset(Dataset):
    def __init__(self, data_root: Path, cfg: TrainConfig) -> None:
        self.data_dir = resolve_data_dir(data_root)
        print(f"loading tensors from {self.data_dir}", flush=True)
        raw: Dict[str, List] = {}
        for key in REQUIRED_KEYS:
            raw[key] = torch.load(self.data_dir / f"{key}.pt", map_location="cpu")
        for key in OPTIONAL_META_KEYS:
            p = self.data_dir / f"{key}.pt"
            if p.exists():
                raw[key] = torch.load(p, map_location="cpu")
        n = len(raw["y"])
        limit = n if cfg.max_samples <= 0 else min(n, cfg.max_samples)
        keep = []
        removed = 0
        for i in range(limit):
            y = float(scalar_float(raw["y"][i]))
            pseudo_sample = {key: raw[key][i] for key in raw}
            is_conduction = int(role_from_sample(pseudo_sample)) == 1
            ok = True
            if cfg.clean_labels:
                if is_conduction:
                    ok = cfg.conduction_min <= y <= cfg.conduction_max
                else:
                    ok = cfg.valence_min <= y <= cfg.valence_max
            if ok:
                keep.append(i)
            else:
                removed += 1
        self.data = {key: [values[i] for i in keep] for key, values in raw.items()}
        self.removed = removed
        if not keep:
            raise ValueError("No samples left after filtering")
        self.node_cache = []
        self.edge_cache = []
        t0 = time.time()
        for i in range(len(keep)):
            element = as_float_2d(self.data["element"][i])
            wave = as_float_2d(self.data["wave_l"][i])
            charge = as_float_2d(self.data["charge_l"][i])
            pos = as_float_2d(self.data["pos"][i])
            cell = torch.as_tensor(self.data["cell"][i], dtype=torch.float32)
            pbc = torch.as_tensor(self.data["pbc"][i], dtype=torch.bool).flatten()
            self.node_cache.append(torch.cat([element, wave, charge], dim=-1))
            self.edge_cache.append(build_edges(pos, cell, pbc, cfg.cutoff))
            if (i + 1) % 5000 == 0 or (i + 1) == len(keep):
                print(
                    f"built graph cache {i + 1}/{len(keep)} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.data["y"])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        out = {key: self.data[key][idx] for key in REQUIRED_KEYS}
        out["node_attr"] = self.node_cache[idx]
        out["edge_index"], out["edge_attr"] = self.edge_cache[idx]
        for key in OPTIONAL_META_KEYS:
            if key in self.data:
                out[key] = self.data[key][idx]
        return out


def pbc_shifts(pbc: Tensor, device: torch.device) -> Tensor:
    ranges = []
    for active in pbc.tolist():
        ranges.append([-1, 0, 1] if active else [0])
    shifts = [[i, j, k] for i in ranges[0] for j in ranges[1] for k in ranges[2]]
    return torch.tensor(shifts, dtype=torch.float32, device=device)


def build_edges(pos: Tensor, cell: Tensor, pbc: Tensor, cutoff: float) -> tuple[Tensor, Tensor]:
    n = pos.shape[0]
    device = pos.device
    shifts = pbc_shifts(pbc.bool(), device)
    shift_vec = shifts @ cell
    vec = pos.view(1, n, 1, 3) + shift_vec.view(1, 1, -1, 3) - pos.view(n, 1, 1, 3)
    dist = torch.linalg.norm(vec, dim=-1)
    mask = (dist > 1.0e-8) & (dist <= cutoff)
    ids = torch.nonzero(mask, as_tuple=False)
    if ids.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float32)
    edge_index = ids[:, :2].t().contiguous().to(dtype=torch.long)
    edge_attr = vec[ids[:, 0], ids[:, 1], ids[:, 2]].to(dtype=torch.float32).cpu()
    return edge_index, edge_attr


def collate_batch(samples: Sequence[Dict[str, Tensor]], cutoff: float) -> Dict[str, Tensor]:
    node_parts, batch_parts = [], []
    edge_indices, edge_attrs = [], []
    y_parts, e_parts, role_parts, spin_parts, band_parts, occ_parts = [], [], [], [], [], []
    source_names, case_names = [], []
    node_offset = 0
    for graph_id, sample in enumerate(samples):
        node_attr = as_float_2d(sample["node_attr"])
        n = node_attr.shape[0]
        node_parts.append(node_attr)
        batch_parts.append(torch.full((n,), graph_id, dtype=torch.long))
        edge_index = torch.as_tensor(sample["edge_index"], dtype=torch.long)
        edge_attr = as_float_2d(sample["edge_attr"])
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + node_offset)
            edge_attrs.append(edge_attr)
        node_offset += n
        y_parts.append(scalar_float(sample["y"]))
        e_parts.append(scalar_float(sample["E"]))
        role_parts.append(role_from_sample(sample))
        spin_parts.append(spin_from_sample(sample))
        band_parts.append(band_offset_from_sample(sample))
        occ_parts.append(occupation_from_sample(sample))
        source_names.append(str(sample.get("source_name", sample.get("system", ""))))
        case_names.append(str(sample.get("case_name", sample.get("case_key", sample.get("case", "")))))
    return {
        "node_attr": torch.cat(node_parts, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(edge_attrs, dim=0) if edge_attrs else torch.empty((0, 3), dtype=torch.float32),
        "batch": torch.cat(batch_parts, dim=0),
        "E": torch.stack(e_parts),
        "role": torch.stack(role_parts),
        "spin": torch.stack(spin_parts),
        "band": torch.stack(band_parts),
        "occupation": torch.stack(occ_parts),
        "y": torch.stack(y_parts),
        "source_name": source_names,
        "case_name": case_names,
    }


class RBFExpansion(nn.Module):
    def __init__(self, cutoff: float, bins: int) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, bins)
        self.register_buffer("centers", centers)
        spacing = max(float(centers[1] - centers[0]), 1.0e-6) if bins > 1 else cutoff
        self.gamma = 1.0 / (spacing * spacing)

    def forward(self, edge_attr: Tensor) -> Tensor:
        if edge_attr.numel() == 0:
            return edge_attr.new_zeros((0, self.centers.numel()))
        dist = edge_attr.norm(dim=-1, keepdim=True)
        return torch.exp(-self.gamma * (dist - self.centers.view(1, -1)).square())


class CgcnnAoDualHead(nn.Module):
    def __init__(self, node_dim: int, cfg: TrainConfig) -> None:
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(3, cfg.energy_embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(cfg.energy_embedding_dim),
        )
        self.spin_embedding = nn.Embedding(cfg.num_spin_classes, cfg.spin_embedding_dim)
        self.rbf = RBFExpansion(cfg.cutoff, cfg.edge_rbf_bins)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(cfg.dropout)
        for _ in range(cfg.gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=cfg.edge_rbf_bins))
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))
        head_in = cfg.hidden_dim + cfg.energy_embedding_dim + cfg.spin_embedding_dim
        self.valence_head = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.conduction_head = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, node_attr: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor, role: Tensor, E: Tensor, spin: Tensor, band: Tensor, occupation: Tensor) -> Tensor:
        h = self.node_proj(node_attr)
        e = self.rbf(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            res = h
            h = conv(h, edge_index, e)
            h = self.drop(F.silu(norm(h + res)))
        g = global_mean_pool(h, batch)
        state_h = self.state_proj(torch.stack([E, band, occupation], dim=-1))
        spin_h = self.spin_embedding(spin.clamp(min=0, max=self.spin_embedding.num_embeddings - 1).long())
        gb = torch.cat([g, state_h, spin_h], dim=-1)
        valence_residual = self.valence_head(gb).squeeze(-1)
        conduction_residual = self.conduction_head(gb).squeeze(-1)
        residual = torch.where(role == 0, valence_residual, conduction_residual)
        return residual


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(dataset: AoDataset, cfg: TrainConfig) -> tuple[list[int], list[int]]:
    rng = random.Random(cfg.seed)
    if cfg.split_mode == "sample":
        idx = list(range(len(dataset)))
        rng.shuffle(idx)
        n_train = int(math.floor(len(idx) * cfg.train_ratio))
        return idx[:n_train], idx[n_train:]
    groups: Dict[str, List[int]] = {}
    cases = dataset.data.get("case_name", dataset.data.get("case_key", dataset.data.get("case")))
    sources = dataset.data.get("source_name", dataset.data.get("system"))
    for i in range(len(dataset)):
        if cases is not None:
            source = str(sources[i]) if sources is not None else ""
            key = f"{source}:{cases[i]}"
        else:
            key = str(i)
        groups.setdefault(key, []).append(i)
    items = list(groups.items())
    rng.shuffle(items)
    n_train_groups = int(math.floor(len(items) * cfg.train_ratio))
    train = [i for _, ids in items[:n_train_groups] for i in ids]
    test = [i for _, ids in items[n_train_groups:] for i in ids]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_loaders(cfg: TrainConfig):
    dataset = AoDataset(Path(cfg.data_root), cfg)
    train_idx, test_idx = split_indices(dataset, cfg)
    collate = lambda samples: collate_batch(samples, cfg.cutoff)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, train_loader, test_loader, len(train_idx), len(test_idx), train_idx, test_idx


def move(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def dataset_role_stats(dataset: AoDataset, indices: Sequence[int]) -> Dict[int, tuple[float, float]]:
    ys, roles = [], []
    for i in indices:
        sample = {key: dataset.data[key][i] for key in dataset.data}
        ys.append(scalar_float(dataset.data["y"][i]))
        roles.append(role_from_sample(sample))
    y = torch.stack(ys)
    role = torch.stack(roles)
    stats = {}
    for r in [0, 1]:
        vals = y[role == r]
        stats[r] = (float(vals.mean()), float(vals.std(unbiased=False).clamp_min(1.0e-8))) if vals.numel() else (0.0, 1.0)
    return stats


def norm_by_role(values: Tensor, roles: Tensor, stats: Dict[int, tuple[float, float]]) -> Tensor:
    out = torch.empty_like(values)
    for r, (mean, std) in stats.items():
        mask = roles == r
        if mask.any():
            out[mask] = (values[mask] - mean) / std
    return out


def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.mul_(decay).add_(p, alpha=1.0 - decay)
        for eb, b in zip(ema.buffers(), model.buffers()):
            eb.copy_(b)


def predict(model: nn.Module, batch: Dict) -> Tensor:
    return model(batch["node_attr"], batch["edge_index"], batch["edge_attr"], batch["batch"], batch["role"], batch["E"], batch["spin"], batch["band"], batch["occupation"])


def train_epoch(model, loader, opt, ema, cfg, device, stats) -> float:
    model.train()
    total_abs = 0.0
    total_n = 0
    for batch in loader:
        batch = move(batch, device)
        pred = predict(model, batch)
        y = batch["y"]
        loss = F.smooth_l1_loss(
            norm_by_role(pred, batch["role"], stats),
            norm_by_role(y, batch["role"], stats),
            beta=cfg.huber_delta,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if ema is not None:
            update_ema(ema, model, cfg.ema_decay)
        total_abs += (pred.detach() - y).abs().sum().item()
        total_n += y.numel()
    return total_abs / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    acc = {"all": [0, 0.0], "valence": [0, 0.0], "conduction": [0, 0.0]}
    for batch in loader:
        batch = move(batch, device)
        pred = predict(model, batch)
        err = (pred - batch["y"]).abs()
        acc["all"][0] += err.numel()
        acc["all"][1] += err.sum().item()
        for name, role in [("valence", 0), ("conduction", 1)]:
            mask = batch["role"] == role
            if mask.any():
                acc[name][0] += int(mask.sum())
                acc[name][1] += err[mask].sum().item()
    return {key: total / max(n, 1) for key, (n, total) in acc.items()}


@torch.no_grad()
def save_predictions(model, loader, device, out_csv: Path) -> None:
    model.eval()
    rows = []
    for batch in loader:
        batch_dev = move(batch, device)
        pred = predict(model, batch_dev).cpu()
        y = batch["y"].cpu()
        role = batch["role"].cpu()
        spin = batch["spin"].cpu()
        e_rel = batch["E"].cpu()
        for i in range(y.numel()):
            rows.append(
                {
                    "source_name": batch["source_name"][i],
                    "case_name": batch["case_name"][i],
                    "E_rel_vbm_eV": float(e_rel[i]),
                    "spin": int(spin[i]),
                    "role": "conduction" if int(role[i]) == 1 else "valence",
                    "y_true": float(y[i]),
                    "y_pred": float(pred[i]),
                    "abs_err": abs(float(pred[i] - y[i])),
                }
            )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["source_name", "case_name", "band", "role", "y_true", "y_pred", "abs_err"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/mnt/sdb/user_home/zhangf/qm9/e3nn/crystal/dataset")
    p.add_argument("--output-dir", default="./runs_cgcnn2_band_residual")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--energy-embedding-dim", "--band-embedding-dim", dest="energy_embedding_dim", type=int, default=16)
    p.add_argument("--spin-embedding-dim", type=int, default=8)
    p.add_argument("--num-spin-classes", type=int, default=3)
    p.add_argument("--gnn-layers", type=int, default=4)
    p.add_argument("--edge-rbf-bins", type=int, default=32)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--split-mode", choices=["system", "sample"], default="sample")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--huber-delta", type=float, default=0.03)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--lr-patience", type=int, default=35)
    p.add_argument("--lr-factor", type=float, default=0.7)
    p.add_argument("--min-lr", type=float, default=1.0e-6)
    p.add_argument("--early-stop-patience", type=int, default=200)
    p.add_argument("--clean-labels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--valence-min", type=float, default=-15.0)
    p.add_argument("--valence-max", type=float, default=15.0)
    p.add_argument("--conduction-min", type=float, default=-15.0)
    p.add_argument("--conduction-max", type=float, default=15.0)
    p.add_argument("--max-samples", type=int, default=0)
    return TrainConfig(**vars(p.parse_args()))


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"startup: device={device} data_root={cfg.data_root} split={cfg.split_mode}", flush=True)
    dataset, train_loader, test_loader, n_train, n_test, train_idx, test_idx = make_loaders(cfg)
    print(f"dataset_loaded: data_dir={dataset.data_dir} samples={len(dataset)} removed={dataset.removed} train={n_train} test={n_test}", flush=True)
    stats = dataset_role_stats(dataset, train_idx)
    first = next(iter(train_loader))
    model = CgcnnAoDualHead(first["node_attr"].shape[-1], cfg).to(device)
    ema = copy.deepcopy(model).eval() if cfg.ema_decay > 0 else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
    )
    print(
        "data_dir={} removed={} split={} train={} test={} device={} node_dim={} role_stats={}".format(
            dataset.data_dir,
            dataset.removed,
            cfg.split_mode,
            n_train,
            n_test,
            device,
            first["node_attr"].shape[-1],
            stats,
        )
    )
    print("state inputs are E, relative band offset, occupation, and spin embedding; no defect label")
    best = float("inf")
    bad = 0
    best_source = "raw"
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_mae = train_epoch(model, train_loader, opt, ema, cfg, device, stats)
        raw = evaluate(model, test_loader, device)
        ema_metrics = evaluate(ema, test_loader, device) if ema is not None else raw
        if ema is not None and ema_metrics["all"] <= raw["all"]:
            eval_model, metrics, source = ema, ema_metrics, "ema"
        else:
            eval_model, metrics, source = model, raw, "raw"
        train_eval = evaluate(eval_model, train_loader, device)["all"]
        sched.step(metrics["all"])
        if metrics["all"] < best:
            best = metrics["all"]
            bad = 0
            best_source = source
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "raw_model_state_dict": model.state_dict(),
                    "ema_model_state_dict": None if ema is None else ema.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "config": asdict(cfg),
                    "role_stats": stats,
                    "state_features": "E, relative band offset, occupation, and spin embedding; no defect label",
                    "spin_feature": "spin.pt embedding: 0=nonspin, 1=up, 2=down",
                    "train_mae": train_mae,
                    "test_mae": metrics["all"],
                    "metrics": metrics,
                    "weight_source": source,
                },
                out / "best_model.pt",
            )
            save_predictions(eval_model, test_loader, device, out / "test_predictions_best.csv")
        else:
            bad += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": eval_model.state_dict(),
                "config": asdict(cfg),
                "role_stats": stats,
                "state_features": "E, relative band offset, occupation, and spin embedding; no defect label",
                "spin_feature": "spin.pt embedding: 0=nonspin, 1=up, 2=down",
                "train_mae": train_mae,
                "test_mae": metrics["all"],
                "metrics": metrics,
                "weight_source": source,
            },
            out / "last_model.pt",
        )
        print(
            f"Epoch {epoch:04d}/{cfg.epochs} train_MAE={train_mae:.6f} "
            f"train_eval_MAE={train_eval:.6f} test_MAE={metrics['all']:.6f} "
            f"valence={metrics['valence']:.6f} conduction={metrics['conduction']:.6f} "
            f"raw={raw['all']:.6f} ema={ema_metrics['all']:.6f} selected={source} "
            f"best={best:.6f} best_source={best_source} lr={opt.param_groups[0]['lr']:.3e} time={time.time()-t0:.2f}s"
        )
        if cfg.early_stop_patience > 0 and bad >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    main()
