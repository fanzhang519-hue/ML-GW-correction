"""Predict GW corrections from a trusted checkpoint and processed test data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import train_QM9
import train_crystal
import train_crystal_spin


MODULES = {
    "qm9": train_QM9,
    "crystal": train_crystal,
    "crystal-spin": train_crystal_spin,
}


def scalar(x) -> float:
    return float(torch.as_tensor(x).flatten()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=MODULES, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Predict only the first N filtered samples (smoke test)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    module = MODULES[args.kind]
    # The checkpoint and .pt dataset files are pickle-based. Only load trusted files.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = module.TrainConfig(**saved["config"])

    if args.kind == "qm9":
        dataset = module.DeltaDataset(args.data_dir, cfg.target_key, cfg)
        collate = lambda samples: module.collate_batch(samples, cfg.cutoff)
    else:
        dataset = module.AoDataset(args.data_dir, cfg)
        collate = lambda samples: module.collate_batch(samples, cfg.cutoff)
    view = dataset if args.limit <= 0 else Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(view, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    first = next(iter(loader))

    if args.kind == "qm9":
        model = module.LitDeltaGNN(first["ele"].shape[-1], first["wave"].shape[-1], first["charge"].shape[-1], cfg)
    else:
        model = module.CgcnnAoDualHead(first["node_attr"].shape[-1], cfg)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    systems = dataset.data["system_id"] if args.kind == "qm9" else dataset.data.get("system")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("sample_index", "system_id", "role", "y_true_eV", "y_pred_eV"))
        offset = 0
        with torch.no_grad():
            for batch in loader:
                moved = module.move(batch, device)
                prediction = module.predict(model, moved).detach().cpu().tolist()
                for local_index, y_pred in enumerate(prediction):
                    i = offset + local_index
                    if args.kind == "qm9":
                        role = int(scalar(dataset.data["band_role"][i]))
                    else:
                        role = int(batch["role"][local_index])
                    system = "" if systems is None else str(systems[i])
                    writer.writerow((i, system, role, float(batch["y"][local_index]), float(y_pred)))
                offset += len(prediction)
    print(f"Wrote {offset} predictions to {args.output}")


if __name__ == "__main__":
    main()
