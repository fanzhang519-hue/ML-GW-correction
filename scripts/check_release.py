"""Check the presence of publication artifacts without deserializing PyTorch data."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "QM9/train": ("ele", "wave", "charge", "edge_index", "edge_attr", "band_role", "system_id", "y"),
    "QM9/test": ("ele", "wave", "charge", "edge_index", "edge_attr", "band_role", "system_id", "y"),
    "Crystal": ("element", "wave_l", "charge_l", "pos", "cell", "pbc", "E", "y"),
    "Crystal-spin": ("element", "wave_l", "charge_l", "pos", "cell", "pbc", "E", "spin", "y"),
}
WEIGHTS = (
    "ml_gw/qm9_120k_best_model.pt",
    "ml_gw/crystal_best_model.pt",
    "ml_gw/crystal_spin_best_model.pt",
    "deeph/diamond_best.tar.gz",
    "deeph/silicon_best.tar.gz",
    "deeph/nv_up_best.tar.gz",
    "deeph/nv_down_best.tar.gz",
)
MODELS = ("train_QM9.py", "train_crystal.py", "train_crystal_spin.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    missing: list[str] = []
    total_bytes = 0

    def check(path: Path) -> None:
        nonlocal total_bytes
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(root)))
        else:
            total_bytes += path.stat().st_size

    for name, keys in DATASETS.items():
        folder = root / "dataset" / name
        for key in keys:
            check(folder / f"{key}.pt")
    for name in MODELS:
        check(root / "model" / name)
    for name in WEIGHTS:
        check(root / "weights" / name)
    for name in ("README.md", "requirements.txt", ".gitattributes", ".gitignore"):
        check(root / name)

    if missing:
        print("MISSING/EMPTY:")
        for path in missing:
            print(f"  {path}")
        return 1
    print(f"Required artifacts present. Checked bytes: {total_bytes:,}")
    print("Note: this checks presence only, not sample alignment or scientific validity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
