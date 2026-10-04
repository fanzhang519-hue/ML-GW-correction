# ML-assisted GW quasiparticle corrections

Processed data, training code, and selected trained weights associated with the accompanying manuscript. This directory is a release package, not a collection of raw Quantum ESPRESSO, BerkeleyGW, or HPRO calculation outputs.

## Contents

| Path | Contents |
| --- | --- |
| `dataset/QM9/train`, `dataset/QM9/test` | Paired molecular HOMO/LUMO samples; the released test split contains 5,000 molecules. |
| `dataset/Crystal` | Non-spin crystal electronic-state samples. |
| `dataset/Crystal-spin` | Spin-aware crystal/defect electronic-state samples. |
| `model/train_QM9.py` | Molecular dual-head GW-correction model. |
| `model/train_crystal.py` | Non-spin crystal dual-head GW-correction model. |
| `model/train_crystal_spin.py` | Spin-aware crystal/defect model. |
| `weights/ml_gw` | Matching best PyTorch checkpoints for the three scripts above. |
| `weights/deeph` | Best parameter trees for diamond, silicon, NV-minus spin-up, and NV-minus spin-down Hamiltonian models. |

The `*.pt` files are PyTorch-serialized objects. Only load them from a trusted source. Entries in each dataset directory are aligned **by list index** across the files in that directory. Do not independently shuffle individual files.

The released QM9 pool contains 257,759 state samples in `train` and 10,000 state samples (5,000 HOMO/LUMO pairs) in `test`. `Crystal` contains 26,226 state samples; `Crystal-spin` contains 27,146. These counts describe processed electronic states, not numbers of distinct structures.

## GW-correction models

The molecular model uses atomic/graph features, orbital information, and occupied-orbital charge; the crystal models additionally use the DFT band energy relative to the VBM. The target `y` is the state-resolved GW correction in eV, not the quasiparticle energy. The crystal spin model additionally uses `spin.pt` (`0`: non-spin, `1`: up, `2`: down).

Use Python 3.12, PyTorch 2.7.1, and PyTorch Geometric 2.6.1 as in the original training environment. Install a PyTorch build compatible with your CPU/CUDA runtime before installing PyTorch Geometric. The scripts accept explicit data and output paths, so their historical server-path defaults do not need to exist.

```bash
python model/train_QM9.py \
  --train-data-root dataset/QM9/train \
  --test-data-root dataset/QM9/test \
  --train-molecules 120000 --target-key y \
  --output-dir runs/qm9 --batch-size 128 --num-workers 8

python model/train_crystal.py \
  --data-root dataset/Crystal --output-dir runs/crystal \
  --epochs 2000 --batch-size 256 --num-workers 0 --split-mode sample

python model/train_crystal_spin.py \
  --data-root dataset/Crystal-spin --output-dir runs/crystal-spin \
  --epochs 4000 --batch-size 256 --num-workers 0 --split-mode sample
```

These are training examples, not promises of byte-identical retraining across GPU/software environments. The saved `best_model.pt` files contain the original run configuration and state dictionary. The QM9 run used 120,000 molecules from the provided training pool; it did not train on every released training molecule.

To evaluate a saved molecular checkpoint on the released test data:

```bash
python model/predict_saved.py --kind qm9 \
  --data-dir dataset/QM9/test \
  --checkpoint weights/ml_gw/qm9_120k_best_model.pt \
  --output runs/qm9_test_predictions.csv --device cuda
```

Use `--kind crystal` with `dataset/Crystal` and `crystal_best_model.pt`, or `--kind crystal-spin` with `dataset/Crystal-spin` and `crystal_spin_best_model.pt`. The crystal command predicts **all** released samples; it does not reconstruct the historical train/test split. The CSV reports the GW correction, not the total quasiparticle energy.

## DeepH weights

Each `weights/deeph/*_best.tar.gz` contains `params/best.pytree` and `variables.json`. Extract a given archive into a separate model directory, then place the corresponding `*_train.toml` there as `train.toml` and adjust its input/output paths for your environment. The TOML files preserve architecture and basis settings, but their paths are placeholders. DeepH software, the matching ion basis, overlap matrices, and input structures are also required for inference; the archives alone cannot reproduce a band plot.

The raw DeepH training datasets are **not** included here (the source DFT/HPRO datasets total many tens of GB). Likewise, the raw DFT/GW calculations underlying the processed ML-GW data are not included. If the manuscript's data-availability statement is meant to promise those raw datasets too, deposit them separately and link that record before publication.

## Scope and evaluation caveat

The QM9 test split was curated after inspection of predictions from an earlier model, including moving conspicuous outliers into the training pool. It must not be described as a pristine, never-inspected external test set. For an unbiased generalization estimate, evaluate on a newly held-out split selected before model inspection.

The crystal scripts' default split is state/sample-wise, not by material. States from one material may appear in both train and test. Use `--split-mode system` for a stricter material-level assessment, and report which split generated each metric.

All three original training scripts use their evaluation partition for learning-rate scheduling, checkpoint selection, and early stopping. Their saved `test_mae` values are therefore model-selection scores, not independent final-test estimates. A separately untouched test set is needed for the latter.

```

No repository URL or DOI is assigned by this package. Insert the actual public URL into the manuscript only after the data and LFS objects are accessible to an unauthenticated reader. Choose a license only after confirming the rights to redistribute the derived QM9 and external-code-related materials.
