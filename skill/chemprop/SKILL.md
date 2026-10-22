---
name: chemprop
description: Train and use Chemprop v2 message-passing neural networks (D-MPNN graph neural networks) to predict molecular and reaction properties from SMILES (e.g. QSAR/ADMET properties such as solubility, lipophilicity, or toxicity) - regression, classification, multiclass, multitask, spectra, multicomponent mixtures (e.g. solute + solvent), atom-mapped reactions, atom/bond-level targets, uncertainty estimation, learned fingerprints, atom-level interpretation, hyperparameter optimization, and CheMeleon foundation-model fine-tuning. Use when the user wants a property-prediction model from SMILES or CSV data, wants to run, write, or debug a chemprop command or script, or mentions chemprop.
license: MIT
---

# Chemprop v2

Chemprop trains directed message-passing neural networks (D-MPNNs) on molecular graphs built
from SMILES. It has a CLI (`chemprop train|predict|fingerprint|hpopt|convert`) and a Python API
built on PyTorch Lightning. Prefer the CLI unless the user needs custom architectures, custom
training loops, or integration into existing Python code.

## Read this first: this is v2, not v1

Most examples online and in model training data are for Chemprop v1, which has a different CLI
and Python API. Do not write v1 code.

| v1 (wrong) | v2 (right) |
|---|---|
| `chemprop_train`, `chemprop_predict` | `chemprop train`, `chemprop predict` |
| `--dataset_type regression` / `--dataset-type` | `--task-type regression` (`-t`) |
| `--save_dir`, `--checkpoint_dir` | `--output-dir` (`-o`) for train, `--model-paths` for predict |
| `--features_path` | `--descriptors-path` (`.npz`) |
| `--features_generator rdkit_2d_normalized` | `--molecule-featurizers v1_rdkit_2d_normalized` |
| `--num_folds 5` | `--num-replicates 5` (`--num-folds` is an error) |
| `--gpu 0` | `CUDA_VISIBLE_DEVICES=0` with `--accelerator gpu --devices 1` (see GPUs below) |
| `from chemprop.args import TrainArgs`, `chemprop.train.run_training` | `from chemprop import data, featurizers, models, nn` + Lightning `Trainer` |

Only the subcommands `train`, `predict`, `fingerprint`, `hpopt`, and `convert` exist. Run
`chemprop <subcommand> --help` to see every option for the installed version. Old v1 model files
can be upgraded with `chemprop convert` (see `references/v1-to-v2.md`).

## Install

```text
pip install chemprop                # CLI + Python API
pip install "chemprop[hpopt]"       # adds Ray Tune, hyperopt, optuna for `chemprop hpopt`
```

Requires Python 3.11 to 3.14. PyTorch is installed as a dependency; for a specific CUDA build or a
CPU-only install, install `torch` first following https://pytorch.org. Installing into a fresh
environment is simplest. In an existing conda environment that has RDKit from conda, a plain
`pip install` can replace it and cause crashes: install the dependencies listed in
https://github.com/chemprop/chemprop/blob/main/environment.yml with conda, then
`pip install --no-deps chemprop` (or `pip install --no-deps -e .` in a source checkout). Check the
install with `chemprop --version`.

## Input data

A CSV with a header row, one SMILES column, and one or more target columns:

```text
smiles,target
CCO,-0.77
c1ccccc1O,0.08
```

- Name the columns explicitly with `--smiles-columns` and `--target-columns`. Without them the
  first column is taken as SMILES and every other column becomes a target.
- Leave a cell empty for a missing target in multitask data; it is masked out of the loss.
- Classification targets are 0/1; multiclass targets are integer class indices.
- Mixtures (one SMILES column per component), reactions (atom-mapped reaction SMILES), extra
  descriptors, and atom/bond targets are covered in `references/inputs.md`.

## Train

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --task-type regression --epochs 50 --patience 10 --output-dir runs/target
```

This splits the data 80/10/10 into train/validation/test, keeps the checkpoint with the lowest
validation loss, and evaluates it on the test split. The trained model is
`runs/target/model_0/best.pt`, and `config.toml` records every argument (rerun with
`--config-path`).

- Test metrics are only logged (lines like `test/mse: 0.71` on stderr), not saved to a file; add
  `--logfile runs/target/train.log` to keep them, and don't pass `-q`, which hides them.
- `model_0/test_predictions.csv` holds the test-split **predictions**, in columns named after the
  targets. The true values are in the input CSV, at the test indices listed in `splits.json`.

Other `--task-type` values: `classification`, `multiclass`, `spectral`, and uncertainty-aware
variants (`regression-mve`, `regression-evidential`, `regression-quantile`,
`classification-dirichlet`, `multiclass-dirichlet`). See `references/training.md`.

## Small datasets: fine-tune CheMeleon

For small datasets (roughly under a few thousand molecules), start from the CheMeleon foundation
model instead of random weights. The first run downloads the pretrained weights from Zenodo to
`~/.chemprop/`, so it needs network access.

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --task-type regression --from-foundation CheMeleon \
    --epochs 50 --patience 10 --output-dir runs/target_chemeleon
```

- CheMeleon is for molecule SMILES inputs (`--smiles-columns`), not `--reaction-columns`.
- Message-passing settings (`--depth`, `--message-hidden-dim`, `--aggregation`, ...) come from the
  foundation model. A warning listing them as ignored is printed on every run, even if you did not
  pass them; it is safe to ignore in that case.
- It cannot be combined with `--checkpoint`, a non-default `--multi-hot-atom-featurizer-mode`, or
  `--atom-features-path`/`--atom-descriptors-path`/`--bond-features-path`.

## Predict

```bash
chemprop predict --test-path new.csv --smiles-columns smiles \
    --model-paths runs/target/model_0/best.pt --preds-path runs/target/new_preds.csv
```

- The output is the input CSV plus one column per training target, named after the target
  (`target` here). **If the input already has a column with that name, it is overwritten with the
  predictions**, so keep true values under a different column name or in a separate file.
- `predict` does not read the training config. Repeat every input-related flag from training:
  `--smiles-columns`/`--reaction-columns`, `--descriptors-columns`, `--molecule-featurizers`,
  descriptor and feature files, `--rxn-mode`, `--multi-hot-atom-featurizer-mode`,
  `--keep-h`/`--add-h`.
- `--model-paths` accepts several files or a directory. A directory is searched recursively for
  `*.pt`, and all models found are averaged, with per-model predictions written to
  `<preds stem>_individual.csv`.
- Uncertainty estimates and calibration: `references/uncertainty.md`.
- Atom-level interpretation (which atoms drive a prediction): add `--callback myerson` to
  `chemprop predict` (options via `--callback-params '{"sampling_threshold": 20}'`). It works for
  `regression` and `classification` models, writes `<preds stem>_myerson_explanation.npz` (with
  `_0`, `_1`, ... per model if several are given), and is slow for molecules with many heavy atoms.

## Learned fingerprints

```bash
chemprop fingerprint --test-path new.csv --smiles-columns smiles \
    --model-paths runs/target/model_0/best.pt --ffn-block-index 0 \
    --output runs/target/new_fps.csv
```

`--ffn-block-index` is required: `0` is the learned molecule embedding right after aggregation
(with any extra molecule descriptors appended), `1` is the output of the first FFN layer, and so
on. One file is written per model, with `_0`, `_1`, ... appended to the name (here
`runs/target/new_fps_0.csv`), with columns `fp_0, fp_1, ...` and no SMILES column (rows follow the
input order).

## GPUs

`--accelerator auto --devices auto` (the defaults) use every GPU PyTorch can see, and with more
than one GPU that means multi-GPU training (DDP) whether or not you asked for it. On shared
machines, choose the GPUs explicitly:

```text
CUDA_VISIBLE_DEVICES=1 chemprop train ... --accelerator gpu --devices 1    # one specific GPU
CUDA_VISIBLE_DEVICES=0,1 chemprop train ... --accelerator gpu --devices 2  # DDP on two GPUs
chemprop train ... --accelerator cpu                                        # no GPU
```

- Check that training really runs on the GPU: the log prints `GPU available: True (cuda), used:
  True`. If it says `False`, check `nvidia-smi` and
  `python -c "import torch; print(torch.cuda.is_available())"` (a CPU-only PyTorch build is the
  usual cause).
- DDP runs one process per GPU, so log lines (including `test/...` metrics) appear once per GPU,
  and each process uses `--batch-size`, so the effective batch size is that times the number of
  GPUs. Outputs (`best.pt`, `test_predictions.csv`) are written once, as for a single GPU.
- When calling Chemprop in-process (a notebook, a test, or `chemprop.cli.main.main()`), use a
  single device (`--devices 1` or `pl.Trainer(devices=1)`); multi-process DDP can fail there.

## Pitfalls

- **Always pass `--output-dir`.** The default is a new timestamped directory under
  `chemprop_training/` in the current directory on every run.
- **`--epochs` must be larger than `--warmup-epochs` (default 2).** `--epochs 1` alone fails.
- **Early stopping is off unless you pass `--patience`.** The default runs all `--epochs`
  (default 50) and keeps the best validation checkpoint.
- **The default split is random**, which usually gives an optimistic estimate for molecules
  unlike the training data. Pick the split to match how the model will be used and tell the user
  which one you chose: `--split-type kmeans` (k-means clusters of Morgan fingerprints),
  `kennard_stone`, or `scaffold_balanced` are stricter options, each with known limitations; if the
  user has a preferred split, use it (`--splits-column` or `--splits-file` for custom splits). Use
  `--num-replicates N` to see the variance across splits (not with `kennard_stone`, which gives the
  same split every time).
- **Pass `--target-columns` whenever you use `--descriptors-columns`**, otherwise the descriptor
  columns are also trained as targets.
- External feature files (`--descriptors-path`, `--atom-features-path`, ...) are only supported
  with a single `--data-path` file, not with separate train/val/test files.
- `--num-workers` above 0 can hang on Windows and macOS; the default is 0.
- The default metric is `mse` for regression, `roc` for classification, and `multiclass-mcc` for
  multiclass. `--metrics` replaces that default rather than adding to it, so include it if you
  still want it, e.g. `--metrics mse rmse mae r2`.
- Training is not reproducible run-to-run unless you pass `--pytorch-seed`.

## References

Read these only when the task needs them:

- `references/training.md`: task types, splits, replicates and ensembles, transfer learning,
  hyperparameter optimization (`chemprop hpopt`), output layout.
- `references/inputs.md`: mixtures, reactions, extra molecule/atom/bond features and descriptors,
  atom/bond-level targets, sample weights.
- `references/uncertainty.md`: uncertainty estimation, calibration, and evaluation with
  `chemprop predict`.
- `references/python-api.md`: training and predicting from Python, including CheMeleon.
- `references/v1-to-v2.md`: migrating v1 commands, scripts, and model files.

Full documentation: https://chemprop.readthedocs.io. When this skill and the installed version
disagree, `chemprop <subcommand> --help` is authoritative.
