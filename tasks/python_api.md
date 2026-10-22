Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

Write a Python script `train.py` that uses Chemprop's Python API (not the command-line tool) to train a solubility (`logS`) model on `data/solubility.csv` and write predictions for `data/new_molecules.csv` to `predictions.csv`. Run it.
