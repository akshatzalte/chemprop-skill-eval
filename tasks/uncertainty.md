Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

Train a solubility (`logS`) model on `data/solubility.csv` that also estimates the uncertainty of its predictions, calibrate those uncertainties, and save predictions with their uncertainties for `data/new_molecules.csv` to `predictions.csv`.
