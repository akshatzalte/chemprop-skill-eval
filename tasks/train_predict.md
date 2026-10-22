Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

Train a model to predict aqueous solubility (`logS`) from `data/solubility.csv`, then predict `logS` for the molecules in `data/new_molecules.csv` and save the predictions to `predictions.csv` with one row per input molecule.
