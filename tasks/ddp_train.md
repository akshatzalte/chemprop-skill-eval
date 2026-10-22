Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

Two GPUs are available in this environment. Train a solubility (`logS`) model on `data/solubility.csv` using both GPUs with distributed data-parallel training, then predict `logS` for `data/new_molecules.csv` and save the predictions to `predictions.csv`.
