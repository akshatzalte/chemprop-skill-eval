Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

Train a solubility (`logS`) model on `data/solubility.csv`, but split the data 80/10/10 so that structurally similar molecules stay in the same split: cluster the molecules by their Morgan fingerprints with k-means and keep each cluster in one split. Then predict `logS` for `data/new_molecules.csv` and save the predictions to `predictions.csv`.
