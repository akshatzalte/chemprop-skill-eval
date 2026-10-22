Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

Train a solubility (`logS`) model on `data/solubility.csv`, then use it to compute learned molecular fingerprints (the model's internal molecule embeddings) for `data/new_molecules.csv`. Save them as `fingerprints.csv`, one row per molecule.
