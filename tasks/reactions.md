Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

`data/reactions.csv` contains atom-mapped reaction SMILES (column `reaction`) and their activation energies (`activation_energy`). Train a model on it and save predicted activation energies for the reactions in `data/new_reactions.csv` to `predictions.csv`.
