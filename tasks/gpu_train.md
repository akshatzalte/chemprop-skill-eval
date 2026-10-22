Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

A GPU is available in this environment. Train a model to predict aqueous solubility (`logS`) from `data/solubility.csv` on the GPU (not the CPU), save `logS` predictions for the molecules in `data/new_molecules.csv` to `predictions.csv`, and write the name of the device the model was trained on to `device.txt`.
