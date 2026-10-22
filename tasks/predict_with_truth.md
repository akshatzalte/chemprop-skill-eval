Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

`data/measured_solubility.csv` has molecules with measured solubility (`logS`). Train a solubility model on `data/solubility.csv`, predict these molecules, and save `comparison.csv` with the columns `smiles`, `measured_logS` (the measured values from the file) and `predicted_logS`.
