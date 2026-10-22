Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

`data/solubility_vs_temperature.csv` has solubility (`logS`) measured at different temperatures (`temperature`, in K). Train a model that uses both the molecule and the temperature as inputs to predict `logS`, then predict `logS` for the molecule and temperature pairs in `data/new_conditions.csv` and save the predictions to `predictions.csv`.
