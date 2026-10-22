Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. There is no internet access.

You have 20 minutes in total. Build the most accurate solubility (`logS`) model you can from `data/solubility_train.csv`, using the command-line tool or the Python API as you prefer. Predict `logS` for the molecules in `data/solubility_test_inputs.csv` and save the predictions to `predictions.csv` with the columns `smiles` and `logS`, one row per input molecule in the same order.
