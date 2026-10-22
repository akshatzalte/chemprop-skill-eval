Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

Use the `chemprop` command-line tool to train a solubility (`logS`) model on `data/solubility.csv` with the default train/validation/test split, and write the test-set RMSE of the trained model to `test_rmse.txt` as a single number. Keep the training outputs (model, logs, splits) in this directory.
