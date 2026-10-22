Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

`data/small_lipophilicity.csv` is a small dataset (60 molecules) of lipophilicity (`logD`). Fine-tune the CheMeleon foundation model on it, and save the test-set RMSE of the fine-tuned model as a single number in `rmse.txt`.
