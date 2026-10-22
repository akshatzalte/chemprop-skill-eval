Chemprop is installed in this environment (the `chemprop` command and the Python package), and a checkout of the chemprop repository is in `./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at most 20 epochs.

`data/dyes_in_solvents.csv` has dye SMILES (`dye`), solvent SMILES (`solvent`), and the absorption maximum in nm (`absorption_max_nm`). Train a model that uses both the dye and the solvent, and save predicted absorption maxima for `data/new_dye_solvent_pairs.csv` to `predictions.csv`.
