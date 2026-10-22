"""Re-run an agent's marked-atom model with bond features built here (run with the evaluation
environment's Python, in the grader's no-network box).

usage: <env>/bin/python scripts/predict_marked.py MODEL PROBE_CSV

PROBE_CSV has `smiles` and `probe_is_marked`, a per-atom list with a 1 at an atom other than the one
the SMILES marks. The extra bond feature is 1 for bonds touching that atom, so a model that really
reads the bond feature predicts the probe atom, not the marked one. Atoms stay in SMILES order
(no reordering by atom-map number). Prints one JSON object: {"predictions": [[...], ...]} with the
first atom-level task's value for every atom, or {"error": ...}.
"""

import csv
import json
import sys

from lightning import pytorch as pl
import numpy as np
import torch

from chemprop import data, featurizers, models
from chemprop.utils import make_mol


def main(model_path: str, probe_path: str):
    model = models.MolAtomBondMPNN.load_from_file(model_path, map_location="cpu")
    with open(probe_path, newline="") as f:
        rows = list(csv.DictReader(f))
    points, n_atoms = [], []
    for row in rows:
        site = json.loads(row["probe_is_marked"]).index(1)
        mol = make_mol(row["smiles"], keep_h=False, add_h=False, reorder_atoms=False)
        E_f = np.array(
            [[float(site in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()))] for b in mol.GetBonds()]
        ).reshape(-1, 1)
        points.append(data.MolAtomBondDatapoint(mol, E_f=E_f))
        n_atoms.append(mol.GetNumAtoms())
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_bond_fdim=1)
    loader = data.build_dataloader(
        data.MolAtomBondDataset(points, featurizer), batch_size=64, shuffle=False
    )
    trainer = pl.Trainer(
        logger=False, accelerator="cpu", devices=1, enable_progress_bar=False,
        enable_model_summary=False,
    )  # fmt: skip
    with torch.inference_mode():
        batches = trainer.predict(model, loader)
    atom_preds = torch.cat([b[1] for b in batches]).reshape(sum(n_atoms), -1)[:, 0]
    split = torch.split(atom_preds, n_atoms)
    return [[round(float(x), 5) for x in values] for values in split]


if __name__ == "__main__":
    try:
        print(json.dumps({"predictions": main(*sys.argv[1:3])}))
    except Exception as e:  # noqa: BLE001 - report the failure to the grader
        print(json.dumps({"error": f"{type(e).__name__}: {e}"[:500]}))
