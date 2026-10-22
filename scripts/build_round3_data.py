"""Build the data for the round-3 tasks (run with the evaluation environment's Python, which has RDKit).

usage: <env>/bin/python scripts/build_round3_data.py

Builds .workspace_r3/, the workspace every round-3 run starts from: the chemprop checkout from
.workspace/ *without* its tests/data directory, and only the round-3 task files in data/ and
old_scripts/. The test-set labels go to hidden/, which is never copied into a run. The task data is
derived from chemprop's tests/data, so leaving that directory (or the round-1/2 data files, which
come from the same source) in the workspace would leak the hidden labels. Fixed seeds throughout,
so rebuilding gives identical files. Run scripts/build_workspace.sh first.
"""

import csv
import json
from pathlib import Path
import random
import re
import shutil

from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".workspace" / "chemprop" / "tests" / "data"
WORKSPACE = ROOT / ".workspace_r3"
DATA = WORKSPACE / "data"
HIDDEN = ROOT / "hidden"

if WORKSPACE.exists():
    shutil.rmtree(WORKSPACE)
shutil.copytree(
    ROOT / ".workspace" / "chemprop",
    WORKSPACE / "chemprop",
    ignore=lambda d, names: ["data"] if Path(d).name == "tests" else [],
)
# notebook outputs print data tables (molecules with labels); clear them, keep code and text
for nb in WORKSPACE.rglob("*.ipynb"):
    notebook = json.loads(nb.read_text())
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"], cell["execution_count"] = [], None
    nb.write_text(json.dumps(notebook, indent=1) + "\n")
# never use a molecule that appears anywhere in the checkout (docs, examples) as a test molecule
CHECKOUT_TEXT = "\n".join(
    p.read_text(errors="replace")
    for p in (WORKSPACE / "chemprop").rglob("*")
    if p.is_file() and p.suffix in {".csv", ".ipynb", ".rst", ".md", ".py", ".txt", ".json"}
)


def exposed(identifier: str) -> bool:
    return re.search(r"(?<![^\s,\"'(\[>|])" + re.escape(identifier) + r"(?![^\s,\"')\]<|])", CHECKOUT_TEXT) is not None


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split(rows: list[dict], n_test: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Random train/test split; test rows are drawn only from molecules not shown in the checkout."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    key = next(k for k in ("smiles", "reaction") if k in rows[0])
    counts = {}
    for r in rows:
        counts[r[key]] = counts.get(r[key], 0) + 1
    # not shown in the checkout, and not listed twice (a duplicate could land in train and test)
    eligible = [r for r in rows if not exposed(r[key]) and counts[r[key]] == 1]
    test = eligible[:n_test]
    chosen = {id(r) for r in test}
    return [r for r in rows if id(r) not in chosen], test


def labelled_task(name: str, rows: list[dict], inputs: list[str], targets: list[str], n_test: int):
    """<name>_train.csv and <name>_test_inputs.csv for agents; hidden/<name>_test.csv for graders."""
    train, test = split(rows, n_test, seed=0)
    write(DATA / f"{name}_train.csv", train, inputs + targets)
    write(DATA / f"{name}_test_inputs.csv", test, inputs)
    write(HIDDEN / f"{name}_test.csv", test, inputs + targets)
    return train, test


# solubility: A1 descriptors, B2 attentive aggregation, C1 intervals, J1 buggy script, K1 best model,
# and the training data for G1 (resonance) and H1 (stereo)
solubility = [
    {"smiles": r["smiles"], "logS": r["logSolubility"]} for r in read(SRC / "regression.csv")
]
sol_train, sol_test = labelled_task("solubility", solubility, ["smiles"], ["logS"], n_test=100)

# lipophilicity: B1 CheMeleon with an hourglass FFN
lipo = [{"smiles": r["smiles"], "logD": r["lipo"]} for r in read(SRC / "regression/mol/mol.csv")]
labelled_task("lipophilicity", lipo, ["smiles"], ["logD"], n_test=20)

# reactions: E1 cuik-molmaker
rxn = [
    {"reaction": r["smiles"], "activation_energy": r["ea"]}
    for r in read(SRC / "regression/rxn/rxn.csv")
]
labelled_task("reaction_barriers", rxn, ["reaction"], ["activation_energy"], n_test=20)

# atom-level partial charges: F1, and F4 with per-molecule total-charge constraints
charges = []
for r in solubility:
    mol = Chem.MolFromSmiles(r["smiles"])
    smi = Chem.MolToSmiles(mol)  # canonical, so a canonical round trip keeps the atom order
    mol = Chem.MolFromSmiles(smi)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    q = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
    if mol.GetNumAtoms() < 2 or any(x != x or abs(x) == float("inf") for x in q):
        continue
    q = [round(x, 4) for x in q]
    charges.append({"smiles": smi, "charges": json.dumps(q), "charge_sum": round(sum(q), 4)})
train, test = labelled_task("partial_charges", charges, ["smiles"], ["charges"], n_test=100)
write(DATA / "partial_charges_train_constraints.csv", train, ["charge_sum"])
write(DATA / "partial_charges_test_constraints.csv", test, ["charge_sum"])
write(HIDDEN / "partial_charges_test_constraints.csv", test, ["smiles", "charge_sum"])

# resonance pairs for G1: two non-equivalent resonance forms of the same molecule
# one pair per distinct molecule; forms compared as canonical SMILES so they are really different
pairs, seen = [], set()
for r in solubility:
    mol = Chem.MolFromSmiles(r["smiles"])
    parent = Chem.MolToSmiles(mol)
    forms = []
    for res in Chem.ResonanceMolSupplier(mol, Chem.ALLOW_CHARGE_SEPARATION):
        smi = Chem.MolToSmiles(res)
        if Chem.MolFromSmiles(smi) is not None and Chem.CanonSmiles(smi) not in map(Chem.CanonSmiles, forms):
            forms.append(smi)
    if len(forms) >= 2 and parent not in seen:
        seen.add(parent)
        pairs.append(forms[:2])
pairs = pairs[:30]
rows = [{"smiles": smi, "group": i} for i, pair in enumerate(pairs) for smi in pair]
random.Random(1).shuffle(rows)
write(DATA / "resonance_forms.csv", rows, ["smiles"])
write(HIDDEN / "resonance_forms.csv", rows, ["smiles", "group"])

# stereoisomer pairs for H1
pairs, seen = [], set()
options = StereoEnumerationOptions(onlyUnassigned=False, maxIsomers=4)
for r in solubility:
    parent = Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"]), isomericSmiles=False)
    if parent in seen:
        continue
    isomers = list(
        dict.fromkeys(
            Chem.MolToSmiles(m)
            for m in EnumerateStereoisomers(Chem.MolFromSmiles(r["smiles"]), options=options)
        )
    )
    if len(isomers) >= 2:
        seen.add(parent)
        pairs.append(isomers[:2])
pairs = pairs[:30]
rows = [{"smiles": smi, "group": i} for i, pair in enumerate(pairs) for smi in pair]
random.Random(2).shuffle(rows)
write(DATA / "stereoisomers.csv", rows, ["smiles"])
write(HIDDEN / "stereoisomers.csv", rows, ["smiles", "group"])

# marked atoms for L1 (suggested by Nathan Morgan): one atom per molecule carries atom-map number 1,
# and the model may learn which one only from an extra bond feature. Atoms are shuffled so the mark
# is not tied to a position. The grader also re-runs the agent's model with the mark moved to
# another atom (probe_is_marked, bond features built by the grader), so predictions can't be read
# off the input SMILES.
def marked(rows: list[dict], seed: int) -> list[dict]:
    rng, out = random.Random(seed), []
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol.GetNumAtoms() < 4 or len(Chem.GetMolFrags(mol)) > 1:
            continue
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        mol = Chem.RenumberAtoms(mol, order)
        mol.GetAtomWithIdx(rng.randrange(mol.GetNumAtoms())).SetAtomMapNum(1)
        smi = Chem.MolToSmiles(mol, canonical=False)
        target = [int(a.GetAtomMapNum() == 1) for a in Chem.MolFromSmiles(smi).GetAtoms()]
        probe = rng.choice([i for i, t in enumerate(target) if not t])
        out.append(
            {
                "smiles": smi,
                "is_marked": json.dumps(target),
                "probe_is_marked": json.dumps([int(i == probe) for i in range(len(target))]),
            }
        )
    return out


write(DATA / "marked_atoms_train.csv", marked(sol_train, seed=3), ["smiles"])
marked_test = marked(sol_test, seed=4)
write(DATA / "marked_atoms_test_inputs.csv", marked_test, ["smiles"])
write(HIDDEN / "marked_atoms_test.csv", marked_test, ["smiles", "is_marked", "probe_is_marked"])

for path in sorted(DATA.glob("*")) + sorted(HIDDEN.glob("*")):
    with open(path) as f:
        print(f"{path.relative_to(ROOT)}: {sum(1 for _ in f) - 1} rows")

# J1: a training script with three silent bugs (each verified to hurt the hidden-test RMSE)
BUGGY_SCRIPT = '''"""Train a solubility model and predict new molecules."""
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn

train_df = pd.read_csv("data/solubility_train.csv")
train_df = train_df.head(60)  # quick debug run
points = [data.MoleculeDatapoint.from_smi(s, y) for s, y in zip(train_df["smiles"], train_df[["logS"]].values)]
tr, va, te = data.make_split_indices([p.mol for p in points], "random", (0.8, 0.1, 0.1))
tr, va, te = data.split_data_by_indices(points, tr, va, te)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_set = data.MoleculeDataset(tr[0], featurizer)
scaler = train_set.normalize_targets()
val_set = data.MoleculeDataset(va[0], featurizer)
val_set.normalize_targets(scaler)

mp = nn.BondMessagePassing()
model = models.MPNN(mp, nn.MeanAggregation(), nn.RegressionFFN())
trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="auto", devices=1, max_epochs=30)
trainer.fit(model, data.build_dataloader(train_set), data.build_dataloader(val_set, shuffle=False))

test_df = pd.read_csv("data/solubility_test_inputs.csv")
test_set = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in test_df["smiles"]], featurizer)
with torch.inference_mode():
    preds = trainer.predict(model, data.build_dataloader(test_set))
test_df["logS"] = torch.cat(preds)[:, 0].numpy()
test_df.to_csv("predictions.csv", index=False)
'''
(WORKSPACE / "old_scripts").mkdir(parents=True, exist_ok=True)
(WORKSPACE / "old_scripts" / "train_solubility.py").write_text(BUGGY_SCRIPT)
