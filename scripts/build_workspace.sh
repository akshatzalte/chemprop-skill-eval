#!/bin/bash
# Build the workspace every eval run starts from: a checkout of chemprop (no agent skill) plus the
# task data, which is taken from chemprop's own test data and renamed to look like user files.
#
# usage: scripts/build_workspace.sh /path/to/chemprop/repo [git-ref, default origin/main]
set -euo pipefail

CHEMPROP_REPO=$(realpath "$1")
REF=${2:-origin/main}
ROOT=$(realpath "$(dirname "$0")/..")
WS=$ROOT/.workspace

rm -rf "$WS"
mkdir -p "$WS/chemprop" "$WS/data" "$WS/old_scripts"
git -C "$CHEMPROP_REPO" archive "$REF" | tar -x -C "$WS/chemprop"
git -C "$CHEMPROP_REPO" rev-parse "$REF" > "$ROOT/.workspace_chemprop_commit"

python3 - "$WS/chemprop/tests/data" "$WS/data" <<'EOF'
import csv
import sys

src, out = sys.argv[1], sys.argv[2]


def copy(inp, outp, rename=None, keep=None, rows=None):
    with open(f"{src}/{inp}") as f:
        table = list(csv.reader(f))
    header, body = table[0], table[1:][:rows]
    cols = [header.index(c) for c in keep] if keep else range(len(header))
    with open(f"{out}/{outp}", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([(rename or {}).get(header[i], header[i]) for i in cols])
        writer.writerows([row[i] for i in cols] for row in body)


copy("regression.csv", "solubility.csv", {"logSolubility": "logS"})
copy("test_smiles.csv", "new_molecules.csv")
copy("regression/mol/mol.csv", "small_lipophilicity.csv", {"lipo": "logD"}, rows=60)
copy("regression/rxn/rxn.csv", "reactions.csv", {"smiles": "reaction", "ea": "activation_energy"}, rows=80)
copy("regression/rxn/rxn.csv", "new_reactions.csv", {"smiles": "reaction"}, keep=["smiles"], rows=15)
copy("regression/mol+mol/mol+mol.csv", "dyes_in_solvents.csv", {"smiles": "dye", "peakwavs_max": "absorption_max_nm"})
copy("regression/mol+mol/mol+mol.csv", "new_dye_solvent_pairs.csv", {"smiles": "dye"}, keep=["smiles", "solvent"], rows=15)
copy("regression/test_true.csv", "measured_solubility.csv", {"logSolubility": "logS"})
copy("classification/mol.csv", "tox21_subset.csv")

# solubility at different temperatures (synthetic temperature dependence, fixed seed)
import random

random.seed(0)
with open(f"{src}/regression.csv") as f:
    rows = list(csv.reader(f))[1:]
with open(f"{out}/solubility_vs_temperature.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles", "temperature", "logS"])
    for smi, logs in rows:
        t = random.choice(range(280, 345, 5))
        writer.writerow([smi, t, round(float(logs) + 0.02 * (t - 298) + random.gauss(0, 0.1), 3)])
with open(f"{src}/test_smiles.csv") as f:
    smis = [r[0] for r in list(csv.reader(f))[1:]]
with open(f"{out}/new_conditions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles", "temperature"])
    writer.writerows([smi, 290 + 5 * i] for i, smi in enumerate(smis))
EOF

cat > "$WS/old_scripts/train_and_predict.sh" <<'EOF'
#!/bin/bash
# Trains a solubility model and predicts new molecules (written for Chemprop 1.6)
set -e
chemprop_train --data_path data/solubility.csv --dataset_type regression \
    --smiles_columns smiles --target_columns logS --save_dir old_model \
    --epochs 15 --features_generator rdkit_2d_normalized --no_features_scaling
chemprop_predict --test_path data/new_molecules.csv --checkpoint_dir old_model \
    --features_generator rdkit_2d_normalized --no_features_scaling \
    --preds_path old_predictions.csv
EOF
chmod +x "$WS/old_scripts/train_and_predict.sh"

cat > "$WS/old_scripts/train_v1.py" <<'EOF'
"""Train a solubility model and predict new molecules (written for Chemprop 1.6)."""
import chemprop

train_args = chemprop.args.TrainArgs().parse_args(
    [
        "--data_path", "data/solubility.csv",
        "--dataset_type", "regression",
        "--smiles_columns", "smiles",
        "--target_columns", "logS",
        "--save_dir", "v1_model",
        "--epochs", "15",
    ]
)
mean_score, std_score = chemprop.train.cross_validate(
    args=train_args, train_func=chemprop.train.run_training
)

predict_args = chemprop.args.PredictArgs().parse_args(
    [
        "--test_path", "data/new_molecules.csv",
        "--checkpoint_dir", "v1_model",
        "--preds_path", "v1_predictions.csv",
    ]
)
chemprop.train.make_predictions(args=predict_args)
EOF

# CheMeleon weights are pre-seeded in each run's home so runs do not depend on downloading them
CACHE=$ROOT/.cache
mkdir -p "$CACHE"
if [ ! -f "$CACHE/chemeleon_mp.pt" ]; then
    curl -fsSL -o "$CACHE/chemeleon_mp.pt" https://zenodo.org/records/15460715/files/chemeleon_mp.pt
fi


for f in "$WS"/data/*.csv; do echo "$(basename "$f"): $(($(wc -l < "$f") - 1)) rows"; done
echo "chemprop $REF = $(cat "$ROOT/.workspace_chemprop_commit")"

# round 3 has its own workspace (.workspace_r3) built from this one; rebuild it too
if [ -x "$HOME/anaconda3/envs/chemprop_skill_eval/bin/python" ]; then
    "$HOME/anaconda3/envs/chemprop_skill_eval/bin/python" "$ROOT/scripts/build_round3_data.py" > /dev/null
    echo "rebuilt .workspace_r3 (round 3)"
fi
