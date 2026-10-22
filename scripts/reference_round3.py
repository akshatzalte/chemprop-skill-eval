"""Reference (correct) and naive solutions for the round-3 tasks, used to validate the graders and to
set the pass thresholds.

usage: python scripts/reference_round3.py OUT_DIR [--seeds 5] [--tasks ...]
       python scripts/reference_round3.py OUT_DIR --grade [--python-refs DIR]

The first form runs each CLI reference solution with several seeds, and each naive solution once,
in OUT_DIR/<task>/<kind>-<seed>/, exactly as an agent's working directory would look (run.sh,
predictions.csv, the trained model). The second form writes hidden/thresholds.json from the
correct solutions (mean + 2 SD over seeds) and then grades every correct and naive solution with the
real graders; every correct solution must pass and every naive one must fail. --python-refs points
to Python-variant reference directories (each with train.py, predictions.csv, model.pt and
predictions_seed<N>.csv) to include.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import grade  # noqa: E402

ENV_BIN = Path.home() / "anaconda3" / "envs" / "chemprop_skill_eval" / "bin"
ENV = {
    "PATH": f"{ENV_BIN}:/usr/bin:/bin",
    "HOME": str(Path.home()),
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
}
C = "--accelerator cpu -q"

# helper run inside a solution directory: chemprop predict for atom-level models needs the target
# column in the input (a placeholder works), and writes it twice; keep the predicted one
ATOM_PREDICT = """python - <<'EOF'
import csv
rows = list(csv.DictReader(open("data/partial_charges_test_inputs.csv")))
with open("test_with_placeholder.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["smiles", "charges"]); w.writerows([r["smiles"], "[]"] for r in rows)
EOF
chemprop predict -i test_with_placeholder.csv -s smiles --model-paths out/model_0/best.pt \\
    -o raw_predictions.csv {extra} {C}
python - <<'EOF'
import csv
r = csv.reader(open("raw_predictions.csv")); header = next(r)
w = csv.writer(open("predictions.csv", "w", newline="")); w.writerow(["smiles", "charges"])
w.writerows([row[0], row[-1]] for row in r)
EOF"""


# writes the per-atom 0/1 targets (also a placeholder column for predict) and the 0/1 bond feature
MARKED_HELPER = """cat > make_marked.py <<'EOF'
import csv, json, sys
import numpy as np
from rdkit import Chem
src, out_csv, out_npz = sys.argv[1:4]
E = []
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["smiles", "is_marked"])
    for r in csv.DictReader(open(src)):
        mol = Chem.MolFromSmiles(r["smiles"])
        mark = [a.GetAtomMapNum() == 1 for a in mol.GetAtoms()]
        E.append(np.array([[float(mark[b.GetBeginAtomIdx()] or mark[b.GetEndAtomIdx()])] for b in mol.GetBonds()]).reshape(-1, 1))
        w.writerow([r["smiles"], json.dumps([int(m) for m in mark])])
np.savez(out_npz, *E)
EOF"""

def select_columns(src: str, dst: str, columns: str) -> str:
    return f"python -c \"import pandas as pd; pd.read_csv('{src}')[{columns}].to_csv('{dst}', index=False)\""


def solutions(seed: int) -> dict[str, dict[str, str]]:
    """task -> {"correct": script, "naive": script}; each script is the run.sh an agent would write."""
    s = f"--pytorch-seed {seed} --data-seed {seed}"
    sol_train = "-i data/solubility_train.csv -s smiles --target-columns logS --epochs 30"
    sol_pred = "-i data/solubility_test_inputs.csv -s smiles --model-paths out/model_0/best.pt"
    out = {}
    feats = "--molecule-featurizers v1_rdkit_2d_normalized --no-descriptor-scaling"
    out["rdkit_descriptors_cli"] = {
        "correct": f"chemprop train {sol_train} {feats} -o out {s} {C}\n"
        f"chemprop predict {sol_pred} {feats} -o predictions.csv {C}",
        "naive": f"chemprop train {sol_train} -o out {s} {C}\n"
        f"chemprop predict {sol_pred} -o predictions.csv {C}",
    }
    lipo = "-i data/lipophilicity_train.csv -s smiles --target-columns logD --epochs 30 --from-foundation CheMeleon"
    lipo_pred = "chemprop predict -i data/lipophilicity_test_inputs.csv -s smiles --model-paths out/model_0/best.pt -o predictions.csv"
    out["chemeleon_hourglass_cli"] = {
        "correct": f"chemprop train {lipo} --ffn-hidden-dim 512 128 512 --ffn-num-layers 3 -o out {s} {C}\n{lipo_pred} {C}",
        "naive": f"chemprop train {lipo} --ffn-hidden-dim 512 --ffn-num-layers 3 -o out {s} {C}\n{lipo_pred} {C}",
    }
    # 90% intervals: mean-variance model, conformal calibration on a held-out 10% of the training set
    split = (
        "python -c \"import pandas as pd; d = pd.read_csv('data/solubility_train.csv').sample(frac=1, "
        f"random_state={seed}); d[40:].to_csv('fit.csv', index=False); d[:40].to_csv('cal.csv', index=False)\""
    )
    intervals_pred = (
        f"chemprop predict {sol_pred} --uncertainty-method mve --cal-path cal.csv "
        "--calibration-method conformal-regression --conformal-alpha 0.1 -o raw.csv " + C
    )
    to_bounds = (
        "python -c \"import pandas as pd; d = pd.read_csv('raw.csv'); "
        "d['lower'] = d.logS - d.logS_unc; d['upper'] = d.logS + d.logS_unc; "
        "d[['smiles', 'logS', 'lower', 'upper']].to_csv('predictions.csv', index=False)\""
    )
    naive_bounds = (  # one standard deviation as the half-width: only ~68% coverage
        f"chemprop predict {sol_pred} --uncertainty-method mve -o raw.csv {C}\n"
        "python -c \"import pandas as pd; d = pd.read_csv('raw.csv'); s = d.logS_unc ** 0.5; "
        "d['lower'] = d.logS - s; d['upper'] = d.logS + s; "
        "d[['smiles', 'logS', 'lower', 'upper']].to_csv('predictions.csv', index=False)\""
    )
    mve_train = "chemprop train -i fit.csv -s smiles --target-columns logS -t regression-mve --epochs 30 -o out"
    out["prediction_intervals_cli"] = {
        "correct": f"{split}\n{mve_train} {s} {C}\n{intervals_pred}\n{to_bounds}",
        "naive": f"{split}\n{mve_train} {s} {C}\n{naive_bounds}",
    }
    rxn = "-i data/reaction_barriers_train.csv --reaction-columns reaction --target-columns activation_energy --keep-h --epochs 30"
    rxn_pred = (
        "chemprop predict -i data/reaction_barriers_test_inputs.csv --reaction-columns reaction --keep-h "
        "--model-paths out/model_0/best.pt -o predictions.csv"
    )
    cuik = "--use-cuikmolmaker-featurization"
    out["cuik_reactions_cli"] = {
        "correct": f"chemprop train {rxn} {cuik} -o out {s} {C}\n{rxn_pred} {cuik} {C}",
        "naive": f"chemprop train {rxn} -o out {s} {C}\n{rxn_pred} {C}",
    }
    q = "-i data/partial_charges_train.csv -s smiles --atom-target-columns charges --epochs 30"
    cons_train = "--constraints-path data/partial_charges_train_constraints.csv --constraints-to-targets charges"
    cons_test = "--constraints-path data/partial_charges_test_constraints.csv --constraints-to-targets charges"
    out["atom_charges_cli"] = {
        "correct": f"chemprop train {q} -o out {s} {C}\n" + ATOM_PREDICT.format(extra="", C=C),
        "naive": f"chemprop train {q} --add-h -o out {s} {C}\n"
        + ATOM_PREDICT.format(extra="--add-h", C=C),
    }
    out["constrained_charges_cli"] = {
        "correct": f"chemprop train {q} {cons_train} -o out {s} {C}\n"
        + ATOM_PREDICT.format(extra=cons_test, C=C),
        "naive": f"chemprop train {q} -o out {s} {C}\n" + ATOM_PREDICT.format(extra="", C=C),
    }
    stereo_pred = "chemprop predict -i data/stereoisomers.csv -s smiles --model-paths out/model_0/best.pt -o predictions.csv"
    out["stereoisomers_cli"] = {
        "correct": f"chemprop train {sol_train} --ignore-stereo -o out {s} {C}\n{stereo_pred} --ignore-stereo {C}",
        "naive": f"chemprop train {sol_train} --ignore-stereo -o out {s} {C}\n{stereo_pred} {C}",
    }
    rigr = "--multi-hot-atom-featurizer-mode RIGR"
    res_pred = "chemprop predict -i data/resonance_forms.csv -s smiles --model-paths out/model_0/best.pt -o predictions.csv"
    out["resonance_forms_cli"] = {
        "correct": f"chemprop train {sol_train} {rigr} -o out {s} {C}\n{res_pred} {rigr} {C}",
        "naive": f"chemprop train {sol_train} -o out {s} {C}\n{res_pred} {C}",
    }
    # marked atom: a 0/1 bond feature (bonds touching the atom with map number 1) and 0/1 atom
    # targets, built by a helper; the naive solution lets chemprop rescale the bond feature
    marked_train = (
        f"chemprop train -i marked_train.csv -s smiles --atom-target-columns is_marked "
        f"--bond-features-path marked_train.npz --epochs 30 -o out {s} {C}"
    )
    marked_pred = (
        "chemprop predict -i marked_test.csv -s smiles --model-paths out/model_0/best.pt "
        f"--bond-features-path marked_test.npz -o predictions.csv {C}"
    )
    marked = f"{MARKED_HELPER}\npython make_marked.py data/marked_atoms_train.csv marked_train.csv marked_train.npz\n"
    marked += "python make_marked.py data/marked_atoms_test_inputs.csv marked_test.csv marked_test.npz\n"
    out["marked_atom_cli"] = {
        "correct": f"{marked}{marked_train} --no-bond-feature-scaling\n{marked_pred}",
        "naive": f"{marked}{marked_train}\n{marked_pred}",
    }
    # best model: plain chemprop with its defaults (50 epochs) and early stopping
    out["best_model"] = {
        "correct": "chemprop train -i data/solubility_train.csv -s smiles --target-columns logS "
        f"--patience 10 -o out {s} {C}\nchemprop predict {sol_pred} -o predictions.csv {C}",
        "naive": "chemprop train -i data/solubility_train.csv -s smiles --target-columns logS "
        f"--epochs 3 --warmup-epochs 1 -o out {s} {C}\nchemprop predict {sol_pred} -o predictions.csv {C}",
    }
    return out


def prepare(work: Path):
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copytree(ROOT / ".workspace_r3" / "data", work / "data")
    shutil.copytree(ROOT / ".workspace_r3" / "old_scripts", work / "old_scripts")


def run(out_dir: Path, seeds: list[int], tasks: list[str] | None):
    for seed in seeds:
        for task, kinds in solutions(seed).items():
            if tasks and task not in tasks:
                continue
            for kind, script in kinds.items():
                if kind == "naive" and seed > 0:
                    continue
                work = out_dir / task / f"{kind}-{seed}"
                prepare(work)
                (work / "run.sh").write_text("set -e\n" + script + "\n")
                proc = subprocess.run(
                    ["bash", "run.sh"], cwd=work, env=ENV, capture_output=True, text=True
                )
                (work / "run.log").write_text(proc.stdout + proc.stderr)
                print(f"{task:26} {kind:8} seed {seed}: exit {proc.returncode}", flush=True)


def metric(task: str, work: Path) -> dict:
    """The quantity each performance threshold is set on, for one correct solution."""
    if task.startswith(("atom_charges", "constrained_charges")):
        truth = [json.loads(x) for x in grade.hidden_column("partial_charges_test.csv", "charges")]
        rows = grade.read_csv(work / "predictions.csv")
        pred = [json.loads(r["charges"]) for r in rows]
        errors = [abs(float(a) - b) for p, t in zip(pred, truth) for a, b in zip(p, t)]
        return {"mae": sum(errors) / len(errors)}
    if task.startswith("prediction_intervals"):
        rows = grade.read_csv(work / "predictions.csv")
        widths = sorted(float(r["upper"]) - float(r["lower"]) for r in rows)
        return {"median_width": widths[len(widths) // 2]}
    hidden, column = {
        "chemeleon_hourglass": ("lipophilicity_test.csv", "logD"),
        "cuik_reactions": ("reaction_barriers_test.csv", "activation_energy"),
    }.get(task.rsplit("_", 1)[0], ("solubility_test.csv", "logS"))
    truth = [float(x) for x in grade.hidden_column(hidden, column)]
    pred = [float(r[column]) for r in grade.read_csv(work / "predictions.csv")]
    return {"rmse": grade.rmse(pred, truth)}


def predictions_for_seeds(task: str, out_dir: Path, python_refs: Path | None) -> list[Path]:
    dirs = sorted((out_dir / task).glob("correct-*")) if (out_dir / task).is_dir() else []
    if python_refs and (python_refs / task).is_dir():
        ref = python_refs / task
        dirs = [ref]
        for p in sorted(ref.glob("predictions_seed*.csv")):
            d = out_dir / "_python_seeds" / task / p.stem
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, d / "predictions.csv")
            dirs.append(d)
    return dirs


# thresholds not derived from reference spreads: marked-atom accuracy (references score 1.00, a
# wrong bond order 0.18, no bond feature 0.11)
FIXED_THRESHOLDS = {"marked_atom_cli": {"accuracy": 0.9}, "marked_atom_py": {"accuracy": 0.9}}


NO_THRESHOLD = {
    "stereoisomers_cli",
    "stereoisomers_py",
    "resonance_forms_cli",
    "chemeleon_reactions_cli",
    *FIXED_THRESHOLDS,
}


def limit(values: list[float]) -> float:
    """Pass threshold from reference values: the larger of mean + 2 SD and mean + 10%.

    The SD over seeds of one fixed recipe is small; agents legitimately choose other settings
    (aggregation, layers, epochs), so a relative margin keeps reasonable solutions from failing.
    """
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return max(mean + 2 * sd, mean * 1.10)


def set_thresholds(out_dir: Path, python_refs: Path | None, tasks: list[str]):
    """hidden/thresholds.json from the reference solutions. A task with both a CLI and a Python
    variant uses the more lenient of the two variants' references for both, since either way of
    solving it is legitimate."""
    stats = {}
    for task in tasks:
        if task in NO_THRESHOLD:
            continue
        metrics = [metric(task, d) for d in predictions_for_seeds(task, out_dir, python_refs)]
        if metrics:
            stats[task] = {k: [m[k] for m in metrics] for k in metrics[0]}
    result = {}
    for task in tasks:
        if task in NO_THRESHOLD:
            continue
        base, variant = task.rsplit("_", 1)
        sibling = f"{base}_{'py' if variant == 'cli' else 'cli'}"
        sources = {t: stats[t] for t in (task, sibling) if t in stats}
        if not sources:
            continue
        entry = {}
        for key in next(iter(sources.values())):
            entry[key] = round(max(limit(v[key]) for v in sources.values() if key in v), 4)
            for name, values in sources.items():
                entry[f"{key}_reference_{name}"] = {
                    "mean": round(statistics.mean(values[key]), 4),
                    "sd": round(statistics.stdev(values[key]), 4) if len(values[key]) > 1 else 0.0,
                    "seeds": len(values[key]),
                }
        result[task] = entry
    result.update({t: v for t, v in FIXED_THRESHOLDS.items() if t in tasks})
    path = ROOT / "hidden" / "thresholds.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {path}")
    return result


def grade_all(out_dir: Path, python_refs: Path | None):
    tasks = sorted(grade.GRADERS)
    round3 = [t for t in tasks if (ROOT / "tasks" / "round3" / f"{t}.md").is_file()]
    set_thresholds(out_dir, python_refs, round3)
    rows = []
    for task in round3:
        dirs = []
        if (out_dir / task).is_dir():
            dirs += [("correct", d) for d in sorted((out_dir / task).glob("correct-0"))]
            dirs += [("naive", d) for d in sorted((out_dir / task).glob("naive-0"))]
        if python_refs and (python_refs / task).is_dir():
            dirs.append(("correct", python_refs / task))
            if (python_refs / task / "naive").is_dir():
                dirs.append(("naive", python_refs / task / "naive"))
        for kind, work in dirs:
            run_sh = work / "run.sh"
            commands = run_sh.read_text().splitlines() if run_sh.is_file() else []
            if (work / "commands.txt").is_file():  # Python references: what the agent would run
                commands = (work / "commands.txt").read_text().splitlines()
            outputs = (work / "run.log").read_text() if (work / "run.log").is_file() else ""
            result = grade.GRADERS[task](work, commands, outputs)
            checks, metrics = result if isinstance(result, tuple) else (result, {})
            ok = bool(checks) and all(checks.values())
            failed = [k for k, v in checks.items() if not v]
            rows.append((task, kind, ok, ok == (kind == "correct"), failed, metrics))
    print()
    for task, kind, ok, as_expected, failed, metrics in rows:
        flag = "ok " if as_expected else "BAD"
        print(f"[{flag}] {task:26} {kind:8} passed={ok!s:5} {metrics} failed: {failed or '-'}")
    print(f"\n{sum(r[3] for r in rows)}/{len(rows)} solutions graded as expected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--python-refs", type=Path)
    args = parser.parse_args()
    if args.grade:
        grade_all(args.out_dir, args.python_refs)
    else:
        run(args.out_dir, args.seeds, args.tasks)


if __name__ == "__main__":
    main()
