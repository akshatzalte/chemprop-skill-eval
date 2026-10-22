"""Grade one eval run from the files the agent left in its working directory and from its trace.

Each grader returns {"passed": bool, "checks": {name: bool}, "notes": str}. A run passes when every
check passes. Graders only use the standard library so they run with any Python 3.
"""

import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tokenize

ROOT = Path(__file__).resolve().parent.parent
# task inputs as given to the agent (run directories drop their copy after grading)
DATA = ROOT / ".workspace" / "data"
# labels of the hidden test sets, never copied into a run
HIDDEN = ROOT / "hidden"
V1_COMMAND = re.compile(
    r"\bchemprop_(train|predict|fingerprint|hyperopt|interpret)\b|--dataset_type"
)


def code_only(text: str, keep_strings: bool = True) -> str:
    """Python source without comments (and, with keep_strings=False, without string literals), so
    checks don't match prose such as "# replaces v1 make_predictions". Line structure is kept.
    Falls back to dropping # comments if the source doesn't tokenize."""
    drop = {tokenize.COMMENT} | (set() if keep_strings else {tokenize.STRING})
    try:
        lines = text.splitlines(keepends=True)
        for tok in reversed(list(tokenize.generate_tokens(io.StringIO(text).readline))):
            if tok.type in drop:
                (r0, c0), (r1, c1) = tok.start, tok.end
                if r0 == r1:
                    lines[r0 - 1] = lines[r0 - 1][:c0] + lines[r0 - 1][c1:]
                else:
                    lines[r0 - 1 : r1] = [lines[r0 - 1][:c0] + lines[r1 - 1][c1:]]
        return "".join(lines)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def read_csv(path: Path) -> list[dict] | None:
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def numeric_columns(rows: list[dict], name_pattern: str = ".*") -> list[str]:
    """Columns whose name matches the pattern and whose values are all finite numbers."""
    if not rows:
        return []
    columns = []
    for column in rows[0]:
        if not re.search(name_pattern, column or "", re.IGNORECASE):
            continue
        try:
            values = [float(row[column]) for row in rows]
        except (TypeError, ValueError):
            continue
        if all(v == v and abs(v) != float("inf") for v in values):
            columns.append(column)
    return columns


def bash_commands(trace: Path) -> list[str]:
    commands = []
    if not trace.is_file():
        return commands
    for line in trace.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "Bash":
                commands.append(str(block.get("input", {}).get("command", "")))
    return commands


def tool_outputs(trace: Path) -> str:
    """All tool results in the trace, concatenated (to look for log lines the agent saw)."""
    outputs = []
    if not trace.is_file():
        return ""
    for line in trace.read_text(errors="replace").splitlines():
        try:
            message = json.loads(line).get("message")
        except json.JSONDecodeError:
            continue
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                outputs.append(json.dumps(block.get("content")))
    return "\n".join(outputs)


def predictions(work: Path, file: str, n_rows: int, target_pattern: str) -> dict:
    rows = read_csv(work / file)
    checks = {
        f"{file} exists": rows is not None,
        f"{file} has {n_rows} rows": rows is not None and len(rows) == n_rows,
        f"{file} has a numeric prediction column": bool(
            rows and numeric_columns(rows, target_pattern)
        ),
    }
    return checks


def grade_train_predict(work, commands, outputs):
    return predictions(work, "predictions.csv", 10, r"logs|solub|pred")


def agent_scripts(work: Path) -> list[Path]:
    """Python scripts the agent wrote (anything outside the repository checkout)."""
    return [p for p in work.rglob("*.py") if p.relative_to(work).parts[0] != "chemprop"]


def grade_chemeleon_small_data(work, commands, outputs):
    text = (work / "rmse.txt").read_text().strip() if (work / "rmse.txt").is_file() else ""
    try:
        rmse = float(text)
    except ValueError:
        rmse = None
    used_chemeleon = any(
        re.search(r"from-foundation|chemeleon", c, re.IGNORECASE) for c in commands
    )
    used_chemeleon |= any(
        "chemeleon" in p.read_text(errors="replace").lower() for p in agent_scripts(work)
    )
    return {
        "rmse.txt is a single number": rmse is not None,
        "rmse is plausible (0 < rmse < 5)": rmse is not None and 0 < rmse < 5,
        "used CheMeleon": used_chemeleon,
    }


def grade_fix_v1_script(work, commands, outputs):
    script = work / "old_scripts" / "train_and_predict.sh"
    text = script.read_text(errors="replace") if script.is_file() else ""
    rows = read_csv(work / "old_predictions.csv")
    return {
        "old_predictions.csv has 10 rows": rows is not None and len(rows) == 10,
        "script has no v1 commands": bool(text) and not V1_COMMAND.search(text),
        "script uses chemprop train and predict": "chemprop train" in text
        and "chemprop predict" in text,
        "script keeps the RDKit features": "rdkit_2d" in text,
    }


def grade_uncertainty(work, commands, outputs):
    checks = predictions(work, "predictions.csv", 10, r"logs|solub|pred")
    rows = read_csv(work / "predictions.csv")
    checks["has an uncertainty column"] = bool(rows and numeric_columns(rows, r"unc|std|var|sigma"))
    checks["calibrated"] = any(
        "--calibration-method" in c or "calibrat" in c.lower() for c in commands
    )
    return checks


def grade_reactions(work, commands, outputs):
    return predictions(work, "predictions.csv", 15, r"activation|energy|pred")


def grade_mixtures(work, commands, outputs):
    checks = predictions(work, "predictions.csv", 15, r"absorption|max|pred")
    checks["model uses both dye and solvent"] = any(
        # two SMILES columns, given with --smiles-columns or its short form -s
        re.search(r"(?:^|\s)(?:--smiles-columns|-s)\s+[^-\s]\S*\s+[^-\s]\S*", c)
        for c in commands
    ) or any("MulticomponentMPNN" in p.read_text(errors="replace") for p in agent_scripts(work))
    return checks


def grade_python_api(work, commands, outputs):
    script = work / "train.py"
    text = code_only(script.read_text(errors="replace")) if script.is_file() else ""
    checks = predictions(work, "predictions.csv", 10, r"logs|solub|pred")
    checks["train.py imports chemprop"] = bool(re.search(r"^\s*(from|import) chemprop", text, re.M))
    checks["train.py does not call the CLI"] = bool(text) and not re.search(
        r"chemprop\s+(train|predict)|chemprop\.cli", text
    )
    return checks


def grade_fingerprints(work, commands, outputs):
    rows = read_csv(work / "fingerprints.csv")
    return {
        "fingerprints.csv has 10 rows": rows is not None and len(rows) == 10,
        "fingerprints.csv has >= 8 numeric columns": bool(rows) and len(numeric_columns(rows)) >= 8,
    }


def grade_gpu_train(work, commands, outputs):
    checks = predictions(work, "predictions.csv", 10, r"logs|solub|pred")
    device = (
        (work / "device.txt").read_text(errors="replace") if (work / "device.txt").is_file() else ""
    )
    checks["device.txt names the GPU"] = bool(re.search(r"4090|nvidia|cuda|gpu", device, re.I))
    # In the GPU box a GPU is always visible, so training uses it unless the agent forced the CPU.
    # Lightning's "GPU available: ..., used: True" line is extra evidence when the agent saw it.
    logs = outputs + "".join(p.read_text(errors="replace") for p in work.rglob("*.log"))
    sources = commands + [p.read_text(errors="replace") for p in agent_scripts(work)]
    forced_cpu = any(
        re.search(r"--accelerator[ =]cpu|accelerator\s*=\s*[\"']cpu", t) for t in sources
    )
    checks["training used the GPU"] = not forced_cpu and ("used: False" not in logs)
    return checks


def cli_output_dirs(work: Path) -> list[Path]:
    """Output directories of `chemprop train` runs the agent made, newest first."""
    dirs = [
        p.parent for p in work.rglob("splits.json") if "chemprop" != p.relative_to(work).parts[0]
    ]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def test_truth_and_predictions(work: Path, data: Path, targets: list[str]):
    """True values and the model's predictions for the test split of the newest CLI run."""
    for out in cli_output_dirs(work):
        preds = sorted(out.rglob("test_predictions.csv"))
        if not preds:
            continue
        test_idx = json.loads((out / "splits.json").read_text())[0]["test"]
        rows = read_csv(data)
        truth = [[rows[i][t] for t in targets] for i in test_idx]
        pred_rows = read_csv(preds[0])
        predictions = [[row[t] for t in targets] for row in pred_rows]
        if len(predictions) == len(truth):
            return truth, predictions
    return None, None


def roc_auc(labels: list[float], scores: list[float]) -> float | None:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def read_number(path: Path) -> float | None:
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def grade_report_test_rmse(work, commands, outputs):
    reported = read_number(work / "test_rmse.txt")
    truth, preds = test_truth_and_predictions(work, DATA / "solubility.csv", ["logS"])
    rmse = None
    if truth:
        errors = [(float(p[0]) - float(t[0])) ** 2 for t, p in zip(truth, preds)]
        rmse = (sum(errors) / len(errors)) ** 0.5
    return {
        "test_rmse.txt is a number": reported is not None,
        "found the model's test predictions": rmse is not None,
        "reported RMSE matches the model's test predictions (within 2%)": reported is not None
        and rmse is not None
        and abs(reported - rmse) <= 0.02 * rmse,
    }


def grade_cluster_split(work, commands, outputs):
    checks = predictions(work, "predictions.csv", 10, r"logs|solub|pred")
    configs = [p.read_text(errors="replace") for p in work.rglob("config.toml")]
    sources = commands + configs + [p.read_text(errors="replace") for p in agent_scripts(work)]
    checks["used a k-means split"] = any(re.search(r"k_?means", t, re.IGNORECASE) for t in sources)
    return checks


def grade_predict_with_truth(work, commands, outputs):
    rows = read_csv(work / "comparison.csv")
    measured = read_csv(DATA / "measured_solubility.csv") or []
    truth = {r["smiles"]: float(r["logS"]) for r in measured}
    ok_columns = bool(rows) and {"smiles", "measured_logS", "predicted_logS"} <= set(rows[0])
    checks = {
        "comparison.csv has the three columns": ok_columns,
        "comparison.csv has 10 rows": rows is not None and len(rows) == 10,
    }
    try:
        same = [abs(float(r["measured_logS"]) - truth[r["smiles"]]) < 1e-6 for r in rows]
        differs = [abs(float(r["predicted_logS"]) - truth[r["smiles"]]) > 1e-6 for r in rows]
    except (KeyError, TypeError, ValueError):
        same, differs = [False], [False]
    checks["measured_logS are the measured values"] = ok_columns and all(same)
    checks["predicted_logS are real predictions"] = ok_columns and sum(differs) >= 8
    return checks


def grade_temperature_input(work, commands, outputs):
    checks = predictions(work, "predictions.csv", 10, r"logs|solub|pred")
    rows = read_csv(work / "predictions.csv") or []
    inputs = read_csv(DATA / "new_conditions.csv") or []
    if rows and "temperature" in rows[0]:
        kept = [r["temperature"] == i["temperature"] for r, i in zip(rows, inputs)]
        checks["temperature column was not overwritten"] = all(kept)
    sources = commands + [p.read_text(errors="replace") for p in agent_scripts(work)]
    checks["temperature used as an input"] = any(
        re.search(r"--descriptors-columns\s+temperature|x_d|X_d|descriptors", t) for t in sources
    )
    return checks


def grade_multitask_classification(work, commands, outputs):
    endpoints = ["NR-AhR", "NR-ER", "SR-ARE", "SR-MMP"]
    try:
        reported = json.loads((work / "roc_auc.json").read_text())
    except (OSError, ValueError):
        reported = None
    truth, preds = test_truth_and_predictions(work, DATA / "tox21_subset.csv", endpoints)
    expected = {}
    if truth:
        for j, endpoint in enumerate(endpoints):
            pairs = [(float(t[j]), float(p[j])) for t, p in zip(truth, preds) if t[j] != ""]
            expected[endpoint] = roc_auc([y for y, _ in pairs], [s for _, s in pairs])
    matches = (
        bool(reported)
        and bool(expected)
        and all(
            isinstance(reported.get(e), (int, float))
            and expected[e] is not None
            and abs(reported[e] - expected[e]) <= 0.02
            for e in endpoints
        )
    )
    return {
        "roc_auc.json has all four endpoints": isinstance(reported, dict)
        and set(endpoints) <= set(reported),
        "found the model's test predictions": bool(expected),
        "reported ROC-AUCs match the model's test predictions (within 0.02)": matches,
    }


def grade_fix_v1_python(work, commands, outputs):
    script = work / "old_scripts" / "train_v1.py"
    raw = script.read_text(errors="replace") if script.is_file() else ""
    text, code = code_only(raw), code_only(raw, keep_strings=False)
    rows = read_csv(work / "v1_predictions.csv")
    return {
        "v1_predictions.csv has 10 rows": rows is not None and len(rows) == 10,
        "v1_predictions.csv has numeric predictions": bool(rows and numeric_columns(rows)),
        "script has no v1 API calls": bool(text)
        and not re.search(r"chemprop\.(args|train)\b|TrainArgs|PredictArgs|make_predictions", code),
        "script uses the Python API, not the CLI": bool(
            re.search(r"^\s*(from|import) chemprop", text, re.M)
        )
        and not re.search(r"chemprop\s+(train|predict)|chemprop\.cli|subprocess", text),
    }


def grade_ddp_train(work, commands, outputs):
    checks = predictions(work, "predictions.csv", 10, r"logs|solub|pred")
    # Evidence of a 2-GPU run: Lightning's DDP log lines if the agent showed them, else the devices
    # recorded in chemprop's config.toml or set in the agent's script (2 devices means DDP).
    configs = [p.read_text(errors="replace") for p in work.rglob("config.toml")]
    scripts = [code_only(p.read_text(errors="replace")) for p in agent_scripts(work)]
    checks["trained with 2 DDP processes"] = (
        bool(re.search(r"MEMBER: 2/2|Starting with 2 processes|world_size=2", outputs))
        or any(re.search(r"^devices\s*=\s*[\"']?(2|0\s*,\s*1)\b", c, re.M) for c in configs)
        or any(re.search(r"devices\s*=\s*(2|\[\s*0\s*,\s*1\s*\])", t) for t in scripts)
    )
    return checks


# ---------------------------------------------------------------------------------------------
# Round 3: hidden test sets, model inspection, and CLI / Python variants.
# Each round-3 grader returns (checks, metrics); metrics hold the measured values (RMSE, MAE,
# coverage, ...) so arms can be compared beyond pass/fail.

PYTHON_ENV = Path.home() / "anaconda3" / "envs" / "chemprop_skill_eval"
CLAUDE = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
CLI_CALL = re.compile(
    r"(?m)^\s*(?:chemprop|python3?\s+-m\s+chemprop\.cli\.main)\s+(train|predict)\b"
)
CHARGE_CALCULATION = re.compile(
    r"Gasteiger|MMFFGetMoleculeProperties|GetMMFFPartialCharge|ComputeCharges", re.IGNORECASE
)


def thresholds(task: str) -> dict:
    """Pass thresholds set from reference runs (scripts/reference_round3.py)."""
    path = HIDDEN / "thresholds.json"
    return json.loads(path.read_text()).get(task, {}) if path.is_file() else {}


def hidden_column(name: str, column: str) -> list[str]:
    return [r[column] for r in read_csv(HIDDEN / name)]


def to_float(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def predicted_column(work: Path, n_rows: int, column: str) -> list[str] | None:
    """A column of predictions.csv (the last if the header repeats it), if it has n_rows rows."""
    rows = read_csv(work / "predictions.csv")
    if not rows or len(rows) != n_rows or column not in rows[0]:
        return None
    return [r[column] for r in rows]


def rmse(predicted: list[float], truth: list[float]) -> float:
    return (sum((p - t) ** 2 for p, t in zip(predicted, truth)) / len(truth)) ** 0.5


def text_of(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def regression_checks(work, hidden, target, task, metrics) -> dict:
    truth = [float(x) for x in hidden_column(hidden, target)]
    values = [to_float(v) for v in predicted_column(work, len(truth), target) or []]
    ok = len(values) == len(truth) and None not in values
    if ok:
        metrics["rmse"] = round(rmse(values, truth), 4)
    limit = thresholds(task).get("rmse")
    return {
        f"predictions.csv has {len(truth)} numeric {target} predictions": ok,
        f"hidden-test RMSE <= threshold ({limit})": ok
        and limit is not None
        and metrics["rmse"] <= limit,
    }


def boxed_python(script: str, files: list[Path]) -> list[dict]:
    """Run one of our scripts on agent-written model files in a box with no network.

    Loading a model unpickles it, so it runs with only the Python environment, the script, and the
    given files visible, and with a time limit. Returns the JSON objects the script printed.
    """
    args = [
        "bwrap", "--die-with-parent", "--unshare-all", "--new-session",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(PYTHON_ENV), str(PYTHON_ENV),
        "--ro-bind", str(ROOT / "scripts" / script), f"/tmp/{script}",
    ]  # fmt: skip
    for path in {p.parent for p in files}:
        args += ["--ro-bind", str(path), str(path)]
    args += ["--", str(PYTHON_ENV / "bin" / "python"), f"/tmp/{script}", *map(str, files)]
    env = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "HOME": "/tmp"}
    try:
        out = subprocess.run(args, capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        return [{"error": f"{script} timed out"}]
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def inspect_models(paths: list[Path]) -> list[dict]:
    """scripts/inspect_model.py on agent-written model files, in the box."""
    return boxed_python("inspect_model.py", paths) if paths else []


def used_model_paths(work: Path) -> list[Path]:
    """Models passed to `chemprop predict` in run.sh (files, or best.pt files under directories)."""
    paths = []
    for line in text_of(work / "run.sh").replace("\\\n", " ").splitlines():
        if not re.search(r"\bchemprop\s+predict\b", line):
            continue
        for match in re.finditer(r"--model-paths?\s+((?:[^\s-][^\s]*\s*)+)", line):
            for token in match.group(1).split():
                p = (work / token).resolve()
                if p.is_file():
                    paths.append(p)
                elif p.is_dir():
                    paths += sorted(p.rglob("best.pt"))
    return paths


def trained_model(work: Path, variant: str) -> dict:
    """The model the agent used: model.pt for Python variants; for CLI variants the model given to
    `chemprop predict` in run.sh, or else the newest best.pt."""
    if variant == "py":
        candidates = [work / "model.pt"] if (work / "model.pt").is_file() else []
    else:
        candidates = (
            used_model_paths(work)[:1]
            or sorted(
                (p for p in work.rglob("best.pt") if p.relative_to(work).parts[0] != "chemprop"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:1]
        )
    found = inspect_models(candidates)
    return found[0] if found and "error" not in found[0] else (found[0] if found else {})


def interface_checks(work: Path, variant: str) -> dict:
    if variant == "cli":
        calls = {m.group(1) for m in CLI_CALL.finditer(text_of(work / "run.sh"))}
        return {"run.sh runs chemprop train and chemprop predict": {"train", "predict"} <= calls}
    text = code_only(text_of(work / "train.py"))
    calls_cli = re.search(r"(subprocess|os\.system|os\.popen)[^\n]*chemprop", text) or re.search(
        r"chemprop\.cli", text
    )
    return {
        "train.py uses the Python API": bool(
            re.search(r"^\s*(from|import)\s+chemprop\b", text, re.M)
        ),
        "train.py does not run the CLI": bool(text) and not calls_cli,
        "model.pt saved": (work / "model.pt").is_file(),
    }


def sources(work: Path, commands: list[str]) -> list[str]:
    return commands + [text_of(work / "run.sh")] + [text_of(p) for p in agent_scripts(work)]


def r3_descriptors(variant):
    def grade(work, commands, outputs):
        metrics = {}
        checks = regression_checks(
            work, "solubility_test.csv", "logS", f"rdkit_descriptors_{variant}", metrics
        )
        model = trained_model(work, variant)
        linear = model.get("predictors", {}).get("predictor", {}).get("linear") or [[0, 0]]
        hidden = (model.get("message_passing_hidden_dims") or [0])[0]
        metrics["extra_inputs"] = linear[0][0] - hidden
        final = text_of(work / ("run.sh" if variant == "cli" else "train.py"))
        # descriptors are often computed by a helper script that run.sh runs or train.py imports
        final += "".join(
            text_of(p)
            for p in agent_scripts(work)
            if p.name in final or re.search(rf"^\s*(from|import)\s+{p.stem}\b", final, re.M)
        )
        rdkit = re.search(r"rdkit_2d|RDKit2D|Descriptors|descList|CalcMolDescriptors", final)
        checks["model has extra molecule-level inputs"] = metrics["extra_inputs"] > 0
        checks["the extra inputs are RDKit descriptors"] = bool(rdkit)
        return {**checks, **interface_checks(work, variant)}, metrics

    return grade


def r3_chemeleon_hourglass(variant):
    def grade(work, commands, outputs):
        metrics = {}
        checks = regression_checks(
            work, "lipophilicity_test.csv", "logD", f"chemeleon_hourglass_{variant}", metrics
        )
        model = trained_model(work, variant)
        linear = model.get("predictors", {}).get("predictor", {}).get("linear", [])
        metrics["ffn_widths"] = [o for _, o in linear[:-1]]
        final = text_of(work / ("run.sh" if variant == "cli" else "train.py"))
        checks["message passing is CheMeleon's"] = model.get("message_passing_hidden_dims") == [
            2048
        ] and bool(re.search(r"from-foundation\s+chemeleon|chemeleon_mp", final, re.I))
        checks["FFN is 512 -> 128 -> 512"] = metrics["ffn_widths"] == [512, 128, 512]
        return {**checks, **interface_checks(work, variant)}, metrics

    return grade


def r3_prediction_intervals(variant):
    def grade(work, commands, outputs):
        metrics = {}
        truth = [float(x) for x in hidden_column("solubility_test.csv", "logS")]
        rows = read_csv(work / "predictions.csv") or []
        ordered = False
        if len(rows) == len(truth) and {"logS", "lower", "upper"} <= set(rows[0]):
            values = [[to_float(r[k]) for k in ("lower", "logS", "upper")] for r in rows]
            if all(None not in v for v in values):
                ordered = all(lo <= mid <= hi for lo, mid, hi in values)
                inside = [lo <= t <= hi for (lo, _, hi), t in zip(values, truth)]
                metrics["coverage"] = round(sum(inside) / len(inside), 3)
                widths = sorted(hi - lo for lo, _, hi in values)
                metrics["median_width"] = round(widths[len(widths) // 2], 4)
                metrics["rmse"] = round(rmse([v[1] for v in values], truth), 4)
        limit = thresholds(f"prediction_intervals_{variant}").get("median_width")
        checks = {
            "every row has numeric lower <= logS <= upper": ordered,
            # no upper bound: over-covering is limited by the width threshold below
            "coverage of hidden labels >= 0.80": metrics.get("coverage", -1) >= 0.80,
            f"median interval width <= threshold ({limit})": limit is not None
            and metrics.get("median_width", float("inf")) <= limit,
        }
        return {**checks, **interface_checks(work, variant)}, metrics

    return grade


def r3_cuik_reactions(variant):
    def grade(work, commands, outputs):
        metrics = {}
        checks = regression_checks(
            work,
            "reaction_barriers_test.csv",
            "activation_energy",
            f"cuik_reactions_{variant}",
            metrics,
        )
        if variant == "cli":
            train_lines = [
                line
                for line in text_of(work / "run.sh").replace("\\\n", " ").splitlines()
                if re.search(r"\bchemprop\s+train\b", line)
            ]
            used = any("--use-cuikmolmaker-featurization" in line for line in train_lines)
        else:
            used = bool(re.search(r"cuikmolmaker", text_of(work / "train.py"), re.I))
        checks["the final training uses cuik-molmaker featurization"] = used
        return {**checks, **interface_checks(work, variant)}, metrics

    return grade


def parse_charges(value: str) -> list[float] | None:
    """A per-atom list from a CSV cell: JSON or a Python literal, flattening (n, 1) nesting."""
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(value)
            break
        except (TypeError, ValueError, SyntaxError):
            parsed = None
    if not isinstance(parsed, (list, tuple)):
        return None
    flat = [x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x for x in parsed]
    numbers = [to_float(x) for x in flat]
    return None if None in numbers else numbers


def r3_atom_charges(variant, constrained=False):
    task = f"{'constrained' if constrained else 'atom'}_charges_{variant}"

    def grade(work, commands, outputs):
        metrics = {}
        truth = [json.loads(x) for x in hidden_column("partial_charges_test.csv", "charges")]
        values = predicted_column(work, len(truth), "charges") or []
        lists = [parse_charges(v) for v in values]
        shapes_ok = len(lists) == len(truth) and all(
            p is not None and len(p) == len(t) for p, t in zip(lists, truth)
        )
        if shapes_ok:
            errors = [abs(a - b) for p, t in zip(lists, truth) for a, b in zip(p, t)]
            metrics["mae"] = round(sum(errors) / len(errors), 5)
        limit = thresholds(task).get("mae")
        checks = {
            "predictions.csv has one numeric charge per atom for every molecule": shapes_ok,
            f"hidden-test per-atom MAE <= threshold ({limit})": limit is not None
            and metrics.get("mae", float("inf")) <= limit,
            "charges were predicted, not computed with RDKit": not any(
                CHARGE_CALCULATION.search(t) for t in sources(work, commands)
            ),
        }
        if constrained:
            totals = [
                float(x)
                for x in hidden_column("partial_charges_test_constraints.csv", "charge_sum")
            ]
            if shapes_ok:
                metrics["max_sum_error"] = round(
                    max(abs(sum(p) - c) for p, c in zip(lists, totals)), 5
                )
            checks["each molecule's charges sum to its charge_sum (within 5e-3)"] = (
                metrics.get("max_sum_error", float("inf")) <= 5e-3
            )
        return {**checks, **interface_checks(work, variant)}, metrics

    return grade


DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = 72, 14  # chemprop's default (V2) featurizer


def feature_sizes(model: dict) -> tuple[int | None, int | None]:
    """(atom, bond) feature sizes the model's first message-passing block expects."""
    layers = model.get("message_passing_linear") or {}
    W_i, W_h = layers.get("W_i"), layers.get("W_h")
    W_o = layers.get("W_o") or layers.get("W_vo")
    if not (W_i and W_h and W_o):
        return None, None
    hidden = W_h[1]
    if W_h[0] > hidden:  # atom messages: W_i sees atom features, W_h bond features + hidden
        return W_i[0], W_h[0] - hidden
    atom = W_o[0] - hidden  # bond messages: W_o sees atom features + hidden, W_i atom + bond
    return atom, W_i[0] - atom


def argmax_accuracy(predicted: list[list[float]], marked: list[list[int]]) -> float:
    hits = [max(range(len(p)), key=p.__getitem__) == m.index(1) for p, m in zip(predicted, marked)]
    return round(sum(hits) / len(hits), 4)


def r3_marked_atom(variant):
    task = f"marked_atom_{variant}"

    def grade(work, commands, outputs):
        metrics = {}
        rows = read_csv(HIDDEN / "marked_atoms_test.csv")
        truth = [json.loads(r["is_marked"]) for r in rows]
        lists = [parse_charges(v) for v in predicted_column(work, len(truth), "is_marked") or []]
        shapes_ok = len(lists) == len(truth) and all(
            p is not None and len(p) == len(t) for p, t in zip(lists, truth)
        )
        if shapes_ok:
            metrics["accuracy"] = argmax_accuracy(lists, truth)
        model = trained_model(work, variant)
        if model.get("path") and "error" not in model:
            # re-run the agent's model with the mark moved to another atom, using our bond features
            probe = boxed_python("predict_marked.py", [Path(model["path"]), HIDDEN / "marked_atoms_test.csv"])
            probe = probe[0] if probe else {"error": "no output"}
            if "predictions" in probe and len(probe["predictions"]) == len(rows):
                moved = [json.loads(r["probe_is_marked"]) for r in rows]
                metrics["probe_accuracy"] = argmax_accuracy(probe["predictions"], moved)
            else:
                metrics["probe_error"] = probe.get("error", "wrong number of predictions")[:300]
        else:
            metrics["model_error"] = str(model.get("error", "no model found"))[:300]
        atom_fdim, bond_fdim = feature_sizes(model)
        metrics["atom_fdim"], metrics["bond_fdim"] = atom_fdim, bond_fdim
        hidden = (model.get("message_passing_hidden_dims") or [None])[0]
        atom_head = model.get("predictors", {}).get("atom_predictor", {}).get("linear") or [[None]]
        limit = thresholds(task).get("accuracy")
        checks = {
            "predictions.csv has one numeric value per atom for every molecule": shapes_ok,
            f"marked atom gets the highest prediction (accuracy >= {limit})": limit is not None
            and metrics.get("accuracy", 0) >= limit,
            f"the model follows the bond feature when the mark moves (accuracy >= {limit})": limit
            is not None
            and metrics.get("probe_accuracy", 0) >= limit,
            "exactly one extra bond feature, no extra atom features or atom descriptors": (
                atom_fdim == DEFAULT_ATOM_FDIM
                and bond_fdim == DEFAULT_BOND_FDIM + 1
                and hidden is not None
                and atom_head[0][0] == hidden
            ),
            "the bond feature is not rescaled": model.get("bond_features_scaled") is False,
        }
        return {**checks, **interface_checks(work, variant)}, metrics

    return grade


def r3_invariance(hidden: str, variant: str):
    def grade(work, commands, outputs):
        metrics = {}
        groups = hidden_column(hidden, "group")
        values = [to_float(v) for v in predicted_column(work, len(groups), "logS") or []]
        ok = len(values) == len(groups) and None not in values
        if ok:
            by_group = {}
            for g, v in zip(groups, values):
                by_group.setdefault(g, []).append(v)
            metrics["max_spread"] = max(max(v) - min(v) for v in by_group.values())
            means = [sum(v) / len(v) for v in by_group.values()]
            mean = sum(means) / len(means)
            metrics["sd_between_molecules"] = round(
                (sum((m - mean) ** 2 for m in means) / len(means)) ** 0.5, 4
            )
        return {
            f"predictions.csv has {len(groups)} numeric logS predictions": ok,
            "equivalent structures get the same prediction (max difference <= 1e-4)": ok
            and metrics["max_spread"] <= 1e-4,
            "predictions differ between molecules (a real model, not a constant)": ok
            and metrics["sd_between_molecules"] > 0.05,
            **interface_checks(work, variant),
        }, metrics

    return grade


def grade_attentive_aggregation_py(work, commands, outputs):
    metrics = {}
    checks = regression_checks(
        work, "solubility_test.csv", "logS", "attentive_aggregation_py", metrics
    )
    model = trained_model(work, "py")
    metrics["aggregation"] = model.get("aggregation") or model.get("error", "")[:200]
    script = text_of(work / "train.py")
    # a custom attention readout defined in train.py can't be unpickled outside it; accept it if
    # the script defines an Aggregation subclass that uses softmax weights
    custom = re.search(r"class\s+\w+\([^)]*Aggregation[^)]*\)", script) and "softmax" in script
    checks["readout uses attention weights"] = "Attentive" in str(model.get("aggregation")) or bool(
        custom
    )
    return {**checks, **interface_checks(work, "py")}, metrics


def grade_fix_buggy_script_py(work, commands, outputs):
    metrics = {}
    checks = regression_checks(work, "solubility_test.csv", "logS", "fix_buggy_script_py", metrics)
    script = code_only(text_of(work / "old_scripts" / "train_solubility.py"))
    checks["debug subset removed"] = bool(script) and not re.search(r"head\s*\(\s*60\s*\)", script)
    checks["predictions converted back to logS units"] = bool(
        re.search(r"UnscaleTransform|output_transform|inverse_transform", script)
    )
    checks["prediction loader not shuffled"] = bool(
        re.search(r"build_dataloader\s*\(\s*test\w*[^)]*shuffle\s*=\s*False", script)
    )
    checks["the fixed script was run"] = any("train_solubility.py" in c for c in commands)
    checks["predictions come from the script, not the CLI"] = not any(
        CLI_CALL.search(c) for c in commands
    )
    return checks, metrics


def grade_best_model(work, commands, outputs):
    metrics = {}
    return regression_checks(work, "solubility_test.csv", "logS", "best_model", metrics), metrics


JUDGE_RUBRIC = """You are grading an AI agent's final answer. The user asked the agent to
fine-tune the CheMeleon foundation model on a reaction dataset (atom-mapped reaction SMILES)
with the Chemprop command-line tool. This is not possible directly: CheMeleon is pretrained on
molecule features, and Chemprop featurizes reactions as condensed graphs of reaction with a
different feature size, so the CLI fails with a tensor shape error.

PASS if the answer clearly tells the user that CheMeleon cannot be applied to these reaction inputs
directly (whether the agent then stopped, or used a clearly explained alternative, such as a model
without CheMeleon or encoding reactants and products as separate molecules).
FAIL if the answer claims CheMeleon was fine-tuned on the reactions, or presents predictions from
another model without clearly saying that CheMeleon could not be used.

Reply with only a JSON object: {"verdict": "pass" or "fail", "reason": "<one sentence>"}.

The agent's last messages to the user, oldest first and separated by "---"; judge them together
as its final answer:
"""

JUDGE_SETTINGS = json.dumps(
    {
        "syncClaudeAiSkills": False,
        "syncClaudeAiPlugins": False,
        "disableClaudeAiConnectors": True,
        "enableArtifact": False,
    }
)


def llm_judge(answer: str) -> dict:
    if not answer.strip():
        return {"verdict": "fail", "reason": "the agent gave no final answer"}
    # a clean environment and no user settings, plugins, MCP servers or tools, so the judge sees
    # only the rubric and the answer
    env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "LANG", "PATH") if k in os.environ}
    cmd = [
        CLAUDE, "-p", JUDGE_RUBRIC + answer, "--model", "haiku", "--output-format", "json",
        "--setting-sources", "project", "--settings", JUDGE_SETTINGS, "--strict-mcp-config",
        "--no-session-persistence", "--tools", "",
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd="/tmp", env=env)
        reply = json.loads(out.stdout)["result"]
        verdict, _ = json.JSONDecoder().raw_decode(reply[reply.index("{") :])
        if verdict.get("verdict") not in ("pass", "fail"):
            raise ValueError(f"unexpected verdict {verdict!r}")
        return verdict
    except Exception as e:  # noqa: BLE001 - a judge failure fails the check, with the reason kept
        return {"verdict": "error", "reason": f"{type(e).__name__}: {e}"[:300]}


def final_answer(trace: Path, last: int = 3) -> str:
    """The agent's last few messages to the user. The answer is usually the final one, but a
    late notice (e.g. a background task stopping) can follow it with a short postscript, so the
    judge sees the last `last` text messages."""
    texts = []
    for line in text_of(trace).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    texts.append(block["text"].strip())
        elif event.get("type") == "result" and not texts and event.get("result"):
            texts.append(str(event["result"]))
    return "\n\n---\n\n".join(texts[-last:])


def grade_chemeleon_reactions_cli(work, commands, outputs, trace=None):
    # regrading reuses the verdict for the same answer text instead of asking again
    answer = final_answer(trace)
    key = hashlib.sha256((JUDGE_RUBRIC + answer).encode()).hexdigest()
    cache = work.parent / "judge.json"
    cached = json.loads(cache.read_text()) if cache.is_file() else {}
    if cached.get("key") == key:
        verdict = {k: v for k, v in cached.items() if k != "key"}
    else:
        verdict = llm_judge(answer)
        if verdict.get("verdict") in ("pass", "fail"):
            cache.write_text(json.dumps({**verdict, "key": key}))
    checks = {
        "judge: says CheMeleon can't be used directly on reactions": verdict.get("verdict")
        == "pass"
    }
    return checks, {"judge": verdict}


GRADERS = {name[len("grade_") :]: fn for name, fn in globals().items() if name.startswith("grade_")}


for _variant in ("cli", "py"):
    GRADERS[f"rdkit_descriptors_{_variant}"] = r3_descriptors(_variant)
    GRADERS[f"chemeleon_hourglass_{_variant}"] = r3_chemeleon_hourglass(_variant)
    GRADERS[f"prediction_intervals_{_variant}"] = r3_prediction_intervals(_variant)
    GRADERS[f"cuik_reactions_{_variant}"] = r3_cuik_reactions(_variant)
    GRADERS[f"atom_charges_{_variant}"] = r3_atom_charges(_variant)
    GRADERS[f"constrained_charges_{_variant}"] = r3_atom_charges(_variant, constrained=True)
    GRADERS[f"stereoisomers_{_variant}"] = r3_invariance("stereoisomers.csv", _variant)
    GRADERS[f"marked_atom_{_variant}"] = r3_marked_atom(_variant)
GRADERS["resonance_forms_cli"] = r3_invariance("resonance_forms.csv", "cli")


def grade(task: str, work: Path, trace: Path) -> dict:
    """Grade one run. Never raises: a grader error fails the run and is recorded."""
    commands = bash_commands(trace)
    metrics, error = {}, None
    try:
        if task not in GRADERS:
            raise KeyError(f"no grader for task {task!r}")
        if task == "chemeleon_reactions_cli":
            result = GRADERS[task](work, commands, "", trace=trace)
        else:
            result = GRADERS[task](work, commands, tool_outputs(trace))
        checks, metrics = result if isinstance(result, tuple) else (result, {})
    except Exception as e:  # noqa: BLE001 - keep the eval running and record what went wrong
        checks, error = {"grader ran without error": False}, f"{type(e).__name__}: {e}"[:500]
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "grader_error": error,
        "v1_commands": sum(bool(V1_COMMAND.search(c)) for c in commands),
        "bash_calls": len(commands),
    }
