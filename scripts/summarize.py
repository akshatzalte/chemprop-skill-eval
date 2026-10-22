"""Summarize results/<name>/runs.jsonl into runs.csv and summary.md.

usage: python scripts/summarize.py results/<name>

Tokens are the session totals reported by Claude Code (input + output + cache creation + cache
reads). Cost is Claude Code's API-equivalent estimate, which is what the run would cost at API
prices; on a subscription it is a measure of usage, not a bill.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

TOKEN_KEYS = ["input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens"]


def denied_tool_calls(run: dict) -> int:
    """Tool calls the harness refused (in round 2, a read-block setting refused some legitimate
    shell commands; that setting is gone, but those runs are excluded as harness-affected)."""
    trace = Path(run.get("run_dir", "")) / "trace.jsonl"
    for line in reversed(trace.read_text(errors="replace").splitlines() if trace.is_file() else []):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            return len(event.get("permission_denials") or [])
    return 0


def load(results: Path) -> list[dict]:
    runs = [json.loads(line) for line in (results / "runs.jsonl").read_text().splitlines()]
    for run in runs:
        run["total_tokens"] = sum(run.get(k) or 0 for k in TOKEN_KEYS)
        run["denied_tool_calls"] = denied_tool_calls(run)
        run["valid"] = not run.get("isolation_problems") and not run["denied_tool_calls"]
    return runs


def fmt_pair(without: float, with_: float, spec: str) -> str:
    change = f" ({(with_ - without) / without:+.0%})" if without else ""
    return f"{without:{spec}} → {with_:{spec}}{change}"


def table(runs: list[dict], group_keys: list[str], skill_arm: str) -> str:
    def stat(rows, key, fn):
        values = [r[key] for r in rows if r.get(key) is not None]
        return fn(values) if values else 0.0

    header = group_keys + [
        "runs",
        "pass rate",
        "median tokens",
        "median cost $",
        "median turns",
        "median wall time s",
        "skill used",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    groups = sorted({tuple(r[k] for k in group_keys) for r in runs})
    for group in groups:
        rows = [r for r in runs if tuple(r[k] for k in group_keys) == group]
        arm = {"without": [r for r in rows if r["arm"] == "without"]}
        arm["with"] = [r for r in rows if r["arm"] == skill_arm]
        if not arm["with"] or not arm["without"]:
            continue
        used = sum(r["skill_invoked"] or r["skill_files_read"] > 0 for r in arm["with"])
        cells = list(group) + [
            f"{len(arm['without'])} / {len(arm['with'])}",
            fmt_pair(
                stat(arm["without"], "passed", mean), stat(arm["with"], "passed", mean), ".0%"
            ),
            fmt_pair(
                stat(arm["without"], "total_tokens", median),
                stat(arm["with"], "total_tokens", median),
                ",.0f",
            ),
            fmt_pair(
                stat(arm["without"], "cost_usd", median),
                stat(arm["with"], "cost_usd", median),
                ".3f",
            ),
            fmt_pair(
                stat(arm["without"], "num_turns", median),
                stat(arm["with"], "num_turns", median),
                ".0f",
            ),
            fmt_pair(
                stat(arm["without"], "wall_s", median), stat(arm["with"], "wall_s", median), ".0f"
            ),
            f"{used}/{len(arm['with'])}",
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    runs = load(args.results)
    invalid = [r for r in runs if not r["valid"]]
    runs = [r for r in runs if r["valid"]]

    columns = [
        "model", "task", "arm", "run", "passed", "num_turns", "cost_usd", "total_tokens",
        *TOKEN_KEYS, "duration_s", "wall_s", "skill_invoked", "skill_files_read",
        "repo_files_read", "bash_calls", "v1_commands", "exit_code", "subtype", "is_error",
    ]  # fmt: skip
    with open(args.results / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns + ["checks"], extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow({**r, "checks": json.dumps(r.get("checks"))})

    sections = []
    for skill_arm in [a for a in ("with", "with_short") if any(r["arm"] == a for r in runs)]:
        sections += [
            f"# Without skill → `{skill_arm}` (relative change)",
            "## By model\n\n" + table(runs, ["model"], skill_arm),
            "## By model and task\n\n" + table(runs, ["model", "task"], skill_arm),
        ]
    excluded = (
        f"Excluded runs: {len(invalid)} (isolation problems or tool calls refused by the harness: "
        + ", ".join(f"{r['model']} {r['task']} {r['arm']}" for r in invalid)
        + ")"
        if invalid
        else "Excluded runs: none"
    )
    text = "\n\n".join(sections + [excluded])
    (args.results / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
