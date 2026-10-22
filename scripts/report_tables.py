"""Markdown tables for the README from a results directory.

usage: python scripts/report_tables.py results/<name> [--arm with_pr]

Prints the headline table (pass rate and median tokens, cost and wall time per run, without vs.
with the skill, per model) and the per-task pass counts. Every run is included: timeouts and other
agent failures count as failures.
"""

import argparse
import json
from pathlib import Path
import statistics

MODELS = [("haiku", "Haiku 5.5"), ("sonnet", "Sonnet 5.5"), ("claude-opus-5-5", "Opus 5.5")]


def tokens(r):
    return r["input_tokens"] + r["output_tokens"] + r["cache_creation_tokens"] + r["cache_read_tokens"]


def pct(a, b):
    return f"{100 * (b / a - 1):+.0f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--arm", default="with_pr")
    args = parser.parse_args()
    runs = [json.loads(line) for line in (args.results / "runs.jsonl").read_text().splitlines()]
    arms = ("without", args.arm)

    print("| Model | Pass rate | Median tokens per run | Median cost per run | Median wall time per run |")
    print("|---|---|---|---|---|")
    for m, name in MODELS:
        a, b = ([r for r in runs if r["model"] == m and r["arm"] == arm] for arm in arms)
        pa, pb = (sum(r["passed"] for r in x) for x in (a, b))
        na, nb = len(a), len(b)
        # usage is only reported when a session ends, so runs without a result (timeouts) count
        # as failures but are left out of the token, cost and time medians
        a, b = ([r for r in x if r["subtype"] == "success"] for x in (a, b))
        ta, tb = (statistics.median(map(tokens, x)) for x in (a, b))
        ca, cb = (statistics.median(r["cost_usd"] for r in x) for x in (a, b))
        wa, wb = (statistics.median(r["wall_s"] for r in x) for x in (a, b))
        print(
            f"| {name} | {pa}/{na} → {pb}/{nb} ({100 * pa / na:.0f}% → {100 * pb / nb:.0f}%) "
            f"| {ta / 1e3:.0f}k → {tb / 1e3:.0f}k (**{pct(ta, tb)}**) "
            f"| ${ca:.3f} → ${cb:.3f} (**{pct(ca, cb)}**) "
            f"| {wa:.0f} s → {wb:.0f} s (**{pct(wa, wb)}**) |"
        )
    total = sum(r["cost_usd"] or 0 for r in runs)
    no_result = sum(r["subtype"] != "success" for r in runs)
    print(f"\n{len(runs)} runs, ${total:.2f} API-equivalent. {no_result} runs timed out without a "
          "result: they count as failures and are left out of the medians, since usage is only "
          "reported when a session ends.\n")

    tasks = sorted({r["task"] for r in runs})
    print("| Task | " + " | ".join(name for _, name in MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    for t in tasks:
        cells = []
        for m, _ in MODELS:
            counts = []
            for arm in arms:
                x = [r for r in runs if r["task"] == t and r["model"] == m and r["arm"] == arm]
                counts.append(f"{sum(r['passed'] for r in x)}/{len(x)}")
            cells.append(" → ".join(counts))
        print(f"| `{t}` | " + " | ".join(cells) + " |")
    print("\nCells: runs passed without the skill → with the skill.")


if __name__ == "__main__":
    main()
