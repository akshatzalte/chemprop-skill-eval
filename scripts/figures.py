"""Figures for a results directory (run with the evaluation environment's Python, which has
matplotlib).

usage: <env>/bin/python scripts/figures.py results/<name> [--arm with_pr]

Writes results/<name>/figures/{headline,best_model}_{light,dark}.png:
- headline: pass rate per model without vs. with the skill, and the change with the skill in
  median tokens, cost and wall time per run (bootstrap 95% intervals, resampling runs within each
  task so both arms keep the same task mix).
- best_model: hidden-test RMSE of every best_model run, without vs. with the skill.

Colors follow the dataviz reference palette (slot 1 blue = with the skill, slot 2 orange = no
skill), validated for both modes; light and dark are separate renders, not an inversion.
"""

import argparse
import json
from pathlib import Path
import random
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.weight": "normal", "axes.titleweight": "bold", "axes.labelweight": "normal"})

MODELS = [("haiku", "Haiku 5.5"), ("sonnet", "Sonnet 5.5"), ("claude-opus-5-5", "Opus 5.5")]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", baseline="#c3c2b7", with_="#2a78d6", without="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", baseline="#383835", with_="#3987e5", without="#d95926"),
}  # fmt: skip
METRICS = [
    ("tokens", "Tokens", lambda r: r["input_tokens"] + r["output_tokens"]
     + r["cache_creation_tokens"] + r["cache_read_tokens"]),
    ("cost", "Cost", lambda r: r["cost_usd"]),
    ("wall", "Wall time", lambda r: r["wall_s"]),
]  # fmt: skip


def change(without: list[dict], with_: list[dict], value) -> float:
    """Percent change of the median per run, with the skill vs. without."""
    return 100 * (statistics.median(map(value, with_)) / statistics.median(map(value, without)) - 1)


def bootstrap(without: list[dict], with_: list[dict], value, n=2000, seed=0) -> tuple[float, float]:
    """95% interval of change(), resampling runs within each task in each arm."""
    rng = random.Random(seed)
    by_task = {}
    for arm, runs in (("without", without), ("with", with_)):
        for r in runs:
            by_task.setdefault((arm, r["task"]), []).append(r)
    stats = []
    for _ in range(n):
        sample = {"without": [], "with": []}
        for (arm, _task), runs in by_task.items():
            sample[arm] += rng.choices(runs, k=len(runs))
        stats.append(change(sample["without"], sample["with"], value))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def style_axis(ax, t, grid_axis="x"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])
    ax.tick_params(colors=t["muted"], labelcolor=t["secondary"], length=0, labelsize=10)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def headline(runs: list[dict], arm: str, t: dict, out: Path):
    fig, axes = plt.subplots(
        1, 4, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.15, 1, 1, 1], "wspace": 0.35}
    )
    fig.patch.set_facecolor(t["surface"])
    rows = list(range(len(MODELS)))[::-1]  # Haiku on top
    groups = {
        m: ([r for r in runs if r["model"] == m and r["arm"] == "without"],
            [r for r in runs if r["model"] == m and r["arm"] == arm])
        for m, _ in MODELS
    }  # fmt: skip
    # usage is only reported when a session ends: timed-out runs count in the pass rate only
    finished = {m: tuple([r for r in x if r["subtype"] == "success"] for x in g) for m, g in groups.items()}

    ax = axes[0]
    style_axis(ax, t)
    for y, (m, _) in zip(rows, MODELS):
        without, with_ = groups[m]
        a = 100 * sum(r["passed"] for r in without) / len(without)
        b = 100 * sum(r["passed"] for r in with_) / len(with_)
        ax.plot([a, b], [y, y], color=t["baseline"], linewidth=2, zorder=1)
        for x, color, label in ((a, t["without"], "No skill"), (b, t["with_"], "With skill")):
            ax.scatter([x], [y], s=70, color=color, edgecolor=t["surface"], linewidth=2,
                       zorder=3, label=label if y == rows[0] else None)  # fmt: skip
        ax.annotate(f"{a:.0f}% → {b:.0f}%", (max(a, b), y), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=t["secondary"])  # fmt: skip
    ax.set_xlim(70, 110)
    ax.set_xticks([70, 80, 90, 100])
    ax.set_xticklabels(["70%", "80%", "90%", "100%"])
    ax.set_yticks(rows)
    ax.set_yticklabels([name for _, name in MODELS], color=t["ink"], fontsize=11)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_title("Pass rate", loc="left", color=t["ink"], fontsize=12, pad=44)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2, frameon=False,
                       fontsize=9.5, handletextpad=0.3, columnspacing=1.2)  # fmt: skip
    for text in legend.get_texts():
        text.set_color(t["secondary"])

    for ax, (key, title, value) in zip(axes[1:], METRICS):
        style_axis(ax, t)
        changes = []
        for y, (m, _) in zip(rows, MODELS):
            without, with_ = finished[m]
            c = change(without, with_, value)
            lo, hi = bootstrap(without, with_, value)
            changes += [lo, hi]
            ax.barh(y, c, height=0.42, color=t["with_"], zorder=2)
            ax.plot([lo, hi], [y, y], color=t["ink"], linewidth=1.2, alpha=0.55, zorder=3)
            ha, dx = ("left", 6) if c >= 0 else ("right", -6)
            edge = hi if c >= 0 else lo
            ax.annotate(f"{c:+.0f}%", (edge, y), xytext=(dx, 0), textcoords="offset points",
                        va="center", ha=ha, fontsize=10, color=t["ink"])  # fmt: skip
        span = max(60, max(abs(v) for v in changes) + 25)
        ax.set_xlim(-span, span)
        ax.axvline(0, color=t["baseline"], linewidth=1.2, zorder=1)
        ax.set_yticks(rows)
        ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        ax.set_title(title, loc="left", color=t["ink"], fontsize=12, pad=44)
    fig.text(0.355, 0.985, "Change with the skill (median per run)", color=t["secondary"], fontsize=11,
             va="top")
    fig.text(0.01, -0.06, "Change = median per run with the skill vs. without (lower is cheaper or "
             "faster); lines are bootstrap 95% intervals. "
             "Timed-out runs count as failures and are left out of the medians.", color=t["muted"], fontsize=9)  # fmt: skip
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)


def best_model(runs: list[dict], arm: str, t: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 3.0))
    fig.patch.set_facecolor(t["surface"])
    style_axis(ax, t)
    rows = list(range(len(MODELS)))[::-1]
    for y, (m, _) in zip(rows, MODELS):
        for a, color, dy, label in (("without", t["without"], 0.18, "No skill"),
                                    (arm, t["with_"], -0.18, "With skill")):  # fmt: skip
            values = [r["metrics"]["rmse"] for r in runs if r["model"] == m and r["arm"] == a
                      and r["task"] == "best_model" and "rmse" in r["metrics"]]  # fmt: skip
            # nudge only runs whose values coincide (within 0.004), alternating above and below
            jitter = []
            for i, v in enumerate(values):
                k = sum(abs(v - w) < 0.004 for w in values[:i])
                jitter.append(0.0 if k == 0 else 0.13 * (1 if k % 2 else -1) * ((k + 1) // 2))
            ax.scatter(values, [y + dy + j for j in jitter], s=64, color=color, edgecolor=t["surface"],
                       linewidth=2, zorder=3, label=label if y == rows[0] else None)  # fmt: skip
            if values:
                med = statistics.median(values)
                ax.plot([med, med], [y + dy - 0.12, y + dy + 0.12], color=t["ink"], linewidth=2,
                        zorder=4)  # fmt: skip
    ax.set_yticks(rows)
    ax.set_yticklabels([name for _, name in MODELS], color=t["ink"], fontsize=11)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_xlabel("Hidden-test RMSE (logS), lower is better", color=t["secondary"], fontsize=10)
    ax.set_title('"Best model you can make in 20 minutes": every run', loc="left", color=t["ink"],
                 fontsize=12, pad=10)  # fmt: skip
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(t["secondary"])
    fig.text(0.01, -0.1, "Dots are runs (nudged apart when values coincide); the short vertical line marks each group's median.",
             color=t["muted"], fontsize=9)  # fmt: skip
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--arm", default="with_pr")
    args = parser.parse_args()
    runs = [json.loads(line) for line in (args.results / "runs.jsonl").read_text().splitlines()]
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    for mode, theme in THEMES.items():
        headline(runs, args.arm, theme, out / f"headline_{mode}.png")
        best_model(runs, args.arm, theme, out / f"best_model_{mode}.png")
    print(f"wrote {sorted(p.name for p in out.glob('*.png'))}")


if __name__ == "__main__":
    main()
