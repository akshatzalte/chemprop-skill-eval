# Round 3, 2026-10-21

**Setup.** The 19 harder tasks in [`tasks/round3/`](../../tasks/round3/README.md) × 3 arms
(`without`, `with` = the PR skill, `with_short` = the shorter variant) × 3 runs × Haiku 5.5 /
Sonnet 5.5 / Opus 5.5 = 513 runs, $70.86 API-equivalent. Tables: [`summary.md`](summary.md)
(excludes the 9 runs below with a refused tool call).

## Results

All 513 runs, without → with → with_short:

| | Haiku | Sonnet | Opus |
|---|---|---|---|
| passed | 53 → 52 → 50 of 57 | 55 → 55 → 56 | 56 → 56 → 57 |
| median tokens | 427k → 222k → 236k | 94k → 103k → 96k | 164k → 195k → 188k |
| median cost | $0.016 → 0.011 → 0.011 | $0.084 → 0.099 → 0.096 | $0.264 → 0.288 → 0.277 |
| median wall time | 120 → 99 → 113 s | 62 → 46 → 50 s | 96 → 112 → 89 s |

- **Pass rates are flat.** These tasks are solvable without the skill by all three models; the
  differences are 1–3 runs out of 57 and go both ways.
- **Haiku uses about half the tokens and a third less money with the skill**, as in the DDP task.
  Sonnet and Opus pay for reading the skill (+10–20% cost) and gain little on these tasks.
- The short variant costs Sonnet and Opus a little less than the full one and does at least as
  well; the PR now uses it.

## `best_model`: the skill narrowed what strong models tried

| hidden-test RMSE (median of 3) | without | with | with_short |
|---|---|---|---|
| Haiku | 0.59 | 0.56 | 0.59 |
| Sonnet | 0.58 | 0.71 | 0.71 |
| Opus | 0.52 | 0.71 | 0.72 |

All runs pass the threshold (1.016), so pass rates hide this. With the skill, Sonnet and Opus
fine-tuned CheMeleon (the skill said "For small datasets ... start from the CheMeleon foundation
model") and stopped there. Without it, they tried RDKit descriptors (Sonnet) or descriptors plus an
ExtraTrees model (Opus), which do better on this solubility set. One dataset, where descriptors are
unusually strong, but a clear case of the skill anchoring agents. The PR's CheMeleon section now
says to compare CheMeleon with a plain model and one with RDKit descriptors instead of starting
from it.

## Other failures

- `prediction_intervals_cli` (Haiku 2 → 1 → 0 of 3): in both arms Haiku chose quantile regression
  at default settings, which underfits in 30 epochs (RMSE ≈ 1.76) and gives intervals wider than the
  threshold. Coverage is fine. Not caused by the skill.
- `constrained_charges_cli` (Haiku with_short 1 of 3) and `rdkit_descriptors_py` (Haiku with 1 of 3):
  MAE/RMSE slightly over the threshold; scattered across arms.
- No other pattern: failures are single runs spread across arms.

## Harness corrections

- **10 Sonnet runs lost to the disk-full crash** (03:50–03:54Z): Claude Code reported "the temp
  filesystem ... is full (0MB free)", which an earlier search for "No space left" missed. They ended
  without a result; the records were removed (backup: `runs.jsonl.before-tmpfull-cleanup`) and the
  runs repeated (all 10 passed, $1.18).
- **Grader false-fail in `rdkit_descriptors_cli`**: the "extra inputs are RDKit descriptors" check
  only searched `run.sh`, so 3 correct runs (all `without`) that computed descriptors in a helper
  script `run.sh` calls were failed. The check now also reads scripts that `run.sh` names or
  `train.py` imports; all runs were re-graded and only those 3 changed.
- **9 runs with a refused tool call** (8 Haiku charge tasks, 1 Opus): agents tried to read
  chemprop's installed source in site-packages, which is under the home directory that the file-tool
  deny rules block (the `./chemprop` checkout was readable). Both arms are affected; `summary.md`
  excludes these runs, the table above includes them. Excluding them changes no conclusion. A later
  round should allow reading the environment's site-packages.
- Isolation: 0 problems in 513 runs. Grader errors: 0.
