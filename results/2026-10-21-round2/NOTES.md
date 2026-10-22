# Round 2, 2026-10-21

**Setup.** 15 tasks (8 basic, 6 pitfall tasks with correctness graders, `gpu_train` in the GPU box)
× 3 arms (`without`, `with` = the pull-request skill `9c1d44bcbcead821`, `with_short` = the shorter
variant `3613fd99f77db02f`) × 3 runs × 3 models (Haiku 5.5, Sonnet 5.5, Opus 5.5): 405 runs,
$33.74 API-equivalent. Fable was left out on purpose. Tables: [`summary.md`](summary.md).

## Results

| | Haiku | Sonnet | Opus |
|---|---|---|---|
| pass rate, without → with → with_short | 100% → 98% → 100% | 100% → 100% → 100% | 100% → 100% → 100% |
| median tokens per run | 116k → 110k (−5%) → 114k (−1%) | 75k → 82k (+9%) → 71k (−5%) | 92k → 129k (+40%) → 106k (+15%) |
| median cost per run | +2% / +2% | +19% / +13% | +22% / +3% |
| median wall time per run | −28% / −23% | −29% / −31% | −8% / −15% |
| skill used (with arms) | 39/43, 38/44 | 45/45, 45/45 | 45/45, 45/45 |

- **Pass rates don't separate the arms.** Every model solves these tasks without the skill, even the
  pitfall tasks. The one real failure (Haiku with the skill, `multitask_classification`) wrote
  `roc_auc.json` inside its training output folder instead of the working directory.
- **The skill makes runs faster** (wall time −8% to −31%), with fewer shell commands.
- **The full skill costs tokens, most on Opus** (+40%), because its text sits in the context and is
  re-read on every turn. The **short variant** keeps most of the speed-up at a much lower token cost
  (Opus +15%, Sonnet −5%), which argues for a shorter `SKILL.md` in the pull request.

## Grader and harness corrections (all applied before the numbers above)

- `fix_v1_python`: the "no v1 API" check matched **comments** such as `# equivalent of v1
  make_predictions`, which agents with the skill wrote far more often. It failed 5 Opus runs with the
  skill and none without, a bias against the skill. Code checks now ignore comments (and, for the
  v1 check, strings); the runs were re-graded and pass.
- `cluster_split`: the k-means check was case-sensitive and missed `--split KMEANS`; fixed and
  re-graded.
- 3 Haiku runs (2 `with`, 1 `with_short`) are **excluded**: a harness setting
  (`blockReadsOutsideWorkingDirectories`) refused shell commands it couldn't analyse in advance,
  which is not the agent's fault. The setting was removed for later rounds.
- Graders that read input data now use the workspace copy, because run directories drop theirs
  after grading.
