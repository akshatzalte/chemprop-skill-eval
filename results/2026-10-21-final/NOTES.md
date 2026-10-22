# Final round, 2026-10-21

**Setup.** All 21 tasks in [`tasks/round3/`](../../tasks/round3/README.md) × 2 arms (`without`,
`with_pr`, the version in the pull request, frozen in [`skill_pr/`](../../skill_pr/), tree hash
`42c3f4697cc1f718`) × 3 runs × Haiku 5.5 / Sonnet 5.5 / Opus 5.5 = 378 runs, $53.20
API-equivalent.

The 54 runs of `marked_atom_cli`, `marked_atom_py` and `best_model` come from round 3b (launched
19:17Z). That round used the same arms, skill copy, workspace and harness, and finished just
before the remaining 324 runs started (20:35Z). Results are in the top-level
[`README.md`](../../README.md), tables in [`tables.md`](tables.md), and figures in
[`figures/`](figures/).

## Corrections (applied before the numbers in the README)

- **Judge input.** The judge for `chemeleon_reactions_cli` read only the agent's last message. In
  one Opus run (without the skill), that message was a short postscript about a background helper
  that had stopped. The actual answer came one message earlier and was correct: it explained that
  CheMeleon can't load reaction features and used reactants and products as separate molecules
  instead. The judge now reads the agent's last three messages and caches each verdict under a
  hash of the text it judged. It still judges 5/5 sample answers correctly (two honest, one false
  claim, one silent substitution, and this Opus answer). Re-grading all 378 runs changed only
  that run (fail → pass).
- **Usage of timed-out runs.** Two runs timed out after 30 minutes:
  - Sonnet `marked_atom_py` with the skill, which kept experimenting with message-passing variants;
  - Sonnet `chemeleon_hourglass_py` without the skill, which was still in a long training command.

  Claude Code reports tokens and cost only when a session ends, so these two runs have none. They
  count as failures and are left out of the token, cost and time medians.

## Failures checked by hand (they stand)

- **Haiku `atom_charges_py`, with the skill (1 run).** The agent found that the labels are
  Gasteiger charges and computed them with RDKit on the test molecules to check its predictions.
  Its MAE was 0.0048, against about 0.009 for the references. The task forbids computing charges,
  so the run fails.
- **Sonnet `resonance_forms_cli`, with the skill (1 run).** The agent wrote its own resonance
  canonicalizer instead of using the RIGR featurizer. Its largest within-molecule spread was 2e-4,
  over the 1e-4 limit.
- **Sonnet `atom_charges_cli`, with the skill (3 runs).** All three trained the delivered model on
  a k-means split, following the skill's advice about stricter splits; two also used early
  stopping on that split. MAE was 0.014 against 0.008 without the skill. See the README.
- **`prediction_intervals_cli`.** Coverage was fine in every failing run (0.90–0.97); the
  intervals were wider than the threshold because the models underfit in 30 epochs (RMSE about
  1.5–1.8).
  - Haiku's four failing runs (one without the skill, three with it) all used quantile regression
    at default settings; Haiku did the same in round 3.
  - The failing Sonnet run with the skill used a mean-variance (MVE) model.
- **Haiku `resonance_forms_cli`.** In both arms Haiku canonicalized resonance forms itself, which
  worked in 2 of 3 runs without the skill and 0 of 3 with it. Two of the runs with the skill read
  the reference that mentions RIGR but did not use it.

## Other notes

- **Refused tool calls.** 17 runs (13 Haiku, 2 Sonnet, 2 Opus; 9 of them Haiku with the skill)
  had a tool call refused when they tried to read Chemprop's installed source in site-packages,
  which the file-tool deny rules block. All 17 passed. Excluding them gives Haiku −47% tokens,
  Sonnet +3%, Opus +20%, the same picture.
- **Isolation and grading.** No isolation problems and no grader errors.
- **Disk.** The run directories' checkpoints were deleted after grading, and the free-disk guard
  was never triggered.
