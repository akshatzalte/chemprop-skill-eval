# Pilot, 2026-10-20

**Setup.** 8 CPU tasks × 2 arms × 1 run, on Claude Haiku 5.5 and Sonnet 5.5 (32 runs).
Skill version `1c6f501da40d13b5` (see `skill/FROZEN_FROM.txt`), Chemprop upstream `main` at
`8d9887a2`. Concurrency 4, CPU only (8 threads per run), Claude Code 2.1.305.
Total usage: $1.23 API-equivalent. No run had isolation problems. Tables: [`summary.md`](summary.md).

## Results

- **Every task passed in both arms, on both models.** With a Chemprop checkout and `chemprop
  --help` available, these tasks are within reach of current models without the skill.
- **The baseline did not use the repository's docs.** Agents without the skill read 0 files from
  the checkout and worked from `chemprop --help` and trial and error. No agent in either arm
  tried a v1 command.
- **With the skill, agents explored less but read more.** Fewer shell commands (Haiku 42 → 35,
  Sonnet 39 → 26 over the 8 tasks), less output, and lower median wall time (Haiku −29%, Sonnet
  −12%). Median total tokens went *up* (Haiku +18%, Sonnet +10%), and median cost by 10–14%,
  because the skill's text is added to the context and re-read from cache on every turn.
- **Per-task effects go both ways.** The skill helped most where the baseline had to search:
  `uncertainty` on Sonnet (13 → 7 turns, −57% tokens, −35% cost), `chemeleon_small_data` and
  `python_api` on Haiku. It cost the most on simple tasks the baseline did in 4–6 turns
  (`mixtures`: +126–131% tokens on both models).
- The skill was used in 15 of 16 with-skill runs (Haiku `fix_v1_script` didn't invoke it).

With one run per cell these are anecdotes, not estimates; single runs vary a lot (e.g. Sonnet
`train_predict` wall time 34 s vs 127 s).

## What this means for the next round

1. **The tasks are too easy to show a pass-rate effect.** They check that files were produced, not
   that the result is right. The skill's content is mostly about mistakes that still produce
   files: predictions that overwrite the true values, descriptor columns trained as targets,
   unintended multi-GPU training, a random split where a structure split was asked for, test
   metrics that are never reported, reaction+molecule component order. Tasks should exercise these
   and graders should check correctness (e.g. the reported test RMSE matches the model, the
   requested split was used, the true values survive).
2. **Token cost of loading the skill.** On easy tasks the skill is overhead. Worth testing a
   shorter `SKILL.md` (the always-loaded part) with more moved into `references/`.
3. **More repeats** (3+ per cell) before reading anything into per-task numbers.
4. **Also run the "installed only" baseline** (no repository checkout), which is what most users
   have.

## Fixes made during the pilot

- Grader: `mixtures` only recognised `--smiles-columns`, not the short form `-s`; two runs were
  re-graded with `scripts/regrade.py` (both now pass).
- Runner: the first launch failed because the sandbox's proxy sockets live under `$TMPDIR`, and the
  per-run path was longer than the ~108-character Unix socket limit; each run now gets a short
  `/tmp/cse-*` temporary directory. Those runs were discarded and the pilot restarted.
