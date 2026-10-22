# Round 3: final task and grading list

21 tasks (12 task types; CLI and Python variants where both exist). Every prompt starts with the same
context: Chemprop is installed, a checkout of the repository is in `./chemprop`, data is in `./data`,
there is no internet access, and each training run is capped at 30 epochs (except `best_model`).
CheMeleon tasks also say where its weights are cached. Prompts are in this directory.

**Common rules for every task**
- A run passes only if **every** check passes. A grader error fails the run (it never crashes the
  eval) and is recorded.
- CLI variants (`*_cli`): `run.sh` must run `chemprop train` and `chemprop predict`.
- Python variants (`*_py`): `train.py` must import chemprop, must not run the CLI (comments are
  ignored), and the model must be saved to `model.pt`.
- Predictions are compared with **hidden labels** that never enter the agent's workspace, row by row
  in input order. Test molecules never appear anywhere in the workspace.
- Thresholds come from reference solutions (5 seeds each, CLI and Python): the larger of reference
  mean + 2 SD and reference mean + 10%, the same for both variants of a task (values in
  `hidden/thresholds.json`). The measured value is recorded for every run.
- Model architecture checks load the agent's model (the one passed to `chemprop predict` in
  `run.sh`, or `model.pt`) in a box with no network.

| # | Task | Variants | What the agent must do | Checks (all must pass) |
|---|---|---|---|---|
| 1 | `rdkit_descriptors` | CLI, Py | solubility model with RDKit 2D descriptors as extra inputs, scaled appropriately; predict 100 molecules | hidden-test RMSE ≤ **0.782**; the model has extra molecule-level inputs; they are RDKit descriptors |
| 2 | `chemeleon_hourglass` | CLI, Py | fine-tune CheMeleon on 80 molecules with an FFN of 512 → 128 → 512; predict 20 | RMSE ≤ **1.839**; message passing is CheMeleon's (2048 wide and loaded from CheMeleon); FFN hidden layers are exactly 512, 128, 512 |
| 3 | `attentive_aggregation` | Py | solubility model whose readout uses learned attention weights | RMSE ≤ **1.159**; the readout uses attention (`AttentiveAggregation`, or a custom `Aggregation` subclass using softmax) |
| 4 | `prediction_intervals` | CLI, Py | predictions with 90% intervals (`lower`, `upper`) for 100 molecules | every row has lower ≤ logS ≤ upper; coverage of the hidden labels ≥ **0.80**; median interval width ≤ **6.34** |
| 5 | `cuik_reactions` | CLI, Py | reaction model with cuik-molmaker featurization; predict 20 reactions | RMSE ≤ **2.574**; the final training uses cuik-molmaker (the `chemprop train` line in `run.sh`, or `train.py`) |
| 6 | `atom_charges` | CLI, Py | predict per-atom partial charges for 100 molecules | one numeric charge per atom for every molecule; per-atom MAE ≤ **0.0127**; charges predicted, not computed with RDKit |
| 7 | `constrained_charges` | CLI, Py | per-atom charges that sum exactly to each molecule's given `charge_sum` | as task 6 with MAE ≤ **0.0090**; every molecule's charges sum to its `charge_sum` within 5e-3 |
| 8 | `resonance_forms` | CLI | same prediction for every resonance form of a molecule (RIGR is not named) | 60 numeric predictions; within each molecule max difference ≤ 1e-4; predictions vary across molecules (SD > 0.05) |
| 9 | `stereoisomers` | CLI, Py | stereoisomers must get exactly the same prediction | as task 8, on stereoisomer pairs |
| 10 | `fix_buggy_script` | Py | find and fix a script whose predictions are poor (3 silent bugs), and run it | RMSE ≤ **1.047**; the debug subset (`head(60)`) removed; predictions converted back to logS units; prediction loader not shuffled; the script was run; no CLI used |
| 11 | `best_model` | either | the most accurate solubility model you can in 20 minutes | RMSE ≤ **1.016** (plain `chemprop train` with early stopping); RMSE recorded for comparing arms |
| 12 | `chemeleon_reactions` | CLI | fine-tune CheMeleon on reactions: **not possible** directly | a Haiku judge with a fixed rubric: passes if the final answer says CheMeleon can't be applied to reaction inputs directly (stopping, or a clearly explained alternative); fails if it claims success or silently uses another model |
| 13 | `marked_atom` | CLI, Py | each SMILES marks one atom with atom-map number 1; predict per atom 1 for the marked atom and 0 otherwise, telling the model where the mark is only through one raw (unscaled) 0/1 extra bond feature (1 for bonds touching the marked atom). Suggested by Nathan Morgan | one numeric value per atom for every molecule; the marked atom gets the highest prediction in ≥ **0.90** of test molecules; the grader re-runs the agent's model with its own bond features and the mark moved to another atom, and the moved atom must win in ≥ 0.90 (so predictions can't be read off the input SMILES); the model expects default atom features (72), default bond features + 1 (15) and no atom descriptors; the bond feature is not rescaled |

## How the graders were validated

- **Reference solutions**: for each task, a correct solution (CLI and/or Python) and a naive one
  that falls into the task's pitfall. All 28 are graded as expected: every correct one passes,
  every naive one fails for the intended reason.
- **Negative cases** for Python-only checks: a mean-aggregation model in the attention task, the
  unfixed buggy script, and a `train.py` that shells out to the CLI. All fail.
- **Invariance tasks**: the correct approach gives a max within-group difference of about 3e-7
  (RIGR) or 0 (`--ignore-stereo` at train and predict); the default featurizer gives 0.16–1.6.
- **Marked atom** (`marked_atom_*`, added after round 3): correct references score 1.00 on the test set and on the moved-mark probe (5 seeds each, CLI and Python). Negatives: chemprop's default bond-feature scaling fails only the scaling check; bond features in reversed order 0.06 / 0.17; no extra feature 0.11; the mark as an atom feature, or predictions copied from the SMILES, fail the feature-size and probe checks; in Python, `MolAtomBondDatapoint.from_smi`'s default `reorder_atoms=True` moves the mapped atom to the end and scores 0.38 / 0.17. The threshold is fixed at 0.90 rather than set from reference spreads.
- **Judge**: 4/4 sample answers (two honest, one false claim, one silent substitution) judged
  correctly.
- **Data**: no hidden test molecule appears anywhere in the agent's workspace, with or without
  its label, and none is listed twice in its dataset.

## Known limitations

- Hidden test sets are small for tasks 2 and 5 (20 molecules or reactions), so their RMSE
  thresholds are wide; those tasks rely mostly on the architecture or featurization checks.
- Task 7's naive (unconstrained) solution fails mainly on the sum check; its MAE is only slightly
  above the threshold.
- The CLI path for tasks 6 and 7 needs a placeholder `charges` column to predict (a Chemprop bug,
  kept as is: it affects both arms equally).
- The 30-epoch cap and the 20-minute budget in `best_model` are stated, not enforced.
