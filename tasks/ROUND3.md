# Round 3 task list (proposal)

Rounds 1 and 2 showed that current models pass basic and simple-pitfall Chemprop tasks with or
without the skill (round 2 so far: 100% pass rate in every arm on Haiku). Round 3 makes the tasks
harder and grades the *quality* of the result, not only that output files exist.

Nothing here is built yet. This is the list to agree on before writing tasks, data, and graders.

## Principles

1. **CLI and Python versions.** Where a feature exists in both, each task has a CLI variant (the
   agent writes and runs a bash script, `run.sh`) and a Python variant (the agent writes and runs
   `train.py`). Graders check that the right interface was used.
2. **Hidden test sets.** The agent gets training data and *unlabelled* test inputs. True labels stay
   with the grader, outside the agent's sandbox. This allows grading on performance.
3. **Thresholds from reference runs.** For each performance-graded task, a reference solution (the
   standard way to do the task, written by us) is trained with 5 seeds, as a CLI and, where the task
   has one, as a Python solution. A run passes if its hidden-test error is at most the larger of
   *reference mean + 2 SD* and *reference mean + 10%*, using the more lenient of the two variants'
   references. (First set as mean + 2 SD alone; changed because the SD over seeds of one fixed
   recipe is tiny, about 0.02 RMSE, while agents legitimately choose other settings: a correct
   Python cuik solution failed the plain rule.) The measured error is recorded for every run, so
   arms can also be compared on it directly.
4. **Model inspection.** Graders may load the agent's saved model (`best.pt` / `.ckpt`) with Chemprop
   in the evaluation environment, to check architecture claims (FFN widths, aggregation class,
   featurizer dimensions, frozen weights). Graders run after the agent has finished, outside its
   sandbox.
5. **Not tuned to the skill.** Tasks come from what users do, not from what the skill covers. The
   "skill covers it today" column is for interpreting results; gaps found here are fixed in the
   skill separately and re-measured.
6. **Repeats.** 3 runs per cell for pass/fail tasks; 5 for performance-graded tasks, which are noisier.

Difficulty: **M** = moderately harder than round 2, **H** = hard, **O** = open-ended.
Skill coverage today: ✅ covered, ◐ partly (a line or two), ✗ not covered.

## A. Extra features and descriptors

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| A1 `rdkit_descriptors` | CLI, Py | Add RDKit 2D descriptors as extra inputs, scaled sensibly, then predict new molecules | precompute + scale vs `--molecule-featurizers`; `--descriptors-path` row order; `--no-descriptor-scaling`; in Python, `x_d`, `X_d_transform`, predictor `input_dim`, scaling the validation set too | hidden-test RMSE ≤ threshold; model input dim = hidden size + number of descriptors (inspect) | M | ◐ |
| A2 `atom_features` | CLI, Py | Add per-atom features (e.g. Gasteiger charges) before message passing | one array per molecule in `.npz`, atoms in RDKit order; `extra_atom_fdim` in Python | model atom feature dim = default + n (inspect); predictions for 10 molecules | M | ◐ |
| A3 `conditions_and_descriptors` | CLI | Molecule + temperature + pressure columns as inputs, plus Morgan counts | `--descriptors-columns` with `--target-columns`; `--molecule-featurizers morgan_count` with `--no-descriptor-scaling`; repeating all flags at predict | inputs not overwritten in the output; descriptors used (inspect input dim); hidden-test RMSE | M | ✅ |

## B. Foundation model and architecture

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| B1 `chemeleon_hourglass` | CLI, Py | Fine-tune CheMeleon with an hourglass FFN (e.g. 512 → 128 → 512) | `--ffn-hidden-dim 512 128 512` vs `--ffn-num-layers` (hidden layers only); Python: `input_dim=mp.output_dim`, building the FFN, loading weights | inspect: message passing initialised from CheMeleon, FFN widths as asked; hidden-test RMSE | H | ◐ |
| B2 `attentive_aggregation` | Py | Train with attention-based aggregation instead of mean/norm | not available in the CLI; `AttentiveAggregation(output_size=...)` is not in the registry | inspect: aggregation class is `AttentiveAggregation`; trains and predicts | H | ✗ |
| B3 `transfer_frozen_encoder` | CLI, Py | Pretrain on 500 molecules, fine-tune on 60 with the message-passing layers frozen | `--checkpoint --freeze-encoder`; scalers come from the checkpoint; Python freezing (`requires_grad`, `.eval()` for batch norm) | inspect: encoder weights equal the pretrained model's, FFN weights changed; hidden-test RMSE | M | ✅ |

| B4 `chemeleon_reactions` | CLI | Fine-tune CheMeleon on a reaction dataset. **Not possible directly**: CheMeleon expects molecule features (86) and reactions use condensed-graph-of-reaction features (134); the CLI fails with an unexplained shape error (`mat1 and mat2 shapes cannot be multiplied (244x134 and 86x2048)`) | diagnosing the error instead of hacking around it; saying so honestly | pass if the final answer states that CheMeleon can't be applied to reaction inputs directly (stopping, or using a clearly explained alternative such as encoding reactants and products as molecules); fail if it claims CheMeleon fine-tuning on the reactions worked or silently uses another model. Judged by a small LLM grader (Haiku) with a fixed rubric, checked by hand in the pilot | H | ✗ |

## C. Uncertainty

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| C1 `conformal_intervals` | CLI, Py | 90% prediction intervals for new molecules | which estimator + calibrator + evaluator work together; a held-out calibration set; interval width vs half-width | coverage of the hidden labels ≥ 0.80 (100 test molecules; over-covering is limited by the width threshold); median interval width ≤ threshold (no trivially wide intervals) | M | ✅ |
| C2 `ensemble_uncertainty_classification` | CLI | Calibrated probabilities from an ensemble for a binary task | `ensemble` works for classification but not multiclass; Platt/isotonic need a calibration set | hidden-test log loss / Brier score ≤ threshold | M | ✅ |

## D. Migrating from v1

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| D1 `v1_model_file` | CLI | Predict with a real v1 `.pt` model that was trained with RDKit features | `chemprop convert`; `--multi-hot-atom-featurizer-mode V1`; matching `--molecule-featurizers v1_rdkit_2d_normalized` | predictions equal reference predictions from the converted model to 1e-4 | M | ✅ |
| D2 `v1_python_pipeline` | Py | Port a longer v1 Python pipeline (custom data loading, scaffold split, ensemble, predictions with uncertainty) | no v1 API in v2; split and ensemble semantics differ | runs; predictions + uncertainty for 10 molecules; uses a scaffold split; no v1 API left | H | ✅ |

## E. Reactions and cuik-molmaker

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| E1 `cuik_reactions` | CLI, Py | Train on reactions with the cuik-molmaker featurizer | `--use-cuikmolmaker-featurization` and its restrictions; Python: `LazyReactionDatapoint`, `CuikmolmakerReactionDataset`, `CuikmolmakerCGRFeaturizer` | hidden-test RMSE ≤ threshold; the final `chemprop train` line (or `train.py`) uses cuik-molmaker | H | ✗ |
| E2 `reaction_plus_solvent` | CLI | Reaction + solvent, with a deeper network for the reaction than the solvent | molecules come first, then reactions, in per-component options (`--depth`, `--split-key-molecule`) | inspect: depths assigned to the right components; hidden-test RMSE | H | ✅ |

## F. Atom- and bond-level properties

Data built with RDKit (fixed seed) on ~1,000 molecules: Gasteiger partial charges (atom), bond lengths
from an MMFF-optimised conformer (bond), logP (molecule), total charge (constraint).

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| F1 `atom_charges` | CLI, Py | Predict partial charges per atom | list-valued CSV targets; atom order must match RDKit (`--keep-h`/`--add-h`, `--reorder-atoms`); `MolAtomBondMPNN` in Python | one value per atom; hidden-test per-atom MAE ≤ threshold | H | ◐ |
| F2 `bond_lengths` | CLI, Py | Predict bond lengths per bond | bond order follows RDKit bond indices; Python: bond predictor `input_dim × 2` | one value per bond; hidden-test bond MAE ≤ threshold | H | ◐ |
| F3 `mol_atom_bond_joint` | CLI | One model for molecule, atom, and bond targets | three target types; tracking-metric suffixes (`-mol`, `-atom`, `-bond`); per-type output files | all three hidden-test errors ≤ thresholds | H | ◐ |
| F4 `constrained_charges` | CLI, Py | Partial charges that sum to the molecule's total charge | `--constraints-path`, `--constraints-to-targets`, constraint file format | per-molecule sums equal the given `charge_sum` (≤ 5e-3) **and** hidden-test MAE ≤ threshold | H | ◐ |

## G. RIGR (resonance-invariant featurization)

Test sets of resonance-equivalent SMILES pairs (nitro, carboxylate, amidine, charge-separated forms)
generated with RDKit's resonance enumeration and checked by hand.

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| G1 `resonance_invariant` | CLI | "Predictions must not depend on which resonance structure is drawn" (RIGR not named) | the agent has to realise the default featurizer is not resonance-invariant and find RIGR | predictions for every resonance pair identical (≤ 1e-5); default featurization fails this | H | ◐ |
| G2 `rigr_python` | Py | Same, from Python | `RIGRAtomFeaturizer` + `RIGRBondFeaturizer`; dimensions differ from the defaults (`BondMessagePassing(*featurizer.shape)`) | inspect: RIGR feature dimensions; resonance pairs identical | H | ✗ |
| G3 `rigr_reactions` | CLI | Resonance-invariant model for reactions | RIGR mode with `--reaction-columns` (CGR) | resonance-equivalent reactions give identical predictions | H | ✗ |
| G4 `rigr_vs_default` | CLI, Py | Compare RIGR and the default featurizer on a resonance-sensitive dataset, report which is better | a fair comparison: same splits, several replicates | reported numbers consistent with the agent's own saved models (recomputed); hidden-test RMSE of the chosen model | O | ◐ |

## H. Stereochemistry

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| H1 `stereo_invariant` | CLI, Py | "The property doesn't depend on stereochemistry; stereoisomers must get the same prediction" | `--ignore-stereo` at train *and* predict time; Python `from_smi(..., ignore_stereo=True)` | identical predictions for enantiomer and diastereomer pairs | M | ◐ |
| H2 `stereo_matters` | CLI | Stereoisomers have different labels | keep stereo (default); don't canonicalise it away | predictions differ for stereoisomer pairs; hidden-test RMSE | M | ◐ |
| H3 `no_stereo_leakage` | CLI | Data where the same molecule appears with and without stereo; avoid train/test leakage | `random_with_repeated_smiles` or a custom split on stereo-free keys | no stereo-free canonical SMILES in both train and test (`splits.json`) | H | ◐ |

## I. GPUs

Run in the GPU box with both GPUs visible, so all three run separately, only when both GPUs are free
(I1 only makes sense if the agent could wrongly use both).

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| I1 `pick_one_gpu` | CLI | Two GPUs visible, GPU 0 "busy": train on GPU 1 only | the defaults use every GPU with DDP | log shows a single process on the intended device | M | ✅ |
| I2 `ddp_cli` | CLI | Train with DDP on both GPUs | `--devices 2`; per-process batch size; logs printed twice | log shows 2 processes; predictions written once | M | ✅ |
| I3 `ddp_python` | Py | DDP training from a Python script | Lightning relaunches the script (needs `if __name__ == "__main__"`); predict/test on one device or rank 0 | log shows 2 processes; predictions written once with the right row count | H | ◐ |

## J. Debugging

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| J1 `fix_buggy_script` | Py | A training script that runs but gives poor results; "find and fix the problems" | silent bugs: target scaler never applied to predictions, test set normalised, wrong `input_dim` for a CGR featurizer, `num_workers` hang on macOS-style settings | hidden-test RMSE in original units ≤ threshold; each planted bug fixed (checked in the script) | H | ✅ |
| J2 `fix_broken_cli_pipeline` | CLI | A shell pipeline whose predictions are silently wrong (train/predict flag mismatch, descriptor columns as targets) | `predict` doesn't read the training config | predictions match a reference pipeline's | M | ✅ |

## K. Open-ended, performance-graded

| ID | Variants | Task | What makes it hard | Grading | Diff. | Skill |
|---|---|---|---|---|---|---|
| K1 `best_model_20min` | either | "Build the best solubility model you can in 20 minutes" (~2,000 molecules, hidden test set) | choosing split, epochs, early stopping, CheMeleon or not, ensembling, descriptors | hidden-test RMSE (continuous); compared between arms as distributions | O | ✅ |
| K2 `small_data` | either | Same with 60 training molecules | does the agent choose CheMeleon or an ensemble unprompted? | hidden-test RMSE | O | ✅ |

## Implementation notes (first batch, as built)

- Round 3 uses its own workspace, `.workspace_r3/`: the chemprop checkout without `tests/data` and
  with notebook outputs cleared, plus only the round-3 data. Test molecules are never ones that
  appear anywhere in the checkout or twice in a dataset (checked: no test molecule appears with its
  label anywhere in the workspace).
- Every prompt says there is no internet access; CheMeleon tasks say where its weights are cached.
- Invariance tasks (G1, H1): max difference within a group ≤ 1e-4, and predictions must vary
  across molecules (no constant models).
- Partial-charge tasks fail if the agent computes RDKit charges directly (the labels are Gasteiger
  charges); the CLI needs a placeholder `charges` column to predict (a Chemprop bug, kept as is).
- Graders record the measured values (RMSE, MAE, coverage, ...) and never crash a run; model files
  are inspected in a no-network box.
- K1 `best_model` uses the solubility set (400 training molecules), not ~2,000.
- F4: the naive (unconstrained) solution fails mainly on the sum check; its MAE is only slightly
  above the threshold. The sum check is the point of the task, so MAE alone is a weak separator here.
- Validation: every task has a correct reference solution (CLI and/or Python) that passes and a
  naive one that fails for the intended reason (28/28), plus 3 negative cases for Python-only checks.

## Infrastructure needed

- **Hidden labels:** stored next to the graders (outside the workspace), never copied into a run.
- **Grader environment:** graders that inspect models run with the evaluation environment's Python
  (Chemprop installed); the rest stay standard-library only.
- **Reference solutions and thresholds:** one reference script per performance-graded task, run
  with 5 seeds; thresholds and the reference numbers are stored with the task.
- **Generated data:** atom/bond targets (RDKit), resonance pairs, stereoisomer pairs, a ~2,000
  molecule solubility set with a hidden test split. All with fixed seeds and kept in the repo.
- **Variants:** CLI/Python variants are separate tasks (e.g. `atom_charges_cli`,
  `atom_charges_py`) so they can be reported separately.

## Size and cost

About 30 task variants. With 3 arms, 3 runs (5 for performance tasks), and Haiku + Sonnet + Opus,
that is roughly 900 runs. Harder tasks will take more turns than round 2; assuming 2–3× round 2's
cost per run, the estimate is **$120–180 API-equivalent**, most of it Opus. Options to trim: run
Opus only on the H/O tasks, or 2 runs per cell for pass/fail tasks. Fable stays out until results
are meaningful.

## Open questions

1. Which groups to include first? Suggested first batch: A1, B1, B2, C1, E1, F1, F4, G1, H1, J1, K1
   (11 tasks, ~17 variants).
2. Performance thresholds: "within reference mean + 2 SD", or a fixed percentage of the reference?
3. Should the soft task "CheMeleon with RIGR" (not possible: CheMeleon needs the default
   featurizer) be included, graded on whether the agent says so?
4. Include the "installed only" baseline (no repository checkout) for some tasks?
