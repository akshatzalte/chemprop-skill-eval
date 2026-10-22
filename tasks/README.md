# Tasks and grading

Every prompt starts with the same context: *"Chemprop is installed in this environment (the
`chemprop` command and the Python package), and a checkout of the chemprop repository is in
`./chemprop`. Data files are in `./data`. This is a quick test, so keep any model training to at
most 20 epochs."* The full prompts are in this directory (`<task>.md`). Prompts describe the goal the
way a user would; they never mention the skill, and where possible they don't name the flag that
solves the task.

Graders are in [`../scripts/grade.py`](../scripts/grade.py). A run passes only if every check
passes. Each grader was checked by hand on a correct solution (passes) and on a naive solution
that falls into the task's pitfall (fails).

## Basic tasks (pilot round 1)

These check that the agent can do the work at all. Graders check the files produced.

| Task | Prompt (short) | Checks |
|---|---|---|
| `train_predict` | train a solubility model on 500 molecules, predict 10 new ones into `predictions.csv` | 10 rows; a numeric prediction column |
| `chemeleon_small_data` | fine-tune CheMeleon on 60 molecules, write test RMSE to `rmse.txt` | a number in (0, 5); CheMeleon used (command or script) |
| `fix_v1_script` | make a Chemprop v1 shell script run with v2, run it | 10 predictions; no v1 commands left; `chemprop train`/`predict` used; RDKit features kept |
| `uncertainty` | predictions with calibrated uncertainty | 10 predictions; an uncertainty column; calibration used |
| `reactions` | train on 80 atom-mapped reactions, predict 15 | 15 rows; numeric predictions |
| `mixtures` | train on dye + solvent pairs, predict 15 pairs | 15 rows; numeric predictions; two SMILES inputs used |
| `python_api` | write and run `train.py` using the Python API | 10 predictions; `train.py` imports chemprop and doesn't call the CLI |
| `fingerprints` | learned fingerprints for 10 molecules into `fingerprints.csv` | 10 rows; at least 8 numeric columns |

## Pitfall tasks (round 2)

Each targets a mistake that still produces output files, so the graders check that the result is
*correct*, not only that it exists.

| Task | Prompt (short) | Pitfall | Checks |
|---|---|---|---|
| `report_test_rmse` | train with the CLI, write the test-set RMSE to `test_rmse.txt` | test metrics are only logged; the default metric is MSE, not RMSE; `test_predictions.csv` holds predictions, not true values | the grader recomputes the RMSE from the model's saved test predictions and the true values at the test indices in `splits.json`; the reported number must match within 2% |
| `cluster_split` | split 80/10/10 keeping k-means clusters of Morgan fingerprints together, train, predict | the default split is random | 10 predictions; a k-means split used (`--split-type kmeans`, or k-means in the agent's code) |
| `predict_with_truth` | predict 10 molecules that have measured values; write `comparison.csv` with measured and predicted columns | `chemprop predict` overwrites the input's `logS` column with predictions | `measured_logS` equals the original measured values; `predicted_logS` differs from them for at least 8 of 10 rows |
| `temperature_input` | use molecule *and* temperature as inputs; predict 10 (molecule, temperature) pairs | `--descriptors-columns` without `--target-columns` also trains temperature as a target, and predictions then overwrite the input temperature | 10 predictions; the temperature column (if present) equals the input; temperature used as an input |
| `multitask_classification` | one multitask classifier for four toxicity endpoints with missing labels; write per-endpoint test ROC-AUC to `roc_auc.json` | per-task scores need `--show-individual-scores`; easy to report the overall score for every endpoint | the grader recomputes each endpoint's ROC-AUC from the saved test predictions (skipping missing labels); every reported value must match within 0.02 |
| `fix_v1_python` | update a v1 Python script (`TrainArgs`, `run_training`, `make_predictions`) to v2, keep using the Python API, run it | v1 Python API doesn't exist in v2 | 10 numeric predictions; no v1 API calls left; imports chemprop; no CLI or subprocess |

## GPU tasks (run in the GPU box, see the main README)

| Task | Prompt (short) | GPUs | Checks |
|---|---|---|---|
| `gpu_train` | train on the GPU, predict 10 molecules, write the device name to `device.txt` | GPU 1 | 10 predictions; `device.txt` names the GPU; the training log shows `GPU available: True (cuda), used: True` |
| `ddp_train` | train with distributed data-parallel on both GPUs, predict 10 molecules | GPUs 0 and 1; run separately, only when both are free | 10 predictions; the log shows 2 DDP processes |

## Arms

| Arm | Skill in `./.claude/skills/chemprop/` |
|---|---|
| `without` | none |
| `with` | the skill as proposed in the pull request ([`../skill/`](../skill/)) |
| `with_short` | an experimental shorter `SKILL.md` with the same reference files plus `references/workflows.md` ([`../skill_short/`](../skill_short/)) |
