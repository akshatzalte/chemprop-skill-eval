| Model | Pass rate | Median tokens per run | Median cost per run | Median wall time per run |
|---|---|---|---|---|
| Haiku 5.5 | 59/63 → 56/63 (94% → 89%) | 495k → 263k (**-47%**) | $0.019 → $0.011 (**-39%**) | 147 s → 112 s (**-24%**) |
| Sonnet 5.5 | 58/63 → 57/63 (92% → 90%) | 121k → 122k (**+1%**) | $0.092 → $0.102 (**+11%**) | 69 s → 55 s (**-21%**) |
| Opus 5.5 | 63/63 → 63/63 (100% → 100%) | 173k → 209k (**+21%**) | $0.275 → $0.295 (**+7%**) | 103 s → 107 s (**+4%**) |

378 runs, $53.20 API-equivalent. 2 runs timed out without a result: they count as failures and are left out of the medians, since usage is only reported when a session ends.

| Task | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| `atom_charges_cli` | 3/3 → 3/3 | 3/3 → 0/3 | 3/3 → 3/3 |
| `atom_charges_py` | 3/3 → 2/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `attentive_aggregation_py` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `best_model` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `chemeleon_hourglass_cli` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `chemeleon_hourglass_py` | 3/3 → 3/3 | 2/3 → 3/3 | 3/3 → 3/3 |
| `chemeleon_reactions_cli` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `constrained_charges_cli` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `constrained_charges_py` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `cuik_reactions_cli` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `cuik_reactions_py` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `fix_buggy_script_py` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `marked_atom_cli` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `marked_atom_py` | 3/3 → 3/3 | 3/3 → 2/3 | 3/3 → 3/3 |
| `prediction_intervals_cli` | 2/3 → 0/3 | 3/3 → 2/3 | 3/3 → 3/3 |
| `prediction_intervals_py` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `rdkit_descriptors_cli` | 1/3 → 3/3 | 2/3 → 3/3 | 3/3 → 3/3 |
| `rdkit_descriptors_py` | 3/3 → 3/3 | 0/3 → 3/3 | 3/3 → 3/3 |
| `resonance_forms_cli` | 2/3 → 0/3 | 3/3 → 2/3 | 3/3 → 3/3 |
| `stereoisomers_cli` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |
| `stereoisomers_py` | 3/3 → 3/3 | 3/3 → 3/3 | 3/3 → 3/3 |

Cells: runs passed without the skill → with the skill.
