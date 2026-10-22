# Without skill → `with` (relative change)

## By model

| model | runs | pass rate | median tokens | median cost $ | median turns | median wall time s | skill used |
|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | 3 / 3 | 100% → 100% (+0%) | 107,118 → 106,110 (-1%) | 0.202 → 0.174 (-14%) | 6 → 7 (+17%) | 60 → 46 (-22%) | 3/3 |
| haiku | 3 / 3 | 100% → 100% (+0%) | 167,985 → 72,860 (-57%) | 0.008 → 0.005 (-37%) | 9 → 6 (-33%) | 72 → 30 (-58%) | 3/3 |
| sonnet | 3 / 3 | 100% → 100% (+0%) | 68,754 → 42,427 (-38%) | 0.058 → 0.052 (-11%) | 5 → 5 (+0%) | 42 → 25 (-41%) | 3/3 |

## By model and task

| model | task | runs | pass rate | median tokens | median cost $ | median turns | median wall time s | skill used |
|---|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | ddp_train | 3 / 3 | 100% → 100% (+0%) | 107,118 → 106,110 (-1%) | 0.202 → 0.174 (-14%) | 6 → 7 (+17%) | 60 → 46 (-22%) | 3/3 |
| haiku | ddp_train | 3 / 3 | 100% → 100% (+0%) | 167,985 → 72,860 (-57%) | 0.008 → 0.005 (-37%) | 9 → 6 (-33%) | 72 → 30 (-58%) | 3/3 |
| sonnet | ddp_train | 3 / 3 | 100% → 100% (+0%) | 68,754 → 42,427 (-38%) | 0.058 → 0.052 (-11%) | 5 → 5 (+0%) | 42 → 25 (-41%) | 3/3 |

# Without skill → `with_short` (relative change)

## By model

| model | runs | pass rate | median tokens | median cost $ | median turns | median wall time s | skill used |
|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | 3 / 3 | 100% → 100% (+0%) | 107,118 → 102,951 (-4%) | 0.202 → 0.172 (-15%) | 6 → 7 (+17%) | 60 → 49 (-18%) | 3/3 |
| haiku | 3 / 3 | 100% → 100% (+0%) | 167,985 → 101,259 (-40%) | 0.008 → 0.006 (-27%) | 9 → 7 (-22%) | 72 → 30 (-58%) | 3/3 |
| sonnet | 3 / 3 | 100% → 100% (+0%) | 68,754 → 55,132 (-20%) | 0.058 → 0.052 (-12%) | 5 → 6 (+20%) | 42 → 26 (-39%) | 3/3 |

## By model and task

| model | task | runs | pass rate | median tokens | median cost $ | median turns | median wall time s | skill used |
|---|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | ddp_train | 3 / 3 | 100% → 100% (+0%) | 107,118 → 102,951 (-4%) | 0.202 → 0.172 (-15%) | 6 → 7 (+17%) | 60 → 49 (-18%) | 3/3 |
| haiku | ddp_train | 3 / 3 | 100% → 100% (+0%) | 167,985 → 101,259 (-40%) | 0.008 → 0.006 (-27%) | 9 → 7 (-22%) | 72 → 30 (-58%) | 3/3 |
| sonnet | ddp_train | 3 / 3 | 100% → 100% (+0%) | 68,754 → 55,132 (-20%) | 0.058 → 0.052 (-12%) | 5 → 6 (+20%) | 42 → 26 (-39%) | 3/3 |

Excluded runs: none
