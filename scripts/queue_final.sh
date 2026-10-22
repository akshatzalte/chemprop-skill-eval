#!/usr/bin/env bash
# Final round: all 21 round-3 tasks, no skill vs. the PR skill (with_pr), 3 runs x 3 models.
# Waits for round 3b to finish, reuses its runs (same tasks, arms, frozen skill and workspace), and
# runs the rest. Launch detached: setsid nohup bash scripts/queue_final.sh < /dev/null & disown
set -u
cd "$(dirname "$0")"
LOG=../results/final.log
OUT=../results/2026-10-21-final
echo "--- $(date -u +%H:%MZ) queued: waiting for round 3b ---" >> "$LOG"
while pgrep -f '^python3 run_eval.py --name 2026-10-21-round3b' > /dev/null; do sleep 60; done
mkdir -p "$OUT"
if grep -q '^done:' ../results/round3b.log && [ ! -s "$OUT/runs.jsonl" ]; then
    cp ../results/2026-10-21-round3b/runs.jsonl "$OUT/runs.jsonl"
    echo "--- $(date -u +%H:%MZ) round 3b done; reusing its $(wc -l < "$OUT/runs.jsonl") runs ---" >> "$LOG"
else
    echo "--- $(date -u +%H:%MZ) round 3b did not finish cleanly; not reusing it ---" >> "$LOG"
fi
python3 run_eval.py --name 2026-10-21-final --workspace .workspace_r3 --tasks-dir ../tasks/round3 \
    --models haiku sonnet claude-opus-5-5 --arms without with_pr --runs 3 \
    --budget-usd 70 --per-run-budget-usd 3 --max-turns 80 --timeout-s 1800 --concurrency 4 \
    --min-free-gb 15 --resume >> "$LOG" 2>&1
