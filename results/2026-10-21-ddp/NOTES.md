# DDP task, 2026-10-21

**Setup.** `ddp_train` ("train using both GPUs with distributed data-parallel training, then
predict 10 molecules"), in the GPU box with both RTX 4090s visible, run one at a time once both GPUs
were free. 3 arms × 3 runs × Haiku 5.5 / Sonnet 5.5 / Opus 5.5 = 27 runs, $2.19 API-equivalent.
Tables: [`summary.md`](summary.md).

## Results

All 27 runs pass. Unlike the CPU tasks, the skill reduces tokens and cost here:

| | Haiku | Sonnet | Opus |
|---|---|---|---|
| median tokens, without → with (with_short) | 168k → 73k, −57% (−40%) | 69k → 42k, −38% (−20%) | 107k → 106k, −1% (−4%) |
| median cost | −37% (−27%) | −11% (−12%) | −14% (−15%) |
| median wall time | −58% (−58%) | −41% (−39%) | −22% (−18%) |

Multi-GPU training is one of the things the skill states explicitly with a verified recipe
(`CUDA_VISIBLE_DEVICES=0,1 chemprop train ... --accelerator gpu --devices 2`), including the trap
that the defaults already use every GPU. Without it, agents found the right flags by trial and error.
One task and 3 runs per cell: indicative, not conclusive.

## Grader correction

The first version only accepted Lightning's DDP log lines (`MEMBER: 2/2`) in the output the agent
saw. Agents with the skill trimmed or redirected training output, so 13 runs that did train on both
GPUs were marked failed. The grader now also accepts `devices = 2` in chemprop's `config.toml` (or 2
devices set in the agent's script); all runs were re-graded.
