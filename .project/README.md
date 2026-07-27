# Project Research Organization

This directory is the durable research notebook for the vertical MoE expert-cache work.

## Files

### `intent.md`

The current architectural objective, constraints, non-goals, and success criteria.

Use it to answer:

> What are we trying to build, and what must remain true?

Update it only when the project direction or constraints materially change.

### `progress.md`

Implemented features, verified behavior, current blockers, and the latest known repository state.

Use it to answer:

> What exists in the code right now?

Update it after meaningful implementation or validation work.

### `discoveries.md`

Experimental conclusions, including unexpected findings, dead ends, accepted hypotheses, and ideas that changed the roadmap.

Use it to answer:

> What did the experiments teach us?

Record conclusions with enough context that the same dead end is not rediscovered later.

### `experiments.md`

Chronological research log. Each entry should include:

- hypothesis;
- exact configuration;
- code or environment changes;
- benchmark workload;
- result;
- interpretation;
- next experiment.

Use it to answer:

> What exactly did we try, and how did it go?

### `benchmarks.md`

Compact benchmark tables only. Avoid long interpretation here.

Record:

- date;
- model;
- prompt/context;
- warmup and measured tokens;
- graph mode;
- cache mode;
- throughput;
- GPU utilization;
- output/correctness status;
- artifact directory.

### `todos.md`

Short actionable queue. Keep it current and remove completed work.

### `tasks/`

Multi-session workstreams with scope, milestones, and useful-completion criteria.

## Experimental discipline

1. Write the hypothesis and planned comparison before running the experiment.
2. Change one major variable at a time where possible.
3. Keep workload, seed, context, thread count, device and warmup identical across matched comparisons.
4. Separate trace-enabled diagnostic runs from performance runs.
5. Treat free-running output equality as insufficient for correctness; use forced-token/logit replay when available.
6. Label unsafe architectural probes clearly.
7. Record failed experiments and infrastructure failures, not only wins.
8. Prefer measured conclusions over intuition.

## Current graph-research focus

The immediate question is whether decode can return to a stable, mostly continuous GPU execution graph after the expert cache is populated.

The graph modes currently worth comparing are:

- ordinary llama.cpp baseline;
- dynamic split with CUDA graphs disabled;
- dynamic split with CUDA graphs enabled;
- GPU-only graph with CPU routing callback retained;
- GPU route-map graph with CPU routing callback removed;
- forced graph rebuild after warmup;
- startup-static GPU route-map graph;
- frozen-cache versus changing-cache topology.

The primary metrics are:

- measured decode tokens/s;
- backend submissions per token;
- CPU route-map graph count;
- CUDA graph capture/replay stability;
- GPU utilization and idle fraction;
- graph rebuild cost and VRAM reserve;
- resident-route and complete-layer coverage;
- correctness status.
