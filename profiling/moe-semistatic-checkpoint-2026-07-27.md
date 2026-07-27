# Large-MoE cache checkpoint — 2026-07-27

This checkpoint preserves the unpushed `expert-cache-mainline` experiment before the topology-preserving semi-static continuation.

## Base and scope

- Base: `555881ebc` (`upstream/master` at checkpoint time)
- Runtime changes: persistent expert cache, dynamic/static expert slot maps, CUDA upload paths, GPU route-map support, graph split controls, trace instrumentation, and completion benchmark hooks
- Included: source, tests, benchmark/analysis scripts, prompts/forced-token fixtures, the compact GLM static map, and compact adaptive-static analyses
- Excluded: raw `.out`, `.err`, `.jsonl`, GPU telemetry, profiler captures, build products, and large experiment directories

## Evidence to reproduce

The strongest prior result is the frozen GLM map at:

`profiling/glm52-scale-2026-07-27/static-frequency-map.txt`

Reported behavior, still requiring independent repetition:

- frozen static vertical map: approximately 5.33–5.43 decode tok/s and 152 graph splits;
- default horizontal placement: lower throughput under the same GLM workload;
- current mutable path: approximately 4.24 tok/s and 302 graph splits despite higher resident-route coverage.

Treat these numbers as hypotheses until rerun from this checkpoint.

## Immediate gate

Implement an updates-disabled, double-buffered/versioned GPU route map initialized from the frozen static map. Do not add predictor-driven adaptation until this M0 arm satisfies:

- full-logit parity;
- at most 160 decode graph splits;
- at least 98% of frozen-static throughput.

The route map must publish only complete READY expert bundles and must not rebuild or republish the graph every token.
