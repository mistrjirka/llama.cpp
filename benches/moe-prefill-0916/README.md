# Qwen3.8-Flash-Next request-wide prefill

Use the [setup guide](../../docs/moe-prefill.md) for the benchmark runner. `bench-prefill.cpp` is the portable harness used in the September 16 hardware campaign. `run.py` supplies named presets and uses paths from its arguments instead of the original Sandbox layout.

`HARDWARE.md` and `NUMERICS.md` preserve the original reports and their measurement limits. Absolute paths in those reports and historical command manifests describe the machine where the tests ran; they are not installation requirements. `results/` holds the lightweight observations and manifests. Large model weights, prefix states, captured tensors, logits and profiler databases are not in Git.

The README's Flash-Next graph reads `results/graph.json`. That file identifies its source observations and comparison scope. It is separate from the upstream comparison in `benches/upstream-sync-0912/`.

The request-wide executor and sparse attention are opt-in. Source integration and follow-up checks are recorded in [MERGE.md](MERGE.md).
