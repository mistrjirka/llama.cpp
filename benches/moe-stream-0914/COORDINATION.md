# Implementation and benchmark coordination — 2026-09-14

Working tree: `/workspace/llama-moe-stream-prefill-0914`, branch `experiment/moe-stream-prefill-0914`, base `v100-optimized` eae5d0ec2.

This is the per-MUL_MAT_ID bounded host-weight streaming prototype. Original production tree is untouched. Current compiled implementation built before 13:34 UTC has passed 30 real-weight operator cases on V100 and 30 on RTX 2080 Ti, plus an Ornith sparse/pageable/group-7 compute-sanitizer memcheck (zero errors). Raw artifacts are under this directory.

End-to-end synthetic llama-bench: Ornith ABBA PP2048/ubatch1024/n-cpu-moe24, 3 repeats each: A1 888.677, B1 968.718, B2 976.251, A2 888.731 tokens/s. Short TG64 is noisy and not accepted as regression-free. Next initial AB PP2048/ubatch1024/n-cpu-moe36: 204.184 -> 303.474 tokens/s. These are preliminary synthetic-throughput results, not distribution/quality acceptance. Next verbose log confirms 864 streamed operations; original Ornith nonverbose logs suppress activation traces.

Another continuation under `/workspace/moe-stream-validation-0914` is performing independent full-model checks against a snapshot. Do not overwrite its files or the separate `/workspace/llama-moe-streaming-0914` implementation. All GPU probes use the service gpu:all lock.

The implementation session is adding a separate short Ornith perplexity/log-probability numerical fixture check. It is not a representative quality benchmark. No automatic GPU memory fitting integration or production merge is planned in this step. A later cleanup will move stream-state cleanup after the pre-existing capture lock and guard NVIDIA-only implementation for HIP/MUSA compilation; record and rebuild before treating that as tested.
