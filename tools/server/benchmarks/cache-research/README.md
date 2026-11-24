# Lossless prompt-cache experiments

## Enabled by default

The server retains a complete, independent sequence state at the first user-message boundary when:

- RAM prompt caching and request-level prompt reuse are enabled.
- At least 1,024 tokens precede the first user message.
- The request has no media, LoRA, draft or speculative state, sliding-window attention, or request-prefill execution.
- The full state fits within the retained-prefix portion of the existing cache budget.

Retained states are reused only on exact token-prefix matches and are not consumed by restores. Their memory is counted against the regular cache budget. The retained portion is at most one third of that budget, capped at 2 GiB. The existing 8 GiB default total RAM limit is unchanged. `--no-cache-system-prefix` disables the feature, and `--cache-ram 0` still disables RAM caching entirely. The persistent cache file format is unchanged; a retained snapshot becomes an ordinary cache entry when loaded from a file.

On the supported path, RAM cache selection compares recoverable prefix tokens instead of the previous 25% overlap cutoff. Existing exact live-slot sharing is preserved. The first-user rewind checkpoint also receives eviction priority when available, including on existing speculative paths. Independent full-prefix capture for speculative/MTP, SWA, media and adapter paths remains disabled pending dedicated validation.

## Regression tests

Run `tools/server/tests/integration/test_retained_prefix.py --help`. Tested with a CPU build using:

- stories260K-f32 (native completion with explicit message delimiters)
- mamba-130m-hf Q2_K (native completion)
- Qwen3.5-0.8B Q4_0 (actual model chat template, `--template '' --ram 512`)

The tests verify reuse after erasing the live slot twice, interleaving an unrelated request, matching greedy responses against a cold reference, rejecting a changed prefix, and the two cache opt-outs. The tiny stories fixture cannot reliably infer chat message delimiters, so its regression supplies them explicitly. These are correctness/smoke tests, not statistical model-quality or V100 performance benchmarks.

## Experimental storage (not linked into the server)

Build:
```sh
g++ -O3 -std=c++17 -Wall -Wextra -Werror snapshot-store.cpp -lxxhash -o snapshot-store
```

Run `benchmark.py --help` for a private localhost capture of five synthetic branching snapshots. The standalone store compares fixed-size chunks, content-defined Gear chunks and tensor-aligned chunks where the serialized layout is supported. It verifies each reconstructed byte, including metadata and recurrent state; hash matches are checked with byte comparisons. Reported payload excludes allocator/index overhead. Tensor mode is an offline parser, not a proposed dependency on a particular architecture or state-file version.

An additional Qwen3.5-0.8B CPU capture produced 158,969,656 bytes across five snapshots. Fixed 64 KiB dedup reduced the payload by 16.49%; content-defined 64 KiB dedup reduced it by 23.93%. Both reconstructed every byte. The tensor parser rejected that serializer layout, motivating a general streaming tensor-boundary interface rather than a collection of architecture-specific file parsers.

Next steps: compare generic content-defined chunking and serializer-provided boundaries using a bounded streaming store, reference counts, exact restores, cache-budget stress tests, and GPU latency/quality tests. Keep storage experiments opt-in until validated in a live inference server. Evaluate SSD tiering and demand-aware retention separately.
