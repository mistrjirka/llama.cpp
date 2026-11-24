# Exact prompt-cache research

Experimental, offline prototypes. The production server continues to use the retained-prefix patch (41d021c) and a 4 GiB RAM cache. No research code is linked into the deployed runtime.

## Reproduce

Build the standalone store:
```sh
g++ -O3 -std=c++17 -Wall -Wextra -Werror snapshot-store.cpp -lxxhash -o /tmp/llama-cache-snapshot-store
```

The host-specific capture-hub.sh briefly stops the idle production server, starts a private loopback fixture on port 2346, captures five real states using the historical static system/tool prefix and synthetic branch messages, then restores the service. Do not run while production is active. capture.py expects a private slot-save-enabled server. Snapshots may contain sensitive state and must remain local. capture-transformer.py uses a pinned SHA-256 verified 260K-parameter ordinary-attention test model, on CPU, without stopping production.

For each set of snapshots, run snapshot-store with fixed, gear or tensor mode, the chunk size (bytes for fixed/gear, rows for tensor), mmap or own, and the snapshot filenames. The store checks hash collisions using memcmp and verifies every reconstructed byte. The mmap mode references input files; own copies unique chunks into heap allocations. Reported unique payload does not include index, recipe, mapping, file cache or transient allocation overhead. Timings are for the entire snapshot set; disk and page-cache effects vary. Tensor mode is an offline parser of standard contiguous one-stream v4 KV serialization; other layouts require the generic byte-level mode. A production tensor-aware implementation should instead use the serializer's write_tensor boundaries. Gear is a simple content-defined chunker, not a tuned FastCDC reproduction.

replay.py renders and tokenizes 39 recorded text requests and inserts 18 synthetic unrelated auxiliary requests to approximate the observed interleaving. It simulates whole-snapshot storage, logical shared attention pages, LRU vs reuse-per-byte eviction, adaptive rewind checkpoints, and a hypothetical 16 GiB SSD tier. It evaluates four RAM budgets and two controlled architecture models. It does not execute model inference, measure disk latency, or prove byte identity of token-keyed pages. No accuracy loss is demonstrated by the standalone store's exact reconstruction, not a full live model integration test.

## Findings

Five SenseNova snapshots (two continuations, two branches, one independent prompt) total 4,659,929,076 bytes. Generic fixed blocks save 30.8%; content-defined 64 KiB blocks save 48.0%; tensor-aligned 256-row blocks save 48.1%. All restore checks passed, including the owned-block store. The tiny standard transformer fixture also passed, with 37.6% savings for 16-row tensor blocks. It is a correctness fixture, not a representative production performance model.

For the simulated interleaved hybrid workload, 2 GiB whole-state LRU yields 79.47% main-request token reuse versus 93.38% with shared KV pages. At 4 GiB both reach 93.38%; sharing reduces simulated peak saved-state RAM from 3463.6 to 2868.6 MiB. The chosen value-aware and adaptive checkpoint heuristics do not improve hit rate further on this trace. At 1 GiB the simulated SSD tier reaches 93.38%, but requires 33 restores and about 35.7 GiB of full-state reads (latency not modeled).

## Next experiments

Implement an independent streaming state-block storage interface with reference counts and bounded temporary buffers; compare generic content-defined chunking with tensor-aligned segmentation, then exact page sharing. Test eviction and checkpoint placement on diverse long-conversation/multi-agent traces, not just this one. Evaluate real SSD throughput and end-to-end cache restore latency. Validate exact restored bytes and logits/output against the original runtime across standard, hybrid and sliding-window models before deployment.
