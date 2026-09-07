# Single-layer MTP K/V-only refresh — 7 September 2026

**Research branch: `perf/mtp-kv-refresh-0907`, based on `bd8d1ce0e`. Production remains unchanged.** This implements the first intervention proposed in `../mtp-regression-analysis-0907/REPORT.md`, rather than another attention-kernel approximation. It also fixes zero-proposal handling. Genuine suspended-state journaling/resumption, accepted-prefix-only refresh and draft windowing are not implemented by this patch.

## Implementation

`llama_decode_mtp_kv()` explicitly requests cache-only work. It rejects empty inputs, missing token/target-hidden inputs, implicit or nonzero output flags, and incompatible embedding-output requests before changing state. For the existing single-layer Qwen35 and Qwen35MoE MTP graphs it selects a distinct graph type. Other architectures keep their ordinary decode path rather than applying the single-layer dependency assumption.

The graph preserves token/target-hidden normalization, the embedding/hidden projection, K normalization, RoPE, optional K/V rotations and the existing indexed quantized cache-write operators. It does not allocate an attention mask or evaluate attention, the post-attention projection, FFN or vocabulary projection. Separate Q projection nodes are not rooted; fused QKV weights retain their existing computation to avoid inventing an unsupported split. Graph reuse includes the distinct graph type, so a refresh graph cannot be mistaken for a proposing graph.

The ordinary proposing graph, target graph and speculative acceptance remain unchanged. `common_speculative_impl_draft_mtp::process_impl()` opts into the new API only with `LLAMA_EXPERIMENT_MTP_KV_ONLY=1`, one draft layer and non-shared target/draft memory. Other values, including `0`, leave it disabled. Persistent target/draft K/V precision stays Q8_0.

Two zero-budget fixes are separate from the optimization. The draft loop now rejects zero-length proposals before its first model call, preventing the previously reproduced extra-token/output-capacity assertion. The final server also avoids prompt copies and checkpoint preparation when all speculative implementations have a zero configured proposal limit. This is not equivalent to unloading or suspending MTP: target hidden export and draft cache maintenance still exist.

## Environment and experimental controls

V100-SXM2 32 GiB at 200 W plus modified RTX 2080 Ti 22 GiB at 250 W. Real Ornith AD-Q6_K-Q5_K target, Shisa Q5_0 head, Q8 target and draft K/V, CPU vision projector. Layer ratio RTX:V100 14:35, target batch/ubatch 2048/256, draft ubatch128, pipeline copies1. The same 1.4M physical pool exposes four logical 400k slots.

Actual histories contain 100,000 C++ source tokens per agent with approximately 98k shared prefix; appended requests are 38/42/44/41 tokens, and each active agent generates 128 tokens. This is not four full 400k histories, a 350k benchmark or a complete coding-task quality benchmark. The second-model Qwen check is short-context only.

Timings exclude model loading and snapshot restoration but include HTTP admission, prompt append and generation. TG is the mean server-reported rate per active request. Each condition has two fresh process lifetimes with one excluded warmup and two measured turns per lifetime: four retained observations. Comparisons are mirrored, not randomized. GPU jobs are serialized; no compiler, profiler or validation interposer runs during timed serving. Binary snapshots and matching shared-library paths are fixed. Existing experimental Q8 attention, compact restore, grouping, sampling-view and bulk-hidden flags are disabled. The adaptive comparison alone enables the existing depth controller, in addition to cache-only refresh.

## Exact-state validation

### Same-input cache test

A validation-only interposer saves draft state, performs an ordinary refresh, restores the same input state and performs cache-only refresh. It compares every byte of sequence metadata and K/V values by logical token position, also checking that the intermediate restore preserves those logical bytes.

The first version compared physical serialization order directly and flagged a mismatch. Whole-context restore compacts physical holes; sequence serialization is in physical rather than chronological order. The corrected comparator sorts entire metadata/K/V rows together by logical position, validates unique positions and all row lengths, and discards no data. It is explicitly limited to the tested one-stream, one-layer, non-transposed Qwen M-RoPE draft format. That historical false alarm and the original validator are retained, not counted as a kernel error or hidden.

Ornith passed ten distinct refresh shapes from one to four sequences, including prefill, uneven verification batches and continuation after a four-slot target/draft/spec save/erase/restore. The interposer checked 3,004,349,520 cumulative bytes over these cases, with 23 intercepted refresh calls overall. This is cumulative comparison traffic, not extra VRAM or unique stored data. Invalid null/empty/output-producing API requests also left state unchanged.

Qwen3.8-27B's built-in MTP passed ten shapes, 53 intercepted calls and 93,571,504 cumulative bytes. Its target/draft caches were both Q8. The fixture uses a roughly 2k prefix and four 48-token outputs, not a Qwen 400k performance result.

### Whole-server parity without the interposer

A separate sequential four-agent test compared the old immutable binary, cache-only candidate and feature-disabled candidate. All four 64-token completions, acceptance/draft counters, and subsequent 32-token restored continuation matched exactly. The saved slot's target `.bin`, draft `.draft` and speculative `.spec` files were also byte-identical, without any canonicalization. This independently checks that the new graph preserves state used after model/slot restoration; it is focused evidence, not validation of every sampler, architecture or multimodal path.

## Repeated serving results

### Four active agents

Order: off / MTP3 / cache-MTP3 / cache-MTP1 / MTP1 / MTP1 / cache-MTP1 / cache-MTP3 / MTP3 / off.

| Mode | Complete turn | Observed range | TG per agent |
|---|---:|---:|---:|
| Genuine MTP off | 5.729 s | 5.721–5.741 | 26.08 tok/s |
| Ordinary MTP1 | 6.079 s | 6.012–6.173 | 25.01 tok/s |
| **K/V-only MTP1** | **5.816 s** | **5.737–5.892** | **25.90 tok/s** |
| Ordinary MTP3 | 6.620 s | 6.390–6.843 | 23.31 tok/s |
| K/V-only MTP3 | 6.522 s | 6.436–6.674 | 23.84 tok/s |

At unchanged depth1, cache-only refresh reduces mean turn time by **4.33%** and raises mean per-request TG by **3.58%**. The measured ranges do not overlap in this small screen. At depth3, the 1.48% mean turn reduction overlaps the variation; a robust depth3 speedup is not established.

The best new fixed setting remains **1.52% slower than genuine off** in this workload. Comparing old MTP3 with new MTP1 combines a depth-setting change and the code improvement; do not attribute their entire difference to this patch.

### One active agent, four histories resident

A/B/B/A at fixed MTP3:

| Refresh | Complete turn | Observed range | TG |
|---|---:|---:|---:|
| Ordinary | 2.719 s | 2.699–2.750 | 51.56 tok/s |
| K/V-only | 2.651 s | 2.644–2.661 | 53.47 tok/s |

This is **2.49% less whole-turn time / 3.70% higher TG**. It does not imply the same gain at 350k or for every prompt.

## Nsight verifies the removed computation

A separate Systems capture tags target, draft generation and cache refresh, after warmup and outside model load/restore. Both conditions contain **26 refresh calls**. All recorded CUDA kernels were linked to their launching API; there were no unmatched kernels.

| Refresh-stage GPU work | Ordinary | K/V-only |
|---|---:|---:|
| Sum of GPU kernel durations | 95.690 ms | 3.344 ms |
| Mean per refresh | 3.680 ms | 0.129 ms |
| Attention kernels | 52 calls, 40.977 ms | **0** |
| Full-history Q8→FP16 kernels | 50 calls, 32.884 ms | **0** |
| Matrix multiplication kernels | 476 calls, 18.043 ms | 144 calls, 1.669 ms |

This is approximately **96.5% less GPU kernel time for the refresh stage**, not 96.5% faster overall MTP or serving. The target forward remains the dominant component. Profiler overhead, GPU-duration sums and response latency are separate quantities; only the unprofiled tables above support serving claims. Detailed kernel names and the raw trace hash are retained in the evidence manifest.

## Zero-proposal maintenance and adaptive-depth screen

Before the final server zero-budget guard, fixed zero-proposal contexts now ran without crashing, but remained expensive: 6.552 s with ordinary refresh, 6.320 s with cache-only refresh, versus 5.728 s for genuine off. All returned draft counters were zero. This exposes the difference between suppressing proposals and actually suspending the speculative machinery.

The existing adaptive controller with cache-only refresh averaged **5.906 s**, versus **6.412 s** for fixed cache-only MTP3 in its separate matched A/B/B/A test (7.89% lower mean turn time; ranges 5.873–5.930 versus 6.361–6.521). It did not demonstrate beating the separately measured genuine-off baseline or new fixed MTP1. Its warmup/exploration observations are retained; it is not a new switching-cost-aware controller and this result does not establish safe suspension/resumption.

## Final zero-budget server guard

The server previously asked whether a draft *could fit the context*, but failed to check whether any loaded speculative implementation actually had a positive configured proposal budget. With a zero limit it still copied the entire prompt and prepared draft checkpoint/removal bookkeeping every iteration, even though the corrected draft loop produced no proposals.

A matched off / old-zero / guarded-zero / guarded-zero / old-zero / off comparison keeps K/V-only refresh enabled in both zero arms:

| Mode | Complete turn | Observed range | TG per agent |
|---|---:|---:|---:|
| Genuine off | 5.719 s | 5.674–5.770 | 26.18 tok/s |
| Zero proposals, before server guard | 6.338 s | 6.301–6.369 | 23.24 tok/s |
| **Zero proposals, after server guard** | **5.775 s** | **5.738–5.832** | **26.02 tok/s** |

The guard removes 8.89% of zero-mode turn time in this screen. Its remaining mean gap to true off is approximately **0.96%**, with overlapping ranges, down from 10.81%. It retains the draft cache and ongoing cheap updates. This is a corrected warm/no-proposal path, **not full suspension, model unloading or a new adaptive policy**.

The final binary's positive-depth MTP3 test repeats all output, acceptance, raw target/draft/spec snapshot and restored-continuation comparisons exactly against the pre-guard candidate (`final-equivalence-summary.json`). The separate zero-mode comparison also matches all four output sequences, counters, raw target/draft/spec snapshots and the restored continuation (`zero-equivalence-summary.json`).

## Remaining work and production boundary

This patch implements cache-only refresh and zero-budget fixes. It does **not** implement accepted-prefix-only refresh, suspend/resume journaling, draft-only windowed attention or a new cost controller. Snapshot parity and the strong reduction in refresh work make this a better-supported optimization than earlier kernels that changed model probabilities. Nevertheless, the full four-agent MTP regression is not solved, and production defaults remain unchanged.

Before promotion: longer realistic 350k histories, diverse prompts, changing concurrency, explicit rejection-boundary tests, RAM parked-agent persistence and real multimodal fallback. Then separate the remaining target hidden-export/recurrent configuration and scheduling costs from cheap draft maintenance. Do not replace a failed benchmark with a claim that zero proposals is a free off mode.


## Reproduction and evidence

Result/fixture root: `/workspace/oai-qwen38-pp-lab/results/mtp-kv-refresh-0907`. The source fixture and original server helper live in the sibling `parallel-refined-0907` and `parallel-research-0907` directories. They are existing persistent artifacts, not reconstructed prompts. `manifest.json` identifies immutable source/binary/fixture/trace hashes and which binary each phase uses. Large model files, binary snapshots and Nsight databases remain outside Git; committed raw results omit token arrays but retain hashes, timing counters and launch arguments.

Build the branch with `cmake --build build-sm70-75 --target llama-server -j 12`, after configuring the existing SM70/75 CUDA build. Keep every tested binary in a separately named immutable directory with its matching libraries. The scripts in this directory use `MTP_REFRESH_RESULTS_DIR` (defaulting to the persistent root above); GPU jobs must not overlap.

- `bench.py --suite four`, `--suite one`, `--suite warm`: mirrored serving comparisons.
- `equivalence.py`, `final-equivalence.py`, `zero-equivalence.py`: sequential output/counter/state/reload parity.
- `smoke.py`, `qwen-check.py`: cache interposer tests. Build `validate-kv.cpp` with `c++ -std=c++17 -O2 -shared -fPIC -Iinclude -Iggml/include ... -ldl`. The live interposer intentionally restores context and must never be used for timing.
- `adaptive.py`, `zero-guard.py`: separate adaptive-policy and zero-preparation ablations.
- `profile.py`, `analyze-profile.py`: CUDA launch attribution to target/draft/refresh.

Compile `profile-markers.cpp` as a shared interposer using `-Iinclude -Iggml/include -I/usr/local/cuda/include -shared -fPIC -ldl`. Capture with `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-event-trace=false --cuda-graph-trace=node -o refresh python3 profile.py`, export SQLite, then run `analyze-profile.py`. Stage markers are validation/profiling tools only, not part of production inference. Timing and profiling datasets are deliberately separate.
