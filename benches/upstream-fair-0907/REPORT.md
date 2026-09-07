# Fair four-agent Ornith comparison against upstream — 7 September 2026

## Result

This benchmark answers one narrow serving question: with four Ornith agents already holding **100k cached tokens each**, how quickly can the engine append a short prompt and generate 128 tokens per agent with MTP3?

`aggregate PP+TG` here is deliberately defined as **all newly processed prompt tokens plus all generated tokens divided by whole-turn wall time**. The numerator is fixed at `38+42+44+41 + 4*128 = 677` tokens. `aggregate generated` uses only the 512 generated tokens over the same whole-turn wall time.

| Engine / mode | Whole turn | Aggregate PP+TG | Aggregate generated | Mean TG per agent |
|---|---:|---:|---:|---:|
| Latest upstream `f114f91f9` | 10.647 s | **63.65 tok/s** | 48.13 tok/s | 14.84 tok/s |
| Latest upstream + global `GGML_CUDA_FORCE_MMQ=ON` | 10.720 s | 63.17 tok/s | 47.77 tok/s | 14.68 tok/s |
| `v100-optimized`, common-denominator settings | **7.735 s** | **87.55 tok/s** | **66.21 tok/s** | **19.91 tok/s** |
| `v100-optimized`, normal optimized serving | **5.683 s** | **119.24 tok/s** | **90.18 tok/s** | **27.35 tok/s** |

The strict common-denominator fork result is **37.6% higher aggregate throughput** than latest upstream, with a **27.3% shorter** complete turn. The measured fork runtime is `c30407d98`; published `2ab68dda3` adds documentation only, so its runtime is identical. Relative to the exact upstream revision the fork is based on (`465e49b9c`), the corresponding aggregate gain is **39.6%**.

The normal deployment result reaches **119.24 aggregate PP+TG tok/s** and **90.18 aggregate generated tok/s**. It is intentionally kept separate because it re-enables fork-only serving features such as exact-prefix sharing and deferred MTP prompt handling; it is a practical deployment number, not the strict kernel/runtime A/B.

## Fairness controls

Both timed sides use the same V100 + RTX 2080 Ti, target model, Shisa Q5_0 MTP head, MTP depth 3, Q8 target/draft KV, 14:35 layer split, four 100k histories, four short source-review suffixes and deterministic 128-token completions. The strict fork arm disables fork-only prefix sharing, deferred-MTP prompt handling, separate draft ubatch and custom pipeline-copy count. Fork kernel/runtime optimizations remain enabled—that is the code being measured. Upstream was also tested with its available global `GGML_CUDA_FORCE_MMQ=ON` build option; it did not improve this workload.

Upstream cannot natively restore the fork's MTP-aware `.draft`/`.spec` companions. To avoid charging upstream for rebuilding four 100k histories, a **restore-only benchmark shim** was added to the upstream server. It loads the same target, draft and speculative snapshots before timing. The shim is not executed by timed completion requests and changes no decode, attention, sampling, scheduling or speculative-generation code. Its exact diff is retained here.

Same-base upstream and the strict fork were each run in two mirrored process arms, six retained measurements per arm after one warmup (**12 retained measurements per side**). Latest upstream, global-FORCE_MMQ upstream and the full-production fork each retain six measurements after warmup. Raw per-run summaries, acceptance counts, output hashes and exact argv are stored alongside this report.

Different upstream and fork output hashes are expected because the fork's validated arithmetic/kernel changes can alter floating-point sampling probabilities; acceptance remained in the same broad range. This comparison is therefore a throughput comparison, not a bit-identical-output claim. The earlier production validation separately checks state/output equivalence for changes intended to be exact.
