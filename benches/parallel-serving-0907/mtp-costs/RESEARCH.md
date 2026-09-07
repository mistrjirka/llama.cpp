# MTP cost investigation and next priorities — 7 September 2026

## Executive decision

The strongest overall long-context opportunity is **target attention/layout plus direct-Q8 attention**, not merely shortening the MTP draft. The fastest useful serving experiment is a **cost-aware speculation controller**. The most interesting new MTP-specific long-context experiment is **Windowed-MTP**, applied only to draft attention, with full-context Q8 target verification unchanged.

Production `v100-optimized` and `local-llm-setup` are unchanged. This branch contains analysis/profiling artifacts, not a deployed controller or windowed-MTP implementation. The existing prefix-first experiment remains separate and unapproved because code-output differences are unresolved.

## Actual experiments

All runs use the safe source baseline `7571e2e17`, real Ornith AD-Q6_K-Q5_K, Shisa 12K-derived Q5_0 head, Q8 target and draft KV, CPU projector, four resident slots, 1.4M physical KV pool and four logical 400k caps. GPUs: V10032 GiB at200 W + modified RTX2080Ti22 GiB at250 W. Layer placement RTX:V10014:35; batch/ubatch2048/256, draft ubatch128, pipeline copies1.

A forwarding LD_PRELOAD shim records measured-call wall times and NVTX ranges without changing arguments, introducing extra synchronization or changing model kernels. It adds instrumentation overhead. The Nsight run is for attribution, not a new benchmark speed claim. Startup, warm-up, file restores and shutdown are excluded from measurement windows. Full raw traces/models/snapshots remain in the persistent sandbox; small summaries and hashes are retained here.

### Real source-code prompt screen

Four distinct review tasks with 12,009–12,012 cached tokens, mostly shared source code, append23 input tokens and generate192 tokens each. The one-active case still has four slots resident. Four-active off/MTP3 ran twice; other points ran once. This is not a long-run confidence interval or representative agent benchmark. MTP settings can change numerical execution and generated continuations; output quality/equivalence across depths was not certified.

| Active requests | Draft depth | Mean TG/request | Whole request group |
|---:|---:|---:|---:|
|1|off|77.82 tok/s|2.575 s|
|1|3|97.49 tok/s|2.086 s|
|4|off|45.54 tok/s|4.532 s|
|4|1|37.27 tok/s|5.575 s|
|4|2|34.05 tok/s|6.177 s|
|4|3|35.37 tok/s|5.889 s|

In this small screen, MTP3 helps one request but hurts four. Simply selecting a shallower fixed depth does not solve it. This does not imply a universal four-request cutoff: the earlier 100k synthetic screen had near-equal whole-turn times for off and MTP3, and long-context KV-bound behavior can make speculation beneficial again.

### GPU attribution at 100k

Separate trace: four100k histories sharing98k, each adds1k input and generates192 output. Repetitive synthetic corpus. Only kernels within marked measured-request intervals are included; model loading and slot restoration are excluded.

| Work | Sum of GPU kernel duration | Share |
|---|---:|---:|
| Attention |6375.18 ms|49.74%|
| Q8-to-FP16 conversion |1212.72 ms|9.46%|
| Matmuls |3761.06 ms|29.35%|
| GatedDeltaNet |264.54 ms|2.06%|
| Other |1202.62 ms|9.38%|

These are **summed kernel durations across two devices**, not elapsed-time percentages, utilization or an end-to-end speedup ceiling. The marked interval was15.33s; summed kernels12.82s. Nevertheless the concentration in attention and conversion is clear.

NVTX launch attribution inside drafting separately found about224.91ms Q8-to-FP16 conversion,145.71ms attention,228.84ms matmuls and98.03ms other kernels. Attention plus conversion is about53% of draft kernel time at this workload. These nested phase totals must not be added to the whole-run table again.

Do not interpret the relatively long `draft/sample` host duration as pure CPU top-k work: sampling can wait for asynchronously launched GPU computation. Backend draft sampling is already enabled. Moving a sampler to GPU is therefore not automatically an unimplemented fix.

## Concrete MTP code findings

1. **Per-sequence horizon is applied too late.** In `common/speculative.cpp`, the MTP inner loop stops at global `params.n_max`, while `common_speculative_draft()` subsequently truncates to `dp.n_max`. The simple-draft implementation checks both limits inside its loop. Fix the MTP loop to stop at the requested positive per-sequence limit. Current MTP3 traces show only one or two surplus rows near output limits, so this alone is not a major speedup; it is essential before a per-agent adaptive controller can save real drafting work.

2. **MTP3 is three sequential draft forwards.** The loop batches active sequences at each step, but it does not get all three future tokens from one forward. Larger target-verification batches also change attention/matmul shapes. Choose depth by time per emitted token, not acceptance percentage alone.

3. **Warm no-draft is not full-off.** The full-off benchmark starts without the speculative context. Suppressing proposal generation while retaining MTP can still execute target-to-draft hidden-state catch-up. A controller needs either a warm zero-proposal mode with its real cost measured, or a cold mode that accounts for future rebuild cost. Hysteresis is needed to avoid oscillating between expensive states.

4. **Do not remove draft re-evaluation indiscriminately.** The `TAG_SPEC_AVOID_DRAFT_REEVAL` TODO is not proof all catch-up is redundant. Proposed and verified target hidden states differ even when token IDs agree. The current `process_impl()` refreshes draft state using verified target hidden rows. Any optimization must preserve that relationship across rejection, rollback and save/reload.

5. **Ornith lacks the dense-Qwen shortlist dispatch.** `src/models/qwen35.cpp` marks its MTP projection with `GGML_HINT_MTP_SHORTLIST`; the analogous MoE projection in `qwen35moe.cpp` is a generic full LM-head matmul. A correctly mapped draft-only shortlist is a plausible port. Profile the specific output projection before assuming all draft matmul time comes from it.

6. **Full prompt materialization repeats.** Server drafting calls `get_text_tokens()` into `slot.spec_prompt` every step. MTP itself mostly needs position, last token and hidden carry, while other draft implementations use the full prompt. A capability-aware borrowed view could remove repeated copies without changing other implementations. Treat this as a small cleanup until separately measured.

## Paper-informed priorities

### A. Highest overall benefit: compact target attention and direct-Q8 kernels

The existing unified layout already shares some prefix reads across query tiles. Do not assume four requests always read the same prefix four times. Extend attention kernels to the actual multi-agent/MTP shapes, avoid whole-span Q8-to-FP16 staging, and avoid masked holes/private-tail regions. PAT/Hydragen-style packing complements these changes. Resolve the prefix-first numerical discrepancy before promoting its speed result.

### B. Earliest serving improvement: cost-aware MTP

Use a lightweight controller over depths0/1/2/3, based on active sequences, context lengths, actual emitted tokens per verification, measured draft/verification time and switching cost. Fix per-sequence early stopping first. Evaluate moving averages and hysteresis; do not train a complex scheduler until simple policies establish benefit. Nightjar supplies the switching-cost principle; TETRIS motivates unequal per-request draft allocation within a shared verification budget. MagicDec cautions against assuming large batch implies MTP should always be off.

### C. Best new long-context MTP experiment: window only the draft

Windowed-MTP (23July2026) keeps initial sink tokens plus a recent attention window for the draft head, while leaving target verification full-context. Suggested first sweep: approximately4k/8k/16k recent draft tokens plus64 initial sink tokens. These are experiment choices, not validated defaults for Ornith. Keep both target/draft storage atQ8.

Merely masking older keys while still reading/converting the full KV span will not deliver the intended gain. Preserve absolute positions and construct compact read indices; only then implement a physically bounded ring buffer with correct speculative rollback, prefix sharing and persistent restore. Test acceptance and actual cost before reducing the draft pool. Correct speculative verification preserves the target distribution mathematically, but our implementation still needs finite-precision and state-consistency validation. The paper's B200/H100, often1M-context and six-draft-step results are not a forecast for MTP3 on V100+2080Ti.

### D. Draft vocabulary projection, then training

FR-Spec-style vocabulary restriction is draft-only; target output vocabulary remains complete. First adapt the existing dense-Qwen hint for the MoE graph, then test shortlist sizes on code/token coverage. Rare identifiers and non-English text can hurt acceptance; a static list calibrated on general prose is not automatically suitable for coding. FastMTP-style recursive self-distillation is a later option. Our installed Shisa head already uses target-aligned KL distillation and code-heavy training, so replacing it with an unrelated newly published architecture is not a free improvement. In its model card,12K refers to training prompts, not a maximum supported context.

### E. Mixed PP/TG scheduling

After the above cost model is trustworthy, test three decoding agents plus one substantial prefill. Sarathi-Serve motivates decode-first prefill budgeting; POD-style overlap is a larger subsequent kernel project. Measure p95 inter-token latency and new-request first-token latency as well as total throughput. Tiny chunks everywhere can lower PP through repeated KV scans.

## Acceptance gates

- Start from safe7571 source for independent MTP changes; do not mix unapproved prefix-first code into an MTP A/B.
- Test1/2/4 active requests, code and unrelated prompts,100k and near350k actual cache depth; four logical400k caps alone are not a full-context test.
- Use teacher-forced next-token comparisons, MTP-off controls, rejection/rollback checks and parked-session/model-switch restores. A matching repetitive output hash is insufficient.
- Track whole-turn latency, aggregate throughput, per-request TG, TTFT/p95 inter-token latency, acceptance by draft position, peak VRAM and warm/cold MTP switching costs.
- Keep target weights/full-context attention and Q8 KV unchanged. New policies remain opt-in until regression tested.

## Primary sources

- Windowed-MTP: https://arxiv.org/html/2607.21535v1
- Nightjar (v5,15June2026): https://arxiv.org/html/2512.22420v5
- MagicDec: https://arxiv.org/abs/2408.11049
- TETRIS: https://arxiv.org/abs/2502.15197
- PAT: https://arxiv.org/html/2511.22333v3
- Hydragen: https://arxiv.org/html/2402.05099v2
- FR-Spec: https://arxiv.org/abs/2502.14856
- FastMTP: https://arxiv.org/html/2509.18362v1
- Shisa head model card: https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY
- Sarathi-Serve: https://arxiv.org/html/2403.02310v1

## Reproduction

Large fixture location: `/workspace/oai-qwen38-pp-lab/results/parallel-research-0907`.
Original runner and profiler: `/workspace/oai-qwen38-pp-lab/results/mtp-study-0907`.

Compile the forwarding shim from the repository root (requires CUDA NVTX headers):

```sh
RESULT=/workspace/oai-qwen38-pp-lab/results/mtp-study-0907
mkdir -p "$RESULT"
g++ -shared -fPIC -O2 -std=c++17 -I include -I ggml/include -I /usr/local/cuda/include \
  benches/parallel-serving-0907/mtp-costs/profile.cpp -ldl -o "$RESULT/profile.so"
python3 benches/parallel-serving-0907/mtp-costs/run-profile.py --corpus code --parallel 4 --depth 3
```

Runner defaults deliberately refer to the retained safe baseline binary and compatible snapshot fixtures; explicit paths are visible in the script. Do not replace those with mismatched model/context snapshots. Use an exclusive GPU resource lock for comparisons.
