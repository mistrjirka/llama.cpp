# Upstream sync validation — 7 September 2026

`v100-optimized` was merged with upstream `master` through **`67672dc5b`**, starting from fork revision `addd7bc86`. Upstream has no `develop` branch, so `master` is the development head used for this sync.

## Merge resolution

The upstream range adds 21 commits after the fork's previous upstream base `465e49b9c`. Only two files conflicted:

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: retained the fork's configurable combine-scratch layout while importing upstream `b74f590ea`'s non-divergent barrier/race fix.
- `ggml/src/ggml-cuda/mmvq.cu`: retained the fork's validated Q5_K x4 path while importing upstream's DGX-Spark-only L2 prefetch and current MMVQ routing changes.

The recent research-branch audit found the already-validated Flash-Next final candidate, shared multi-agent cache, MTP K/V-only refresh/readbacks, indexed speculative rollback, and per-request warm MTP budget already present in `v100-optimized`. Rejected experimental attention, compact-restore and grouping branches were deliberately not imported.

## Validation

The merged source was rebuilt from scratch for CUDA SM70+SM75 (`GGML_CUDA=ON`, `GGML_CUDA_FORCE_MMQ=OFF`) and all four focused unit gates passed: `test-sampling`, `test-mtp-draft-budget`, `test-kv-indexed-remove`, and `test-batch-alloc`.

### Ornith four-agent 100k/MTP3

Real four-agent Ornith Q6/Q5, Shisa Q5_0 MTP3, Q8 target/draft KV, V100 + RTX 2080 Ti, 100k cached source tokens per agent, 38/42/44/41 appended tokens and 128 generated tokens per agent. Mirrored old/new/new/old process order; one warmup per process excluded.

| build | wall time | aggregate generated | aggregate PP+TG | mean TG/agent |
|---|---:|---:|---:|---:|
| pre-sync production | 5.777 s | 88.68 tok/s | 117.26 tok/s | 27.09 tok/s |
| upstream-synced | **5.690 s** | **90.01 tok/s** | **119.01 tok/s** | **27.38 tok/s** |

The synced build is 1.5% shorter in this small sample. A deterministic 64-token sequential control matched exactly, including MTP acceptance.

### Dense Qwen3.8

| build | PP | TG | output |
|---|---:|---:|---|
| pre-sync final optimized | 34.34 tok/s | 25.82 tok/s | exact SHA |
| upstream-synced | **34.96 tok/s** | 25.70 tok/s | exact SHA |

TG changed by -0.49% and is treated as neutral. Output SHA is identical.

### Qwen3.8 Flash-Next

100k restored Q8 KV + 1k prompt, 35:14 V100/2080Ti split, `n-cpu-moe=18`, raw-q8 tiled QSA stack, 128 generated tokens:

| build | PP | TG | wall | output |
|---|---:|---:|---:|---|
| pre-sync final optimized | 240.50 | 23.00 | 9.742 s | exact SHA |
| upstream-synced | **242.21** | **23.04** | **9.723 s** | exact SHA |

The retained output SHA is `fa4eabafcd8e2e384c8918d644386b7b9a66a1bb330c3420903e4244411f2046` on both builds.

### Built-in Qwen MTP smoke

Stable layer split, 2k cached code prefix + short suffix, MTP3: output SHA and acceptance are identical old/new. Both produced `243f474526dfc3b0bceba2c0c3236a8fe845fca1f79568100d57b8603ab02a49` with **40/66** accepted/proposed. The attempted tensor-parallel control was discarded because the *old baseline* binary failed during NCCL initialization before serving; it is not used as merge evidence.

## Conclusion

No validated fork optimization was dropped, and no measured regression was found in the focused V100/2080Ti gates. This report is a sync/regression gate, not a new vanilla-vs-fork headline benchmark.
