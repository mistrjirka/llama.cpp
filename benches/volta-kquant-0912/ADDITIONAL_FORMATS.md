# Additional Volta quant formats — continuation notes

Date: 2026-09-12

## Scope

Continue the generic SM70/V100 quantization study beyond Q2_K–Q6_K, but only retain formats that can be validated on a real model in `/models` or have unusually strong direct evidence. Synthetic-only wins are not sufficient for integration.

Current real-model candidates:

- IQ4_XS: `/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/`
- IQ2-family: `/models/laguna-s-2.1/Laguna-S-2.1-UD-IQ2_M.gguf`

No real TQ model was found in `/models` at the start of this continuation, so TQ is out of scope unless a validation model appears.

## Integration policy

- Q2_K/Q3_K/Q4_K/Q5_K/Q6_K are already validated and should be integrated into `/workspace/muse-glimmer-v100` (`v100-optimized`) first.
- Additional formats must pass bitwise correctness where applicable, Compute Sanitizer, conversion/full-path microbenchmarks, and a real-model prompt-processing gate when practical.
- Reject model-specific or synthetic-only changes that do not survive end-to-end testing.
- Keep SM70-only dispatch and stock fallbacks for other architectures/small work.
- Commit locally to `v100-optimized` when a group is ready; do not push unless explicitly requested.

## Running status

- K-quants: ready for integration.
- IQ4_XS: pending tensor/path inspection.
- IQ2-family: pending tensor/path inspection.
- TQ: not currently testable from `/models`.
