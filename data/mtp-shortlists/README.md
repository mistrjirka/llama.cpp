# Qwen3.8 MTP shortlist maps

`qwen38-27b-exact-131072.i32` is the validated 131,072-row proposal-head map used by the Volta Qwen3.8-27B MTP shortlist path.

- format: little-endian signed `int32_t`, strictly increasing real vocabulary row IDs
- full proposal head: 248,320 rows
- shortlist rows: 131,072
- SHA-256: `d2f2c068e3a637224b44741d40c914c987111807450d4d6f3c7ffd04b7f0f8cd`
- held-out proposal-token coverage in the validation corpus: 99.1926%
- training-corpus coverage: 99.3912%

The shortlist construction follows the NInfer-V100 frequency-ranked proposal-head idea, including explicit forcing of the validated Qwen special-token set. The binary here was generated for this fork's Qwen3.8 validation corpus; it is not a copied model tensor.

The map is used only for the MTP **proposal** head. The target model still verifies proposals with its normal full head, so this is a speculative-decoding performance optimization rather than a target-model vocabulary truncation.

Enable it with:

```bash
export GGML_CUDA_QWEN35_MTP_SHORTLIST="$PWD/data/mtp-shortlists/qwen38-27b-exact-131072.i32"
```

The CUDA path is deliberately restricted to the validated Qwen3.5/3.8 27B Q6_K output-head geometry (`5120 x 248320`) on sm70. Invalid maps or other geometries fall back to the full proposal head.
