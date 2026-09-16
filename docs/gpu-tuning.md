# Ornith and multi-slot tuning

### V100

Use the normal build and enable MMQ only for routed experts:

```bash
export GGML_CUDA_VOLTA_FORCE_MMQ=moe
```

This is safe in a shared Qwen/Ornith launcher: dense Qwen measured within ~0.2% of the selector being unset, while globally forcing MMQ is much slower for dense Qwen. On the fresh V100 Ornith control, selective `MMQ=moe` reached **976.39 tok/s** versus **943.65 tok/s** with global FORCE_MMQ.

### RTX 2080 Ti and mixed Ornith

For the tested RTX-only and V100+RTX Ornith configurations, both comparison arms use a dedicated FORCE_MMQ build:

```bash
cmake -S . -B build-ornith-mmq -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;75' \
  -DLLAMA_BUILD_UI=OFF
cmake --build build-ornith-mmq -j --target llama-server
```

Do not use this globally forced build for dense Qwen.

The headline Ornith configurations are:

| Setup | Model | Batch / ubatch | Placement |
|---|---|---:|---|
| V100 | AD-Q6_K/Q5_K | `2048/512` | V100 only, selective `MMQ=moe` |
| RTX 2080 Ti | AD-Q5_K/Q4_K | `4096/512` | RTX only, FORCE_MMQ build |
| V100 + RTX | AD-Q6_K/Q5_K | `2048/1024` | tensor split `1:1`, internal all-reduce, FORCE_MMQ build |

## Four active Ornith slots with MTP

For the production-style four-agent workload, MTP1 is faster than the older MTP3 setup. Four restored 100k-token histories generate 128 tokens each concurrently:

| Four active slots | Aggregate output | Mean per-agent TG |
|---|---:|---:|
| No MTP, tuned 18:31 RTX:V100 layer split | 90.29 tok/s | 26.14 tok/s |
| Q4 Shisa, MTP1, target-head reuse | **106.59 tok/s** | **32.58 tok/s** |
| Change | **+18.05%** | **+24.63%** |

For this profile, q4_0 draft K/V recovers about 684 MiB on the RTX versus Q8 draft K/V with effectively unchanged aggregate throughput. Enable target-head reuse with `LLAMA_MTP_SHARE_TARGET_IO=head`.

<details>
<summary>Example four-slot Ornith MTP command</summary>

```bash
export GGML_CUDA_VOLTA_FORCE_MMQ=moe
export GGML_CUDA_ALLREDUCE=internal
export GGML_CUDA_AR_COPY_THRESHOLD=131072
export LLAMA_MTP_SHARE_TARGET_IO=head

./build/bin/llama-server \
  --model /path/to/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf \
  --device CUDA1,CUDA0 --tensor-split 14,35 --split-mode layer \
  --gpu-layers all --fit off \
  --ctx-size 1400000 --parallel 4 --kv-unified --kv-unified-per-slot 400000 \
  --override-kv qwen35moe.context_length=int:400000 \
  --rope-scaling yarn --rope-scale 1.52587890625 --yarn-orig-ctx 262144 \
  --flash-attn on --batch-size 512 --ubatch-size 128 --pipeline-copies 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --slot-fork-prefix \
  --spec-type draft-mtp \
  --spec-draft-model /path/to/mtp-shisa-ornith15-all-q4.gguf \
  --spec-draft-device CUDA1,CUDA0 --spec-draft-ngl all \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  --spec-draft-ubatch 64 --spec-draft-n-max 1 --spec-mtp-defer-prompt
```

The command assumes RTX=`CUDA1` and V100=`CUDA0`; verify with `--list-devices`.

</details>

Detailed MTP tuning, acceptance, memory, and rejected placement experiments are in [`benches/gemma4-0911/NOTES.md`](../benches/gemma4-0911/NOTES.md).


[Back to the README](../README.md)
