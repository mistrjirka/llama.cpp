# RTX 3060 experiment notes

These are candidate mechanisms, not measured speedups on the hub. Keep the original Q6_K weights, Q8 KV cache, 110K context, and target verification while comparing one change at a time.

| Candidate | Small experiment | Constraint |
| --- | --- | --- |
| Share recurrent checkpoints | Share identical immutable host payloads between active and parked states | Count each allocation once, including active owners. Distinct states do not necessarily share bytes. |
| Faster state transfers | Compare pageable and pinned 4 MiB staging with the same synchronous copies | Pinning has a cost. Measure complete save/restore time, not only DMA bandwidth. |
| Adaptive MTP length | Vary draft depth while retaining the existing target verifier | Compare diverse prompts. A repetitive sequence with perfect acceptance is not representative. |
| ReplaySSM | Compare existing lazy recurrent kernel with conventional rollback before a new kernel rewrite | Existing base-plus-update storage already removes per-token state snapshots. It does not implement ReplaySSM's cross-step periodic flush. |
| VeriCache | Measure transfer cost and memory feasibility before dual-cache implementation | It needs a compressed GPU cache and a full CPU/storage cache, plus verification reloads and recurrent-state recovery. |

## Pinned staging experiment

Set `LLAMA_STATE_STREAM_PINNED=1` for the candidate. Unset it or set it to `0` for the baseline. Only sequence streaming changes. The callback bytes, tensor boundaries, synchronization, and 4 MiB staging ceiling stay the same. Each stream owns one host allocation until the save or restore completes. Debug logs report the selected buffer type.

The code asks the model's output device for a host buffer through the existing backend API. CUDA tries pinned host memory and falls back to a CPU buffer if allocation fails. A device without a host-buffer type uses the original vector. `GGML_CUDA_NO_PINNED=1` can exercise the CUDA fallback, but also changes other CUDA host allocations, so it is a correctness check rather than an isolated performance comparison.

Measure complete snapshot save and restore latency, cache hit latency, process RSS/PSS, GPU memory, and exact restored bytes or deterministic outputs. The experiment does not pipeline transfers or claim LMCache's full architecture. Add asynchronous double buffering only if transfer time remains significant and the extra staging memory is accounted.

## Primary sources and applicability

- [CUDA Best Practices Guide, pinned memory and asynchronous transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory). NVIDIA recommends pinned memory for maximum transfer bandwidth and notes that excessive pinning can hurt system performance. Pin/unpin is relatively expensive. Async overlap requires pinned host storage and suitable streams.
- [LMCache paper](https://arxiv.org/abs/2510.09665). Its mechanisms include batching small cache pages, reducing copies, and overlapping layer transfers with computation. The hub already transfers bounded larger chunks. Its synchronous serializer does not implement LMCache's separate-stream, layer-wise pipeline.
- [Dynamic speculation, Intel and Hugging Face](https://huggingface.co/blog/dynamic_speculation_lookahead). Draft confidence can stop an unproductive draft round early. Acceptance-based heuristics are a distinct simpler control. Published model-pair results do not predict the best native MTP depth for SenseNova on an RTX 3060.
- [ReplaySSM](https://arxiv.org/abs/2610.13778), especially Appendix E. The paper evaluates Mamba-2 and Gated DeltaNet in vLLM on H100 and B300. Its standard-decoding batch-one results are 0.96-0.99x of its tuned baseline, while gains grow at large serving batches. Its low-precision checkpoint path is separate from the exact-storage changes considered here.
- [VeriCache](https://arxiv.org/abs/2605.17613), especially Sections 8 and 10. Evaluation uses RTX PRO 6000 or H100 with PCIe 5.0 x16. The paper explicitly retains full KV on CPU or storage in addition to compressed GPU KV. The hub's tighter host memory, single active request, and slower interconnect require a separate feasibility result.
- [Marconi](https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html). It motivates admitting and evicting hybrid cache entries according to expected reuse and compute saved per memory byte. It does not make distinct recurrent states byte-identical. The existing repository replay already found no additional hit-rate gain from its tested value-aware heuristic on one trace.

## Existing hub implementation

At base `4166068`, the CUDA build targets SM86 with CUDA graphs and Flash Attention enabled. Forced MMQ and forced cuBLAS are both disabled, leaving automatic dispatch. The recurrent CUDA path stores a base state and rank-one updates for rollback. Its lazy-path eligibility includes `n_rs_seq + 1 >= ssm_d_conv - 1`. Check this condition when comparing low MTP depths because it can change the kernel path.
