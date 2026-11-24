# RTX 3060 host/device transfer benchmark

Build without running GPU work:

```sh
/opt/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 tools/server/benchmarks/rtx3060/transfer_bench.cu -o /tmp/rtx3060-transfer-bench
```

Run while the GPU is idle:

```sh
/tmp/rtx3060-transfer-bench > transfer-results.json
```

The benchmark measures 4 MiB and 64 MiB copies in both directions, with pageable and pinned host buffers. It uses `cudaMemcpyAsync` followed by stream synchronization for each copy, matching the production backend's synchronous wrapper. Each measurement follows eight warmup transfers and continues until both 512 MiB and 0.5 seconds have passed. Three repetitions alternate pageable/pinned order. Exact byte comparison runs before and after every measurement.

One device allocation is 64 MiB. At most one 64 MiB host allocation is pinned. An additional 64 MiB ordinary host allocation holds the deterministic reference bytes. CUDA context/driver allocations are outside these explicit buffer limits. Allocation and verification time are excluded from the timed transfer loop. Production save/restore measurements must include allocation, hashing, copying, and serialization costs.

Output is JSON. `GB_per_second` uses decimal GB (1,000,000,000 bytes). It reports the CUDA runtime and driver versions, device, PCI address, maximum link speed/width, and current link speed/width immediately after each timed loop. Missing sysfs fields are empty strings. PCIe speed is the signal rate from Linux sysfs, not measured payload bandwidth. An active x1 link must not be treated as x16 merely because the GPU supports x16.

For the current target model, 110,080 tokens at 30,464 bytes/token occupy 3,353,477,120 bytes of target Q8 KV. Divide this byte count by measured H2D bytes/second for a minimum full-cache reload time. This excludes verification compute, recurrent state, packing, contention, and any compressed-cache eviction. It is a feasibility estimate for a VeriCache-style design, not an implementation or benchmark of VeriCache.
