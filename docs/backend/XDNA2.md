# llama.cpp on the AMD XDNA2 NPU (Ryzen AI)

The `xdna2-universal` branch of this fork adds a ggml backend for the AMD XDNA2 NPU, the NPU in Ryzen AI 300 (Strix Point, Krackan Point) and
Ryzen AI Max (Strix Halo) chips. It changes how llama.cpp computes, not how you use it: `llama-server`, `llama-cli`, `llama-bench`, the
OpenAI-compatible API and GGUF models work as upstream.

What runs where:

| Work | Where |
|---|---|
| Prompt processing: weight matmuls | NPU |
| Prompt processing: attention (head dim 128/256, 4 or 8 KV heads, up to 4 query heads per KV head, causal or sliding window) | NPU |
| Prompt processing: everything else (norms, RoPE, the Qwen3.5 linear-attention layers, other attention shapes) | CPU |
| Token generation (decode) | CPU (AVX-512), reading the weights from the NPU's copy in place |

Decode stays on the CPU on purpose: a single decode stream is limited by memory bandwidth, and the CPU cores read DRAM faster (~70 GB/s here)
than the NPU can (~53 GB/s).

Weight formats on the NPU: Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0. Other formats (IQ*, Q5_0/Q5_1, F16, BF16, ...) work but run on the CPU.
Mixture-of-experts layers (`MUL_MAT_ID`) run on the CPU.

Only tested on one machine: HP OmniBook Ultra 14 (Ryzen AI 9 365, 32 GB RAM), Arch Linux, kernel 7.2, XRT 2.21.75, NPU firmware
1.1.2.64. XDNA1 (Ryzen 7040/8040) is not supported. Windows is not supported.

## Install

### 1. NPU driver and XRT

The `amdxdna` kernel driver is in mainline Linux since 6.14. User space needs XRT and its amdxdna plugin:

- Arch Linux: `sudo pacman -S xrt xrt-plugin-amdxdna`
- Ubuntu, Fedora and others: build and install XRT and the plugin from [amd/xdna-driver](https://github.com/amd/xdna-driver)
  (its README has the steps per distribution). That XRT lives in `/opt/xilinx/xrt`: run `source /opt/xilinx/xrt/setup.sh` before
  building and running, so CMake and the loader find it.

Check that the NPU is visible:

```sh
xrt-smi examine        # should list an NPU device under "Device(s) Present"
```

### 2. Locked-memory limit

XRT locks a 64 MiB buffer when it opens the NPU. The backend raises the soft `RLIMIT_MEMLOCK` to the hard limit by itself, so this only needs
fixing when the hard limit is small too (check with `ulimit -Hl`; systemd services and some distributions default to 8 MiB). Then the backend prints
`cannot open the NPU device ... the locked-memory limit is 8192 KiB` and runs everything on the CPU (about 100 tok/s prompt processing instead of 500+).
To raise it for your user:

```sh
echo -e "$USER soft memlock unlimited\n$USER hard memlock unlimited" | sudo tee /etc/security/limits.d/99-amdxdna.conf
```

then log out and back in. If `ulimit -Hl` still shows 8192 (systemd sessions can ignore limits.d), set it in systemd as well:
`sudo systemctl edit user@.service`, add `LimitMEMLOCK=infinity` under `[Service]`, and log in again. A service that runs `llama-server`
needs `LimitMEMLOCK=infinity` in its own unit.

### 3. NPU kernel toolchain

The NPU kernels are compiled on the first run (once per model shape, then cached). That needs [mlir-aie](https://github.com/Xilinx/mlir-aie) and
its Peano compiler (`llvm-aie`) in a Python venv. `scripts/xdna2-setup.sh` checks steps 1 and 2 and creates that venv (needs Python 3.11 to 3.14,
about 1.4 GB on disk, no root):

```sh
git clone -b xdna2-universal https://github.com/mistrjirka/llama.cpp
cd llama.cpp
scripts/xdna2-setup.sh
```

It installs `mlir_aie==1.4.0` and the `llvm-aie` build that mlir-aie v1.4.0 itself uses (the versions this branch is tested with) into
`~/.local/share/ggml-xdna2/venv`, where the backend finds them without any flag. To use another venv, pass its path to the script and
set `GGML_XDNA2_MLIR_AIE_ENV` at run time (or `-DGGML_XDNA2_MLIR_AIE_ENV=` when configuring).

### 4. Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA2=ON
cmake --build build -j
```

The CMake output should contain `ggml-xdna2: XRT found` and `ggml-xdna2: NPU kernel toolchain found`. CMake warns if either is missing. GCC and
Clang both work. `cmake --install build` also installs the kernel sources (`share/ggml-xdna2/kernels`), which the backend finds next to the
library, so an installed build does not need the source tree (`GGML_XDNA2_KERNEL_DIR` overrides the location). In a prefix that is not on the
loader path, set `LD_LIBRARY_PATH=<prefix>/lib` as for any llama.cpp install.

### 5. Run

```sh
./build/bin/llama-server -m model.gguf -c 16384        # OpenAI-compatible API on http://127.0.0.1:8080/v1
./build/bin/llama-bench  -m model.gguf -fa 1 -ub 2048 -p 512,2048 -n 64
```

`llama-server --list-devices` should show `XDNA2: AMD XDNA2 NPU`. No extra flags are needed: the backend is used automatically, flash
attention is on by default, and when an XDNA2 device is present the default `-ub` is 2048 instead of 512 (llama-bench does not apply
that default, so pass `-ub 2048` there).

The first prompt takes 20-45 s longer while the NPU kernels compile; they are cached (a few MB) in `~/.cache/ggml-xdna2`. If they cannot be
built, the log says so (`ggml-xdna2: kernels: no kernel in ... and no toolchain to build it`) and that work runs on the CPU.

Batches of up to 8 tokens (decode, speculative-decoding checks) always run on the CPU. Attention runs on the NPU for batches of at least 32
tokens once the context holds at least 1024 tokens.

The server was checked with the official `openai` Python client: model list, chat (streaming and not, with usage), tool calls, the legacy
completions endpoint and 4 concurrent requests. Thinking models (Qwen3.5, Gemma 4) return their reasoning in `reasoning_content` before
the answer in `content`.

## Performance

Measured on the laptop above (Ryzen AI 9 365, Radeon 880M iGPU, 32 GB), one engine at a time, each through its own OpenAI-compatible
server and the same client:

- prefill: one chat request with a 4.3k-4.5k-token prompt (time to the first streamed token);
- decode: 256 generated tokens after a short prompt;
- power: the APU package power reported by the amdgpu driver (`PPT` in `/sys/class/hwmon/*/power1_input`), sampled every 20 ms during each
  phase. It covers CPU, iGPU, NPU and the memory controller, not the screen, the DRAM chips or the rest of the laptop. Idle is 4-5 W in
  power-saver and 7-10 W in performance.

Each number is the second of two runs; repeated runs differ by about 5 %.

Engines:

- this branch: `llama-server -c 16384 -fa on` (default `-ub` 2048);
- upstream llama.cpp Vulkan and ROCm 7.2.4 builds of the same commit: `llama-server -c 16384 -fa on -ngl 99`;
- [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) 1.0.9: `flm serve <model> -c 16384`, its default NPU power mode.

FastFlowLM is the NPU engine [Lemonade](https://github.com/lemonade-sdk/lemonade) uses on Linux; Lemonade's GPU engines are the llama.cpp
Vulkan/ROCm builds. Lemonade itself was not run: these are the engines it would start.

The models are not byte-identical. llama.cpp runs Q4_K_M GGUFs of Qwen3.5-4B and of Ornith-1.5-9B (a Qwen3.5-9B fine-tune with the same
architecture and size) and Google's QAT Q4_0 GGUF of Gemma 4 12B. FastFlowLM runs its own Q4NX conversions of Qwen3.5-4B, Qwen3.5-9B and
Gemma 4 12B. Output quality was not compared.

### Power-saver profile

The laptop's default profile, and the one that matters on battery.

Qwen3.5-4B

| Engine | Prefill tok/s | Prefill W | Prefill mJ/token | Decode tok/s | Decode W | Decode J/token |
|---|---:|---:|---:|---:|---:|---:|
| this branch (NPU + CPU) | 560 | 16.3 | 29 | 19.2 | 16.4 | 0.85 |
| llama.cpp Vulkan (iGPU) | 528 | 33.2 | 63 | 23.6 | 32.4 | 1.37 |
| llama.cpp ROCm (iGPU) | 454 | 31.0 | 68 | 22.3 | 31.4 | 1.41 |
| FastFlowLM (NPU) | 479 | 12.9 | 27 | 15.2 | 12.0 | 0.79 |

Qwen3.5-9B / Ornith-1.5-9B

| Engine | Prefill tok/s | Prefill W | Prefill mJ/token | Decode tok/s | Decode W | Decode J/token |
|---|---:|---:|---:|---:|---:|---:|
| this branch (NPU + CPU) | 408 | 18.3 | 45 | 11.9 | 17.8 | 1.50 |
| llama.cpp Vulkan (iGPU) | 307 | 30.7 | 100 | 15.1 | 29.7 | 1.97 |
| llama.cpp ROCm (iGPU) | 287 | 29.7 | 104 | 13.6 | 29.4 | 2.17 |
| FastFlowLM (NPU) | 338 | 11.8 | 35 | 9.3 | 11.5 | 1.23 |

Gemma 4 12B

| Engine | Prefill tok/s | Prefill W | Prefill mJ/token | Decode tok/s | Decode W | Decode J/token |
|---|---:|---:|---:|---:|---:|---:|
| this branch (NPU + CPU) | 136 | 17.8 | 131 | 9.2 | 18.4 | 2.01 |
| llama.cpp Vulkan (iGPU) | 261 | 29.2 | 112 | 11.3 | 29.7 | 2.61 |
| llama.cpp ROCm (iGPU) | 179 | 28.8 | 161 | 10.4 | 29.2 | 2.80 |
| FastFlowLM (NPU) | 278 | 16.3 | 58 | 6.8 | 14.3 | 2.10 |

### Performance profile

Qwen3.5-4B

| Engine | Prefill tok/s | Prefill W | Prefill mJ/token | Decode tok/s | Decode W | Decode J/token |
|---|---:|---:|---:|---:|---:|---:|
| this branch (NPU + CPU) | 624 | 40.1 | 64 | 20.2 | 43.6 | 2.15 |
| llama.cpp Vulkan (iGPU) | 480 | 43.1 | 90 | 23.6 | 41.0 | 1.74 |
| llama.cpp ROCm (iGPU) | 415 | 43.3 | 104 | 22.6 | 48.0 | 2.13 |
| FastFlowLM (NPU) | 554 | 22.0 | 40 | 15.2 | 17.6 | 1.16 |

Qwen3.5-9B / Ornith-1.5-9B

| Engine | Prefill tok/s | Prefill W | Prefill mJ/token | Decode tok/s | Decode W | Decode J/token |
|---|---:|---:|---:|---:|---:|---:|
| this branch (NPU + CPU) | 435 | 34.1 | 78 | 12.3 | 39.2 | 3.20 |
| llama.cpp Vulkan (iGPU) | 284 | 36.4 | 128 | 15.1 | 42.2 | 2.80 |
| llama.cpp ROCm (iGPU) | 280 | 41.2 | 147 | 13.2 | 41.5 | 3.16 |
| FastFlowLM (NPU) | 372 | 19.0 | 51 | 9.3 | 14.7 | 1.57 |

Gemma 4 12B

| Engine | Prefill tok/s | Prefill W | Prefill mJ/token | Decode tok/s | Decode W | Decode J/token |
|---|---:|---:|---:|---:|---:|---:|
| this branch (NPU + CPU) | 156 | 39.0 | 250 | 9.3 | 40.2 | 4.30 |
| llama.cpp Vulkan (iGPU) | 255 | 39.9 | 157 | 10.8 | 34.5 | 3.20 |
| llama.cpp ROCm (iGPU) | 168 | 42.1 | 251 | 10.6 | 40.8 | 3.85 |
| FastFlowLM (NPU) | 296 | 22.8 | 77 | 6.8 | 18.1 | 2.66 |

### Reading the numbers

- Prefill on Qwen3.5: this branch is the fastest of the four engines. In power-saver it draws about half the iGPU's power; its energy per
  prompt token is close to FastFlowLM's on the 4B and about 30 % higher on the 9B.
- Prefill on Gemma 4 12B: this branch is the slowest of the four. Its 8 global-attention layers run attention on the CPU (see
  Limitations), and that cost grows with the prompt length.
- Decode: the iGPU (Vulkan) is fastest everywhere. This branch decodes on the CPU cores: in power-saver at about 60 % of the iGPU's power,
  in performance at about the same power as the iGPU, because the cores clock up. FastFlowLM decodes on the NPU: slowest, but at the
  lowest power.
- In the performance profile, everything except FastFlowLM runs into the package power limit (about 40-48 W), so speed decides the
  energy per token there.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `GGML_XDNA2_MLIR_AIE_ENV` | build-time value, else `~/.local/share/ggml-xdna2/venv` | mlir-aie venv used to compile kernels |
| `GGML_XDNA2_KERNEL_DIR` | `<prefix>/share/ggml-xdna2/kernels`, else the source tree | kernel sources and build scripts |
| `GGML_XDNA2_CACHE_DIR` | `$XDG_CACHE_HOME/ggml-xdna2` or `~/.cache/ggml-xdna2` | kernel cache |
| `GGML_XDNA2_DISABLE` | unset | `1`: do not register the NPU device |
| `GGML_XDNA2_FA` | on | `0`: attention on the CPU |
| `GGML_XDNA2_FUSE` | on | `0`: do not fuse SwiGLU/GeGLU and residual adds into the NPU output pass |
| `GGML_XDNA2_THREADS` | physical cores | host threads of the backend |
| `GGML_XDNA2_STATS` | unset | `1`: print a time breakdown at exit |
| `GGML_XDNA2_DEBUG` | unset | `1`: log every op the backend rejects |

## Limitations

- Decode is 0-5 % slower than a plain CPU build of the same commit on the models above, and about 13 % slower with Q4_0 weights
  (measured on Llama 3.2 3B).
- Gemma 4: the global-attention layers (head dim 512, one KV head) and all attention layers of E2B/E4B (1-2 KV heads) run attention on the
  CPU, which limits long prompts on these models.
- The NPU computes in bfp16 (8-bit mantissas with a shared exponent per 8 values). Perplexity stays within 0.2-0.6 % of the CPU build on the
  models tested, but outputs are not bit-identical to the CPU.
