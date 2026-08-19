# Optional PFlash proxy

This directory contains the fork's **approximate, opt-in** PFlash integration. It is separate from the lossless llama.cpp server optimizations in this branch: without this proxy, the server path is unchanged.

The proxy sits in front of an OpenAI-compatible `llama-server` and uses a Qwen3-0.6B BF16 [Lucebox](https://github.com/Luce-Org/lucebox) PFlash scorer to remove query-irrelevant aged assistant/tool text before cold prefill.

## Why this policy is different from simple prompt pruning

A normal agent session depends heavily on llama.cpp prefix caching. Recompressing its old history on every turn destroys that cache, while permanently discarding a message can hide information that a later query suddenly needs.

This proxy therefore uses these rules:

- `off` is the default and performs no compression.
- `auto` is **cold-long only**. If the first request observed for a session is below the threshold, that session remains byte-for-byte pass-through forever. A growing warm conversation is never retrofitted with PFlash.
- A cold-long request is scored against its current query once. The compressed aged history is then frozen.
- System/user messages and structured/non-string content are never compressed. Only aged string assistant/tool content is eligible.
- Omitted original text is retained in proxy memory. Later turns can recover it without rewriting the frozen prefix.
- Exact/rare identifiers use a cheap lexical recovery path. If a genuine new user turn has no strong lexical anchor, a small query-conditioned semantic PFlash recovery pass is used (1% by default).
- Agent tool continuations do not repeatedly run semantic recovery. New recovery is appended to the newest tail message and then frozen, preserving the old prefix.
- Recovered snippets are deduplicated across the session.
- Long cold histories are scored in bounded windows (`--scorer-max-tokens`, default 22k) so the Qwen3-0.6B scorer can coexist with the target on an 8 GB 3060 Ti.
- Scorer failures or implausibly empty selections fail safe to the verbatim request.

Because tokens are deliberately removed, PFlash is **not lossless** even though the recovery design reduces the risk of later information loss.

## Validated defaults on V100 32 GB + RTX 3060 Ti 8 GB

The tested target is Qwen3.8-27B UD-Q5_K_XL at 262,144 context with the lossless server settings described in the repository root README. The PFlash scorer runs on the 3060 Ti using a native sm86 build.

Recommended proxy defaults:

```text
--mode auto
--threshold-tokens 32000
--ratio 0.70
--recovery-ratio 0.01
--semantic-recovery user
--recovery-no-anchor-fallback pflash
--scorer-max-tokens 22000
--scorer-overlap 64
```

A 48,512-token cold retrieval prompt measured:

| path | target prompt | end-to-end wall | result |
|---|---:|---:|---|
| direct target | 48,512 | 78.75 s | `A731|B284|C915` |
| segmented PFlash + target | 36,020 | about 67.0 s | `A731|B284|C915` |

The PFlash path used about 9.05 s for scorer/loading/reconstruction and 57.98 s for target inference, roughly **1.17× end-to-end / 15% faster** in this test.

The recorded passing `pydicom-1256` Pi trajectory starts short. In `auto` mode all 39 requests remain unchanged, and even deliberately invalid scorer paths are never touched. This is intentional: the existing llama.cpp prefix cache is better for a conversation that became long gradually.

## Prepare the Lucebox scorer

The integration was validated against Lucebox commit:

```text
99ab4cebd331310adcc37c5ab89e30323b3ddf27
```

Clone it and build the scorer for the 3060 Ti (`sm86`):

```bash
git clone --recursive https://github.com/Luce-Org/lucebox.git
cd lucebox
git checkout 99ab4cebd331310adcc37c5ab89e30323b3ddf27
git submodule update --init --recursive
cd -

./tools/pflash/prepare-lucebox.sh ./lucebox 86
```

For a V100 scorer build use CUDA architecture `70` instead.

`prepare-lucebox.sh` applies the small daemon patch in this directory so a backend GPU can be selected with `--gpu=N`, and replaces the vendored llama.cpp MMVQ translation unit with `mmvq-bf16-stub.cu`. The stub is intentional for this **BF16-only scorer build**: Qwen3-0.6B BF16 does not use quantized MMVQ, and quantized MMVQ calls abort rather than silently using an unvalidated path. The original Lucebox MMVQ source is preserved as `mmvq.cu.pflash-original`.

The full Lucebox project remains an external dependency and is not vendored into this fork.

### Drafter model + tokenizer

Using Lucebox's documented workflow:

```bash
cd lucebox
uv sync
uv run hf download Qwen/Qwen3-0.6B \
    model.safetensors tokenizer.json \
    --local-dir server/models/drafter/

uv run python server/deps/llama.cpp/convert_hf_to_gguf.py \
    server/models/drafter \
    --outtype bf16 \
    --outfile server/models/Qwen3-0.6B-BF16.gguf
cd -
```

The proxy needs both the BF16 GGUF and the same Qwen3-0.6B `tokenizer.json`.

## Python environment

The proxy only adds the Hugging Face `tokenizers` runtime dependency:

```bash
python -m venv .venv-pflash
. .venv-pflash/bin/activate
pip install -r tools/pflash/requirements.txt
```

## Run it

Start the target server first (the root README has the full tested Qwen3.8 command), then launch the proxy:

```bash
.venv-pflash/bin/python tools/pflash/pflash_proxy.py \
  --listen 18321 \
  --target http://127.0.0.1:18320 \
  --bin ./lucebox/build-pflash-sm86/pflash_daemon \
  --gguf ./lucebox/server/models/Qwen3-0.6B-BF16.gguf \
  --tokenizer ./lucebox/server/models/drafter/tokenizer.json \
  --gpu 1 \
  --mode auto \
  --threshold-tokens 32000 \
  --ratio 0.70 \
  --recovery-ratio 0.01 \
  --semantic-recovery user \
  --scorer-max-tokens 22000 \
  --log /tmp/pflash-proxy.jsonl
```

Point the client at `http://127.0.0.1:18321/v1` instead of the target server's port.

**Lucebox's backend device index is not guaranteed to match `nvidia-smi` numbering.** The patched daemon prints its backend-device list at startup; choose the index corresponding to the desired scorer GPU. On the validated machine, Lucebox device `1` was the RTX 3060 Ti.

## Request controls

The proxy supports optional per-request headers:

- `X-PFlash-Mode: off|auto|frozen`
- `X-PFlash-Keep-Ratio: 0.70`
- `X-PFlash-Recovery-Ratio: 0.01`
- `X-PFlash-Session: <stable session id>`

`frozen` forces the PFlash policy even if the session is already active. That intentionally rewrites the old prompt once and can invalidate an existing target prefix cache. Prefer `auto` for normal use.

## State and safety notes

- Proxy session state is in RAM. Restarting the proxy loses the frozen/omitted history map.
- For best behavior, start/restart the proxy together with the target server. If the proxy is restarted in front of a target that already has a warm long session, it cannot know that the target's old prefix should be preserved.
- The current implementation serializes prompt transformation/state updates, then forwards requests normally. This avoids races between concurrent turns of the same session.
- `auto` pass-through preserves the original request bytes inside the proxy; no scorer is loaded for a short-started session.
- PFlash changes model input and can change model output. Keep it disabled for tasks that require strict losslessness.
