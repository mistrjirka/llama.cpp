#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/llama-mainline-cache
OUTROOT="$ROOT/profiling/selective-copy-cache-2026-08-01/qwen-equal-vram-sweep"
MODEL=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
PROMPT="$ROOT/profiling/qwen35-current-mlp-retest-2026-07-31/short-coding-prompt.txt"
TOKENS="$ROOT/profiling/long-coding-cache-2026-07-27/reference.tokens.txt"
GPU_INDEX=${GPU_INDEX:-1}
TOKENS_TO_RUN=${TOKENS_TO_RUN:-256}
mkdir -p "$OUTROOT"

wait_for_gpu_idle() {
    local stable=0 deadline=$((SECONDS + 600))
    while ((SECONDS < deadline)); do
        local sample free util
        sample=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        free=${sample%%,*}; util=${sample##*,}
        if [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] && ((free >= 30000 && util <= 5)); then
            stable=$((stable + 1))
            if ((stable >= 3)); then sleep 3; return 0; fi
        else
            stable=0
        fi
        sleep 2
    done
    return 1
}

run_one() {
    local cache_mib=$1 fit_target=$2 arm=$3
    local out="$OUTROOT/cache-${cache_mib}/${arm}"
    mkdir -p "$out"; rm -f "$out"/*
    wait_for_gpu_idle
    echo "### $(date -Is) cache_mib=$cache_mib fit_target=$fit_target arm=$arm"

    local cache_env=()
    if [[ "$arm" == cache ]]; then
        cache_env+=("GGML_EXPERT_CACHE_MIB=$cache_mib" GGML_EXPERT_CACHE_PROFILE=1)
    fi

    local start_ns end_ns rc=0
    start_ns=$(date +%s%N)
    env -i HOME="$HOME" PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 \
        "GGML_COMPLETION_FORCE_TOKENS=$TOKENS" \
        GGML_COMPLETION_BENCH_WARMUP_TOKENS=0 \
        "GGML_COMPLETION_BENCH_MEASURE_TOKENS=$TOKENS_TO_RUN" \
        "${cache_env[@]}" \
        timeout 1800s "$ROOT/build-v100/bin/llama-completion" \
            -m "$MODEL" -f "$PROMPT" -n "$TOKENS_TO_RUN" -c 4096 -b 512 -ub 128 -fa on -dev CUDA0 \
            -t 40 -tb 48 --poll 50 -fit on -fitt "$fit_target" -fitc 4096 \
            -s 1 --temp 0 --ignore-eos --single-turn --no-conversation \
            --no-display-prompt --simple-io -v \
            >"$out/stdout.txt" 2>"$out/stderr.txt" || rc=$?
    end_ns=$(date +%s%N)

    python3 - "$cache_mib" "$fit_target" "$arm" "$TOKENS_TO_RUN" "$rc" "$start_ns" "$end_ns" "$out" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
cache_mib, fit_target, arm, expected, rc, start_ns, end_ns, out = sys.argv[1:]
out = Path(out)
text = (out/'stderr.txt').read_text(errors='replace')
data = (out/'stdout.txt').read_bytes()
def last(pattern):
    values = re.findall(pattern, text, re.M)
    return values[-1] if values else None
marker = last(r'benchmark decode measurement complete after (\d+) tokens; elapsed = ([0-9.]+) ms, ([0-9.]+) tokens per second')
placement = last(r'offloaded (\d+)/(\d+) layers to GPU')
cuda_buffer = last(r'CUDA0 model buffer size\s*=\s*([0-9.]+) MiB')
cache_line = last(r'^(?:.*? )?(expert-cache: entries=.*)$')
profile_line = last(r'^(?:.*? )?(expert-cache-profile:.*)$')
r = {
    'cache_mib': int(cache_mib), 'fit_target_mib': int(fit_target), 'arm': arm,
    'exit': int(rc), 'wall_s': (int(end_ns)-int(start_ns))/1e9,
    'sha256': hashlib.sha256(data).hexdigest(),
    'tokens': int(marker[0]) if marker else None,
    'elapsed_s': float(marker[1])/1000 if marker else None,
    'tps': float(marker[2]) if marker else None,
    'offloaded_layers': int(placement[0]) if placement else None,
    'total_layers': int(placement[1]) if placement else None,
    'cuda_model_buffer_mib': float(cuda_buffer) if cuda_buffer else None,
    'cache_summary': cache_line,
    'cache_profile': profile_line,
}
(out/'summary.json').write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps(r, indent=2))
if r['exit'] != 0 or r['tokens'] != int(expected) or r['tps'] is None:
    raise SystemExit(1)
PY
}

# Equal-placement pairs. The fit target is increased by the nominal cache size
# relative to the 1,024 MiB clean baseline target.
for spec in "256 1280" "512 1536" "1024 2048"; do
    read -r cache_mib fit_target <<<"$spec"
    run_one "$cache_mib" "$fit_target" control
    run_one "$cache_mib" "$fit_target" cache
done

python3 - "$OUTROOT" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
results = []
for directory in sorted(root.glob('cache-*')):
    control = json.loads((directory/'control/summary.json').read_text())
    cache = json.loads((directory/'cache/summary.json').read_text())
    def number(line, pattern):
        match = re.search(pattern, line or '')
        return float(match.group(1)) if match else None
    row = {
        'cache_mib': cache['cache_mib'],
        'fit_target_mib': cache['fit_target_mib'],
        'control_tps': control['tps'],
        'cache_tps': cache['tps'],
        'cache_vs_control_percent': (cache['tps']/control['tps']-1)*100,
        'hash_matches': control['sha256'] == cache['sha256'],
        'placement_matches': (
            control['offloaded_layers'], control['total_layers'], control['cuda_model_buffer_mib']
        ) == (
            cache['offloaded_layers'], cache['total_layers'], cache['cuda_model_buffer_mib']
        ),
        'entries': number(cache['cache_summary'], r'entries=(\d+)'),
        'allocated_mib': number(cache['cache_summary'], r'allocated=([0-9.]+) MiB'),
        'hits': number(cache['cache_summary'], r'hits=(\d+)'),
        'misses': number(cache['cache_summary'], r'misses=(\d+)'),
        'uploaded_mib': number(cache['cache_summary'], r'uploaded=([0-9.]+) MiB'),
        'avoided_mib': number(cache['cache_summary'], r'avoided=([0-9.]+) MiB'),
        'decode_copy_mib': number(cache['cache_profile'], r'decode-bytes=([0-9.]+) MiB'),
        'copy_issue_ms': number(cache['cache_profile'], r'copy-issue=([0-9.]+) ms'),
        'control': control,
        'cache': cache,
    }
    results.append(row)
(root/'results.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
if not all(row['hash_matches'] and row['placement_matches'] for row in results):
    raise SystemExit('hash or placement mismatch')
PY
