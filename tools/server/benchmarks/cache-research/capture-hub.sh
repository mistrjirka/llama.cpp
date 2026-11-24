#!/usr/bin/env bash
set -euo pipefail
repo="$HOME/programy/llama-sensenova-abliterated-build"
out=$(mktemp -d "$HOME/.cache/llama-cache-research.XXXXXX")
chmod 700 "$out"
printf '%s\n' "$out" >/tmp/llama-cache-research-path
pid=""
stopped=0
cleanup() {
  rc=$?
  trap - EXIT
  if [[ -n "$pid" ]]; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep .25; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  if (( stopped )); then
    systemctl --user start llama-server.service || rc=1
    for _ in $(seq 1 90); do
      if curl -fsS --max-time 2 http://127.0.0.1:2345/health >/dev/null 2>&1; then break; fi
      sleep 2
    done
    curl -fsS --max-time 3 http://127.0.0.1:2345/health || rc=1
    echo
  fi
  echo "CAPTURE_DIR=$out EXIT=$rc"
  exit "$rc"
}
trap cleanup EXIT
curl -fsS --max-time 4 http://127.0.0.1:2345/slots | python -c 'import json,sys; assert not any(s["is_processing"] for s in json.load(sys.stdin)), "Service is busy"'
systemctl --user stop llama-server.service
stopped=1
# Keep the production launcher untouched. Only a loopback test server can save files.
sed -e 's/--host 0.0.0.0 --port 2345/--host 127.0.0.1 --port 2346/' \
    -e "s@--jinja --host@--slot-save-path $out --jinja --host@" \
    -e 's/--ctx-size "${SENSENOVA_CTX_SIZE:-110000}"/--ctx-size 32768/' \
    "$HOME/.config/llama.cpp/sensenova-normal-110k.sh" >"$out/launch.sh"
chmod 700 "$out/launch.sh"
CUDA_VISIBLE_DEVICES=0 "$out/launch.sh" >"$out/server.log" 2>&1 &
pid=$!
for _ in $(seq 1 90); do
  if curl -fsS --max-time 2 http://127.0.0.1:2346/health >/dev/null 2>&1; then break; fi
  kill -0 "$pid"
  sleep 2
done
curl -fsS --max-time 3 http://127.0.0.1:2346/health
echo
python -u "$repo/tools/server/benchmarks/cache-research/capture.py" \
  --trace "$HOME/.blackdogbot/logs/llm-trace.jsonl" --out "$out"
