#!/usr/bin/env bash
# One-time setup for the ggml XDNA2 (AMD Ryzen AI NPU) backend: checks the NPU stack and creates the python venv with the
# toolchain that compiles the NPU kernels on the first run (mlir-aie + Peano). Needs no root. Safe to run again.
#
#   scripts/xdna2-setup.sh [venv-dir]        default: ${XDG_DATA_HOME:-~/.local/share}/ggml-xdna2/venv  (found by the backend without any flag)
set -u

VENV=${1:-${XDG_DATA_HOME:-$HOME/.local/share}/ggml-xdna2/venv}
MLIR_AIE_VERSION=1.4.0
LLVM_AIE_VERSION=21.0.0.2026072001+ce8c0f8f      # the Peano build that mlir-aie $MLIR_AIE_VERSION itself uses
problems=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  FIX   %s\n' "$*"; problems=1; }

echo "NPU stack"
if compgen -G "/dev/accel/accel*" >/dev/null; then ok "NPU device node: $(ls /dev/accel/accel* | head -1)"
else warn "no /dev/accel/accel*: needs a Ryzen AI (XDNA2) CPU, Linux 6.14+ with the amdxdna driver, and the NPU enabled in the BIOS"; fi

XRT_DIR=
for d in "${XILINX_XRT:-}" /opt/xilinx/xrt /usr; do
  [ -n "$d" ] && [ -f "$d/include/xrt/xrt_bo.h" ] && { XRT_DIR=$d; break; }
done
if [ -n "$XRT_DIR" ]; then ok "XRT headers in $XRT_DIR"
else warn "XRT not found. Arch: sudo pacman -S xrt xrt-plugin-amdxdna. Others: build+install amd/xdna-driver (https://github.com/amd/xdna-driver), then 'source /opt/xilinx/xrt/setup.sh'"; fi
if [ "$XRT_DIR" = /opt/xilinx/xrt ] && [ -z "${XILINX_XRT:-}" ]; then
  warn "XRT is in /opt/xilinx/xrt: run 'source /opt/xilinx/xrt/setup.sh' in the shell that builds and runs llama.cpp"; fi

limit=$(ulimit -l)
if [ "$limit" = unlimited ] || [ "$limit" -ge 131072 ] 2>/dev/null; then ok "locked-memory limit: $limit"
else
  hard=$(ulimit -Hl)
  if [ "$hard" = unlimited ] || [ "$hard" -ge 131072 ] 2>/dev/null; then
    ok "locked-memory limit: soft $limit KiB, hard $hard (the backend raises the soft limit itself)"
  else
    warn "locked-memory limit is $limit KiB (XRT needs 64 MiB, else the NPU cannot be opened): run
          echo -e \"\$USER soft memlock unlimited\\n\$USER hard memlock unlimited\" | sudo tee /etc/security/limits.d/99-amdxdna.conf
        and log in again. If 'ulimit -l' is still small, also: sudo systemctl edit user@.service  -> [Service] LimitMEMLOCK=infinity
        (a llama-server systemd service needs LimitMEMLOCK=infinity in its own unit)"
  fi
fi

echo "Kernel toolchain: $VENV"
if [ -x "$VENV/bin/aiecc" ] && "$VENV/bin/python" -c "import importlib.metadata as m; assert m.version('mlir_aie').startswith('$MLIR_AIE_VERSION')" 2>/dev/null; then
  ok "mlir_aie $MLIR_AIE_VERSION already installed"
else
  PY=
  for c in python3 python3.14 python3.13 python3.12 python3.11; do
    command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(not (3,11) <= sys.version_info[:2] <= (3,14))' 2>/dev/null && { PY=$c; break; }
  done
  if [ -z "$PY" ]; then warn "need python 3.11 to 3.14 with the venv module (Debian/Ubuntu: sudo apt install python3-venv)"
  else
    echo "  creating the venv with $PY and installing mlir_aie $MLIR_AIE_VERSION + llvm-aie (about 1 GB download)"
    if "$PY" -m venv "$VENV" \
       && "$VENV/bin/pip" install -q "mlir_aie==$MLIR_AIE_VERSION" -f "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v$MLIR_AIE_VERSION" \
       && "$VENV/bin/pip" install -q "llvm-aie==$LLVM_AIE_VERSION" -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
       && [ -x "$VENV/bin/aiecc" ]; then ok "toolchain installed"
    else warn "installing the toolchain into $VENV failed (see the pip output above)"; fi
  fi
fi

echo
if [ $problems = 0 ]; then
  cat <<MSG
Ready. Build and run:
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA2=ON && cmake --build build -j
  ./build/bin/llama-server -m model.gguf        # the first prompt is 20-45 s slower while the NPU kernels compile (cached afterwards)
MSG
else echo "Fix the items marked FIX above, then run this script again."; fi
exit $problems
