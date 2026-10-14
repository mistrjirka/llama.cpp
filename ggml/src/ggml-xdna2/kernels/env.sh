# Toolchain environment for building NPU kernels (mlir-aie IRON + peano), adapted from the lab's stage0/env.sh.
# usage: . env.sh <mlir-aie-env-dir>      (a python venv with mlir_aie and llvm-aie installed)
E=$1
export PATH="$E/bin:$PATH"
PYSP=$(ls -d "$E"/lib/python3*/site-packages | head -1)
export PEANO_INSTALL_DIR="$PYSP/llvm-aie"
MLIR_AIE_INC="$PYSP/mlir_aie/include"
PEANOCC="$PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -I $MLIR_AIE_INC -D__AIE_API_AIE_ADF_HPP__"
