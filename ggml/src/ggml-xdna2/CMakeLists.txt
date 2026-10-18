# ggml-xdna2: AMD XDNA2 NPU backend (device type ACCEL)
#
# XRT is optional at build time: without it only the host-memory arena and the host reference compute path
# are built (useful for validating the weight layout on machines without an NPU toolchain).
#
# find_package(xrt) is intentionally not used: the shipped xrt-config.cmake references libxilinxopencl.a,
# an FPGA-only library that is missing on NPU-only installs. The NPU relevant pieces are looked up directly.

find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt_bo.h
    HINTS /usr/include /usr/local/include $ENV{XILINX_XRT}/include)

find_library(XRT_COREUTIL_LIB
    NAMES xrt_coreutil
    HINTS /usr/lib /usr/local/lib $ENV{XILINX_XRT}/lib)

ggml_add_backend_library(ggml-xdna2
    ggml-xdna2.cpp
    ggml-xdna2-compute.cpp
    ggml-xdna2-cpu.cpp
    ggml-xdna2-bfp.cpp
    ggml-xdna2-attn.cpp
    ggml-xdna2-kernels.cpp
    ggml-xdna2-npu.cpp
)

# Kernel build step (kernels/: IRON generator + peano kernel sources, run as an external process at model load).
# GGML_XDNA2_MLIR_AIE_ENV: python environment with mlir_aie + llvm-aie (bin/aiecc, bin/python). Without it, kernels
# can only come from the cache and weights without a cached kernel stay on the CPU.
set(GGML_XDNA2_MLIR_AIE_ENV "$ENV{MLIR_AIE_ENV}" CACHE PATH "ggml-xdna2: mlir-aie python environment used to build NPU kernels (default: the venv made by scripts/xdna2-setup.sh)")
if (GGML_XDNA2_MLIR_AIE_ENV STREQUAL "")
    if (DEFINED ENV{XDG_DATA_HOME} AND NOT "$ENV{XDG_DATA_HOME}" STREQUAL "")
        set(_xdna2_default_env "$ENV{XDG_DATA_HOME}/ggml-xdna2/venv")
    else()
        set(_xdna2_default_env "$ENV{HOME}/.local/share/ggml-xdna2/venv")
    endif()
    if (EXISTS "${_xdna2_default_env}/bin/aiecc")
        message(STATUS "ggml-xdna2: NPU kernel toolchain found at ${_xdna2_default_env}")
    else()
        message(WARNING "ggml-xdna2: no NPU kernel toolchain found. Run scripts/xdna2-setup.sh (or pass -DGGML_XDNA2_MLIR_AIE_ENV=<venv>), "
                        "otherwise the NPU kernels cannot be built on the first run")
    endif()
elseif (NOT EXISTS "${GGML_XDNA2_MLIR_AIE_ENV}/bin/aiecc")
    message(WARNING "ggml-xdna2: ${GGML_XDNA2_MLIR_AIE_ENV}/bin/aiecc not found: is GGML_XDNA2_MLIR_AIE_ENV a venv with mlir_aie and llvm-aie?")
endif()
set(GGML_XDNA2_KERNEL_DEFS
    GGML_XDNA2_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    GGML_XDNA2_MLIR_AIE_ENV="${GGML_XDNA2_MLIR_AIE_ENV}")
target_compile_definitions(ggml-xdna2 PRIVATE ${GGML_XDNA2_KERNEL_DEFS})
target_link_libraries(ggml-xdna2 PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

target_compile_features(ggml-xdna2 PRIVATE cxx_std_17)
target_include_directories(ggml-xdna2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The host reference path emulates the NPU kernel's f32 arithmetic (separate multiply and add, no FMA)
set_source_files_properties(ggml-xdna2-compute.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

set_source_files_properties(ggml-xdna2-bfp.cpp PROPERTIES COMPILE_OPTIONS "-O3")
set_source_files_properties(ggml-xdna2-attn.cpp PROPERTIES COMPILE_OPTIONS "-O3")
set_source_files_properties(ggml-xdna2-cpu.cpp PROPERTIES COMPILE_OPTIONS "-O3")

if (GGML_OPENMP_ENABLED)
    target_compile_definitions(ggml-xdna2 PRIVATE GGML_USE_OPENMP)
    target_link_libraries(ggml-xdna2 PRIVATE ${GGML_OPENMP_TARGET_CXX})
endif()

if (XRT_INCLUDE_DIR AND XRT_COREUTIL_LIB)
    message(STATUS "ggml-xdna2: XRT found (${XRT_INCLUDE_DIR}, ${XRT_COREUTIL_LIB})")
    target_compile_definitions(ggml-xdna2 PRIVATE GGML_XDNA2_HAVE_XRT)
    target_sources(ggml-xdna2 PRIVATE ggml-xdna2-xrt.cpp)
    target_include_directories(ggml-xdna2 SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(ggml-xdna2 PRIVATE ${XRT_COREUTIL_LIB})
else()
    message(WARNING "ggml-xdna2: XRT not found (need xrt/xrt_bo.h and libxrt_coreutil): building WITHOUT NPU support. "
                    "Install XRT (Arch: pacman -S xrt xrt-plugin-amdxdna; others: build amd/xdna-driver and 'source /opt/xilinx/xrt/setup.sh'), "
                    "then configure again. See docs/backend/XDNA2.md")
endif()

# The kernel sources are read at run time (compiled for each model's shapes on first use): ship them with the library
install(DIRECTORY kernels/ DESTINATION ${CMAKE_INSTALL_DATADIR}/ggml-xdna2/kernels USE_SOURCE_PERMISSIONS
        PATTERN "*.sh" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# unit test of the device weight layout (header only, no ggml / XRT dependency)
if (GGML_BUILD_TESTS OR LLAMA_BUILD_TESTS)
    enable_testing()
    add_executable(test-xdna2-layout tests/test-xdna2-layout.cpp)
    target_compile_features(test-xdna2-layout PRIVATE cxx_std_17)
    target_include_directories(test-xdna2-layout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (GGML_OPENMP_ENABLED)
        target_compile_definitions(test-xdna2-layout PRIVATE GGML_USE_OPENMP)
        target_link_libraries(test-xdna2-layout PRIVATE ${GGML_OPENMP_TARGET_CXX})
    endif()
    add_test(NAME test-xdna2-layout COMMAND test-xdna2-layout)

    # CPU decode GEMV on the NPU layout: accuracy test + bandwidth microbenchmark
    add_executable(test-xdna2-cpu tests/test-xdna2-cpu.cpp ggml-xdna2-cpu.cpp ggml-xdna2-bfp.cpp)
    target_compile_features(test-xdna2-cpu PRIVATE cxx_std_17)
    target_include_directories(test-xdna2-cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(test-xdna2-cpu PRIVATE -O3)
    target_link_libraries(test-xdna2-cpu PRIVATE ggml-base Threads::Threads)
    if (GGML_OPENMP_ENABLED)
        target_compile_definitions(test-xdna2-cpu PRIVATE GGML_USE_OPENMP)
        target_link_libraries(test-xdna2-cpu PRIVATE ${GGML_OPENMP_TARGET_CXX})
    endif()

    # kernel cache exerciser (builds NPU kernels, does not touch the NPU)
    add_executable(xdna2-kernel-tool tests/xdna2-kernel-tool.cpp ggml-xdna2-kernels.cpp)
    target_compile_features(xdna2-kernel-tool PRIVATE cxx_std_17)
    target_include_directories(xdna2-kernel-tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(xdna2-kernel-tool PRIVATE ${GGML_XDNA2_KERNEL_DEFS})
    target_link_libraries(xdna2-kernel-tool PRIVATE Threads::Threads)
endif()
