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
    ggml-xdna2-bfp.cpp
    ggml-xdna2-kernels.cpp
    ggml-xdna2-npu.cpp
)

# Kernel build step (kernels/: IRON generator + peano kernel sources, run as an external process at model load).
# GGML_XDNA2_MLIR_AIE_ENV: python environment with mlir_aie + llvm-aie (bin/aiecc, bin/python). Without it, kernels
# can only come from the cache and weights without a cached kernel stay on the CPU.
set(GGML_XDNA2_MLIR_AIE_ENV "$ENV{MLIR_AIE_ENV}" CACHE PATH "ggml-xdna2: mlir-aie python environment used to build NPU kernels")
set(GGML_XDNA2_KERNEL_DEFS
    GGML_XDNA2_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    GGML_XDNA2_MLIR_AIE_ENV="${GGML_XDNA2_MLIR_AIE_ENV}")
target_compile_definitions(ggml-xdna2 PRIVATE ${GGML_XDNA2_KERNEL_DEFS})
target_link_libraries(ggml-xdna2 PRIVATE Threads::Threads)

target_compile_features(ggml-xdna2 PRIVATE cxx_std_17)
target_include_directories(ggml-xdna2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The host reference path emulates the NPU kernel's f32 arithmetic (separate multiply and add, no FMA)
set_source_files_properties(ggml-xdna2-compute.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

set_source_files_properties(ggml-xdna2-bfp.cpp PROPERTIES COMPILE_OPTIONS "-O3")

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
    message(STATUS "ggml-xdna2: XRT not found - building without NPU support (host reference compute only)")
endif()

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

    # kernel cache exerciser (builds NPU kernels, does not touch the NPU)
    add_executable(xdna2-kernel-tool tests/xdna2-kernel-tool.cpp ggml-xdna2-kernels.cpp)
    target_compile_features(xdna2-kernel-tool PRIVATE cxx_std_17)
    target_include_directories(xdna2-kernel-tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(xdna2-kernel-tool PRIVATE ${GGML_XDNA2_KERNEL_DEFS})
    target_link_libraries(xdna2-kernel-tool PRIVATE Threads::Threads)
endif()
