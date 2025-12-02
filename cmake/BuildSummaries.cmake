# Print build configuration summary

include(Helpers)

print_info("――――――――――――――――――BUILD SUMMARY――――――――――――――――――" "0")

# Platform
print_info("[INFO] Platform: ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})" "91")
print_info("[INFO] Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" "96")

# Feature flags
print_info("[INFO] Features:" "95")

if(HAS_OPENMP)
  print_info("  ✓ OpenMP: ${OpenMP_CXX_VERSION}" "92")
else()
  print_info("  ✗ OpenMP: disabled" "90")
endif()

if(HAS_MPI)
  print_info("  ✓ MPI: ${MPI_CXX_VERSION}" "92")
else()
  print_info("  ✗ MPI: disabled" "90")
endif()

if(HAS_SIMD)
  print_info("  ✓ SIMD: ${SIMD_TYPE}" "92")
else()
  print_info("  ✗ SIMD: disabled" "90")
endif()

if(HAS_CUDA)
  print_info("  ✓ CUDA: ${CUDAToolkit_VERSION}" "92")
  print_info("    Compiler: ${CMAKE_CUDA_COMPILER}" "90")
else()
  print_info("  ✗ CUDA: disabled" "90")
endif()

if (HAS_PYBIND11)
  print_info("  ✓ pybind11: ${pybind11_VERSION}" "92")
else()
  print_info("  ✗ pybind11: disabled" "90")
endif()

if(HAS_PYTORCH)
  print_info("  ✓ PyTorch: ${Torch_VERSION}" "92")
else()
  print_info("  ✗ PyTorch: disabled" "90")
endif()

if(HAS_ACCELERATE)
  print_info("  ✓ Apple Accelerate framework" "92")
endif()

if(CCACHE_PROGRAM)
  print_info("  ✓ ccache: ${CCACHE_PROGRAM}" "92")
endif()

# Source files
print_info("" "0")
make_paths_relative(REL_SOURCES HPC_SOURCES)
make_paths_relative(REL_HEADERS HPC_HEADERS)

make_preview_string(REL_HEADERS 3)
print_info("[TRACE] Headers: ${PREVIOUS_SCOPE_VAR}" "94")

make_preview_string(REL_SOURCES 3)
print_info("[TRACE] Sources: ${PREVIOUS_SCOPE_VAR}" "96")

if(HAS_CUDA)
  make_paths_relative(REL_CUDA_HEADERS HPC_CUDA_HEADERS)
  make_paths_relative(REL_CUDA_SOURCES HPC_CUDA_SOURCES)
  
  make_preview_string(REL_CUDA_HEADERS 3)
  print_info("[TRACE] CUDA headers: ${PREVIOUS_SCOPE_VAR}" "94")
  
  make_preview_string(REL_CUDA_SOURCES 3)
  print_info("[TRACE] CUDA sources: ${PREVIOUS_SCOPE_VAR}" "96")
endif()

print_info("――――――――――――――――――――――――――――――――――――――――――――――" "0")
