# Print build configuration summary

print_info("――――――――――――――――――BUILD SUMMARY――――――――――――――――――" "0")

# Platform
print_info("[INFO] Platform: ${CMAKE_SYSTEM_NAME}" "92")

# Dependencies
if(HAS_OPENMP)
  print_info("[INFO] OpenMP C: ${OpenMP_C_VERSION}" "90")
  print_info("[INFO] OpenMP CXX: ${OpenMP_CXX_VERSION}" "90")
else()
  print_info("[INFO] OpenMP: disabled" "93")
endif()

if(HAS_MPI)
  print_info("[INFO] MPI C: ${MPI_C_VERSION}" "90")
  print_info("[INFO] MPI CXX: ${MPI_CXX_VERSION}" "90")
else()
  print_info("[INFO] MPI: disabled" "93")
endif()

if (HAS_PYBIND11)
  print_info("[INFO] pybind11: ${pybind11_VERSION}" "90")
else()
  print_info("[INFO] pybind11: disabled" "93")
endif()

if(HAS_PYTORCH)
  print_info("[INFO] PyTorch: ${Torch_VERSION}" "90")
else()
  print_info("[INFO] PyTorch: disabled" "93")
endif()

if(HAS_CUDA)
  print_info("[INFO] CUDA: ${CUDAToolkit_VERSION}" "90")
  print_info("[INFO] CUDA compiler: ${CMAKE_CUDA_COMPILER}" "90")
else()
  print_info("[INFO] CUDA: disabled" "93")
endif()

if(HAS_ACCELERATE)
  print_info("[INFO] Apple Accelerate: enabled" "90")
endif()

# Source files
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
