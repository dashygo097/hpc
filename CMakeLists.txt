cmake_minimum_required(VERSION 3.21)
project(hpc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Generate and include build configuration
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  include("${CMAKE_BINARY_DIR}/config.cmake")
endif()

# Include modular cmake components
include(Platforms)
include(CompilerSetup)

# Build the main library
include(Helpers)

# Find all dependencies based on configuration
include(Dependencies)

# HPC lib
file(GLOB_RECURSE HPC_SOURCES src/*.c src/*.cc src/*.cpp)
file(GLOB_RECURSE HPC_HEADERS include/*.h include/*.hh include/*.hpp)

if (HAS_CUDA)
  file(GLOB_RECURSE HPC_CUDA_HEADERS include/hpc/cuda/*.cuh)
  file(GLOB_RECURSE HPC_CUDA_SOURCES src/cuda/kernels/*.cu)
  add_library(hpc ${HPC_SOURCES} ${HPC_HEADERS} ${HPC_CUDA_HEADERS} ${HPC_CUDA_SOURCES})

  set_target_properties(hpc PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_include_directories(hpc PUBLIC 
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_include_directories(hpc PUBLIC ${Python3_INCLUDE_DIRS})
  set_source_files_properties(${HPC_CUDA_SOURCES} PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

  target_link_libraries(hpc PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  target_link_libraries(hpc PUBLIC MPI::MPI_C MPI::MPI_CXX)
  target_link_libraries(hpc PRIVATE Python3::Python "${TORCH_LIBRARIES}")
  set_target_properties(hpc PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON 
  )

  set_target_properties(hpc PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

else()
  add_library(hpc INTERFACE)
  target_include_directories(hpc INTERFACE 
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(hpc INTERFACE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  target_link_libraries(hpc INTERFACE MPI::MPI_C MPI::MPI_CXX)
  target_link_libraries(hpc INTERFACE Python3::Python "${TORCH_LIBRARIES}")

  if (APPLE)
    add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
    find_library(ACCELERATE_LIB Accelerate)
    target_link_libraries(hpc INTERFACE ${ACCELERATE_LIB})
  endif()
endif()

# Stastics
print_info("――――――――――――――――――LIB SCOPE――――――――――――――――――" "0")
print_info("[INFO] Openmp C version: ${OpenMP_C_VERSION}" "90")
print_info("[INFO] Openmp CXX version: ${OpenMP_CXX_VERSION}" "90")
print_info("[INFO] MPI C version: ${MPI_C_VERSION}" "90")
print_info("[INFO] MPI CXX version: ${MPI_CXX_VERSION}" "90")
print_info("[INFO] PyTorch version: ${Torch_VERSION}" "90")
if (HAS_CUDA)
  print_info("[INFO] CUDA compiler: ${CMAKE_CUDA_COMPILER}" "90")
else()
  print_info("[INFO] CUDA not available" "90")
endif()

make_paths_relative(REL_SOURCES HPC_SOURCES)
make_paths_relative(REL_HEADERS HPC_HEADERS)

if (HAS_CUDA)
  make_paths_relative(REL_CUDA_HEADERS HPC_CUDA_HEADERS)
  make_paths_relative(REL_CUDA_SOURCES HPC_CUDA_SOURCES)
  make_preview_string(REL_CUDA_HEADERS 3)
  print_info("[TRACE] CUDA headers: ${PREVIOUS_SCOPE_VAR}" "94")
  make_preview_string(REL_CUDA_SOURCES 3)
  print_info("[TRACE] CUDA sources: ${PREVIOUS_SCOPE_VAR}" "96")

else()
  set(REL_CUDA_HEADERS "")
endif()

make_preview_string(REL_HEADERS 3)
print_info("[TRACE] Header files: ${PREVIOUS_SCOPE_VAR}" "94")
make_preview_string(REL_SOURCES 3)
print_info("[TRACE] Source files: ${PREVIOUS_SCOPE_VAR}" "96")


# Testing
include(GoogleTest)
enable_testing()

# Apps
add_subdirectory(tests)
add_subdirectory(demos)
