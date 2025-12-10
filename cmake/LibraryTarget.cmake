# Define the main library target

# Build hpc library
add_library(hpc ${HPC_SOURCES} ${HPC_HEADERS} ${HPC_CUDA_HEADERS} ${HPC_CUDA_SOURCES})
  
set_target_properties(hpc PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
  
target_include_directories(hpc PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
  
if(HAS_PYTORCH)
  target_include_directories(hpc PUBLIC ${Python3_INCLUDE_DIRS})
endif()

# Define feature macros based on what's enabled
if(HAS_CUDA)
  target_compile_definitions(hpc PUBLIC ENABLE_CUDA HAS_CUDA)
endif()

if(HAS_OPENMP)
  target_compile_definitions(hpc PUBLIC ENABLE_OPENMP HAS_OPENMP)
endif()

if(HAS_SIMD)
  target_compile_definitions(hpc PUBLIC ENABLE_SIMD HAS_SIMD)
  
  # Platform-specific SIMD defines
  if(SIMD_TYPE STREQUAL "Apple")
    target_compile_definitions(hpc PUBLIC SIMD_APPLE)
  elseif(SIMD_TYPE STREQUAL "NEON")
    target_compile_definitions(hpc PUBLIC SIMD_NEON)
  elseif(SIMD_TYPE STREQUAL "AVX")
    target_compile_definitions(hpc PUBLIC SIMD_AVX)
  elseif(SIMD_TYPE STREQUAL "SSE")
    target_compile_definitions(hpc PUBLIC SIMD_SSE)
  endif()
endif()

if(HAS_MPI)
  target_compile_definitions(hpc PUBLIC ENABLE_MPI HAS_MPI)
endif()

if(HAS_PYTORCH)
  target_compile_definitions(hpc PUBLIC ENABLE_PYTORCH HAS_PYTORCH)
endif()

if(HAS_ACCELERATE)
  target_compile_definitions(hpc PUBLIC 
    ENABLE_ACCELERATE 
    ACCELERATE_NEW_LAPACK 
    ACCELERATE_LAPACK_ILP64
  )
endif()

# Link libraries
if(HAS_OPENMP)
  target_link_libraries(hpc PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()
  
if(HAS_MPI)
  target_link_libraries(hpc PUBLIC MPI::MPI_C MPI::MPI_CXX)
endif()
  
if(HAS_PYTORCH)
  target_link_libraries(hpc PRIVATE Python3::Python "${TORCH_LIBRARIES}")
endif()
  
if(HAS_CUDA)
  target_link_libraries(hpc PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(HAS_ACCELERATE)
  target_link_libraries(hpc PUBLIC ${ACCELERATE_LIB})
endif()
  
# Export compile features
target_compile_features(hpc PUBLIC cxx_std_17)
