# Define the main library target

# Build hpc library
add_library(hpc ${HPC_SOURCES} ${HPC_HEADERS} ${HPC_CUDA_HEADERS} ${HPC_CUDA_SOURCES})
  
set_target_properties(hpc PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
)
  
target_include_directories(hpc PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
  
if(HPC_HAS_PYTORCH)
  target_include_directories(hpc PUBLIC ${Python3_INCLUDE_DIRS})
endif()

# Define feature macros based on what's enabled
if(HPC_HAS_CUDA)
  target_compile_definitions(hpc PUBLIC HPC_ENABLE_CUDA HPC_HAS_CUDA)
endif()

if(HPC_HAS_OPENMP)
  target_compile_definitions(hpc PUBLIC HPC_ENABLE_OPENMP HPC_HAS_OPENMP)
endif()

if(HPC_HAS_SIMD)
  target_compile_definitions(hpc PUBLIC HPC_ENABLE_SIMD HPC_HAS_SIMD)
  
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

if(HPC_HAS_MPI)
  target_compile_definitions(hpc PUBLIC HPC_ENABLE_MPI HPC_HAS_MPI)
endif()

if(HPC_HAS_PYTORCH)
  target_compile_definitions(hpc PUBLIC HPC_ENABLE_PYTORCH HPC_HAS_PYTORCH)
endif()

if(HPC_HAS_CUBLAS)
  target_compile_definitions(hpc PUBLIC HPC_ENABLE_CUBLAS HPC_HAS_CUBLAS)
endif()

if(HPC_HAS_OPENBLAS)
  target_compile_definitions(hpc PUBLIC 
    HPC_ENABLE_OPENBLAS 
    OPENBLAS_NEW_LAPACK
    OPENBLAS_LAPACK_ILP64
  )
endif()

if(HPC_HAS_ACCELERATE)
  target_compile_definitions(hpc PUBLIC 
    HPC_ENABLE_ACCELERATE 
    ACCELERATE_NEW_LAPACK 
    ACCELERATE_LAPACK_ILP64
  )
endif()

# Link libraries
if(HPC_HAS_OPENMP)
  target_link_libraries(hpc PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()
  
if(HPC_HAS_MPI)
  target_link_libraries(hpc PUBLIC MPI::MPI_C MPI::MPI_CXX)
endif()
  
if(HPC_HAS_PYTORCH)
  target_link_libraries(hpc PRIVATE Python3::Python "${TORCH_LIBRARIES}")
endif()
  
if(HPC_HAS_CUDA)
  target_link_libraries(hpc PUBLIC CUDA::cudart)
endif()

if(HPC_HAS_CUBLAS)
  target_link_libraries(hpc PUBLIC CUDA::cublas)
endif()

if(HPC_HAS_OPENBLAS)
  target_link_libraries(hpc PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

if(HPC_HAS_ACCELERATE)
  target_link_libraries(hpc PUBLIC ${ACCELERATE_LIB})
endif()
  
# Export compile features
target_compile_features(hpc PUBLIC cxx_std_17)
