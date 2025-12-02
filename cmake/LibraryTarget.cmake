# Define the main library target

if(HAS_CUDA)
  # Build as a regular library with CUDA sources
  add_library(hpc ${HPC_SOURCES} ${HPC_HEADERS} ${HPC_CUDA_HEADERS} ${HPC_CUDA_SOURCES})
  
  set_target_properties(hpc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
  
  target_include_directories(hpc PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  
  if(HAS_PYTORCH)
    target_include_directories(hpc PUBLIC ${Python3_INCLUDE_DIRS})
  endif()
  
  # Link dependencies
  if(HAS_OPENMP)
    target_link_libraries(hpc PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  endif()
  
  if(HAS_MPI)
    target_link_libraries(hpc PUBLIC MPI::MPI_C MPI::MPI_CXX)
  endif()
  
  if(HAS_PYTORCH)
    target_link_libraries(hpc PRIVATE Python3::Python "${TORCH_LIBRARIES}")
  endif()
  
else()
  # Build as header-only interface library
  add_library(hpc INTERFACE)
  
  target_include_directories(hpc INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  
  if(HAS_OPENMP)
    target_link_libraries(hpc INTERFACE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  endif()
  
  if(HAS_MPI)
    target_link_libraries(hpc INTERFACE MPI::MPI_C MPI::MPI_CXX)
  endif()
  
  if(HAS_PYTORCH)
    target_link_libraries(hpc INTERFACE Python3::Python "${TORCH_LIBRARIES}")
  endif()
  
  if(HAS_ACCELERATE)
    target_link_libraries(hpc INTERFACE ${ACCELERATE_LIB})
  endif()
endif()
