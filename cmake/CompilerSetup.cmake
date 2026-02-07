# Compiler setup and ccache configuration

if(HPC_USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(ENV{CCACHE_CUDA} "1")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  else()
    message(STATUS "ccache requested but not found")
  endif()
endif()

if(HPC_USE_HIGH_LEVEL_OPTIMIZATIONS)
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(CMAKE_CXX_FLAGS_RELEASE 
      "${CMAKE_CXX_FLAGS_RELEASE} \
      -O3 \
      -mtune=native \
      -ffast-math \
      -funroll-loops \
      -fvectorize \
      -fslp-vectorize \
      -ffp-contract=fast \
      -fno-math-errno \
      -fno-trapping-math"
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    message(STATUS "High-level optimizations enabled for Apple ARM64")
  else()
    message(STATUS "High-level optimizations are only configured for Apple ARM64 in this setup")
  endif()
endif()

# Platform-specific compiler flags
if(APPLE AND HPC_ENABLE_ACCELERATE)
  # macOS-specific flags
  add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
endif()
