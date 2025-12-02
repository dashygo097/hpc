# Compiler setup and ccache configuration

if(USE_CCACHE)
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

# Platform-specific compiler flags
if(APPLE AND ENABLE_ACCELERATE)
  # macOS-specific flags
  add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
endif()
