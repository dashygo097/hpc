# Platform detection and platform-specific defaults

if(APPLE)
  message(STATUS "Building on macOS")
  
  # macOS defaults
  set(ENABLE_CUDA_DEFAULT "OFF")
  set(ENABLE_OPENMP_DEFAULT "ON")
  set(ENABLE_MPI_DEFAULT "ON")
  set(ENABLE_PYTORCH_DEFAULT "ON")
  set(ENABLE_TESTS_DEFAULT "ON")
  set(ENABLE_DEMOS_DEFAULT "ON")
  set(USE_CCACHE_DEFAULT "ON")
  set(ENABLE_PCH_DEFAULT "ON")
  
elseif(UNIX)
  message(STATUS "Building on Unix-like system")
  
  # Linux defaults
  set(ENABLE_CUDA_DEFAULT "ON")
  set(ENABLE_OPENMP_DEFAULT "ON")
  set(ENABLE_MPI_DEFAULT "ON")
  set(ENABLE_PYTORCH_DEFAULT "ON")
  set(ENABLE_TESTS_DEFAULT "ON")
  set(ENABLE_DEMOS_DEFAULT "ON")
  set(USE_CCACHE_DEFAULT "ON")
  set(ENABLE_PCH_DEFAULT "ON")
  
elseif(WIN32)
  message(STATUS "Building on Windows")
  
  # Windows defaults
  set(ENABLE_CUDA_DEFAULT "ON")
  set(ENABLE_OPENMP_DEFAULT "ON")
  set(ENABLE_MPI_DEFAULT "OFF")
  set(ENABLE_PYTORCH_DEFAULT "ON")
  set(ENABLE_TESTS_DEFAULT "ON")
  set(ENABLE_DEMOS_DEFAULT "ON")
  set(USE_CCACHE_DEFAULT "OFF")
  set(ENABLE_PCH_DEFAULT "ON")
  
else()
  message(FATAL_ERROR "Unsupported platform")
endif()

# Create platform config string for config.cmake. in
set(PLATFORM_CONFIG "# Platform: ${CMAKE_SYSTEM_NAME}")
