cmake_minimum_required(VERSION 3.21)
project(hpc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(_HPC_ORIGINAL_MODULE_PATH ${CMAKE_MODULE_PATH})

# Add cmake module path
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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

# Collect source files
include(SourceCollections)

# Build the main library
include(LibraryTarget)

# Print build summary
include(BuildSummaries)

# Apps
add_subdirectory(demos)

# Benchmarks
if(ENABLE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

set(CMAKE_MODULE_PATH ${_HPC_ORIGINAL_MODULE_PATH})
unset(_HPC_ORIGINAL_MODULE_PATH)
