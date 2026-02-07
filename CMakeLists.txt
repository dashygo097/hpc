cmake_minimum_required(VERSION 3.21)
project(hpc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HPC_STANDALONE ON)
else()
  set(HPC_STANDALONE OFF)
  message(STATUS "HPC: Building as subdirectory")
endif()

# Add cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

# Generate and include build configuration
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/config.cmake")
  include("${CMAKE_CURRENT_BINARY_DIR}/config.cmake")
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

if(HPC_STANDALONE)
  # Apps
  if(HPC_ENABLE_DEMOS)
    add_subdirectory(demos)
  endif()

  # Benchmarks
  if(HPC_ENABLE_BENCHMARKING)
    add_subdirectory(benchmarks)
  endif()
endif()
