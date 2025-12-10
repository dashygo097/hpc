cmake_minimum_required(VERSION 3.21)
project(hpc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add cmake module path
set(_HPC_ORIGINAL_MODULE_PATH ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Generate and include build configuration
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  include("${CMAKE_BINARY_DIR}/config.cmake")
endif()

# Include modular cmake components
include(${CMAKE_SOURCE_DIR}/cmake/Platforms.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CompilerSetup.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Helpers.cmake)

# Find all dependencies based on configuration
include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)

# Collect source files
include(${CMAKE_SOURCE_DIR}/cmake/SourceCollections.cmake)

# Build the main library
include(${CMAKE_SOURCE_DIR}/cmake/LibraryTarget.cmake)

# Print build summary
include(${CMAKE_SOURCE_DIR}/cmake/BuildSummaries.cmake)

# Apps
add_subdirectory(demos)

# Benchmarks
if(ENABLE_BENCHMARKING)
  add_subdirectory(benchmarks)
endif()

set(CMAKE_MODULE_PATH ${_HPC_ORIGINAL_MODULE_PATH})
unset(_HPC_ORIGINAL_MODULE_PATH)
