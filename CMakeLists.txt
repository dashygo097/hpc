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

# Collect source files
include(SourceCollections)

# Build the main library
include(LibraryTarget)

# Print build summary
include(BuildSummaries)

# Apps
add_subdirectory(demos)
