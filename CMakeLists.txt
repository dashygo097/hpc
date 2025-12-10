cmake_minimum_required(VERSION 3.21)

# Detect standalone vs subdirectory build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hpc LANGUAGES C CXX)
  set(HPC_STANDALONE ON)
else()
  set(HPC_STANDALONE OFF)
  message(STATUS "HPC:  Building as subdirectory")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  include("${CMAKE_BINARY_DIR}/config.cmake")
elseif(HPC_STANDALONE)
  message(STATUS "Using default HPC configuration")
endif()

include(Platforms)
include(CompilerSetup)
include(Helpers)

if(HPC_STANDALONE)
  include(Dependencies)
endif()

include(SourceCollections)
include(LibraryTarget)

if(HPC_STANDALONE)
  include(BuildSummaries)
  
  # Apps
  add_subdirectory(demos)
  
  # Benchmarks
  if(ENABLE_BENCHMARKING)
    add_subdirectory(benchmarks)
  endif()
endif()
