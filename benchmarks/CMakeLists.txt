cmake_minimum_required(VERSION 3.21)
project(benchmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# L1 benchmarks
add_executable(axpy l1/axpy.cpp)
target_link_libraries(axpy PRIVATE 
  hpc 
  benchmark::benchmark
  # benchmark::benchmark_main
)
set_target_properties(axpy PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)

# L3 benchmarks
add_executable(mmul l3/mmul.cpp)
target_link_libraries(mmul PRIVATE 
  hpc 
  benchmark::benchmark
  # benchmark::benchmark_main
)
set_target_properties(mmul PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)
