cmake_minimum_required(VERSION 3.21)
project(benchmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEMO_COUNT 0)
print_info("◆ BENCHMARKING \n" "0")

# L1 benchmarks
file(GLOB L1_BENCHMARK_SOURCES l1/*.cpp)
foreach (source_file ${L1_BENCHMARK_SOURCES})
  get_filename_component(basename ${source_file} NAME_WE)
  add_executable(${basename} ${source_file})
  target_link_libraries(${basename} PRIVATE 
    hpc 
    benchmark::benchmark
    # benchmark::benchmark_main
  )
  set_target_properties(${basename} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
  )
  math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
  set(DEMO_COUNT ${DEMO_COUNT} PARENT_SCOPE)
endforeach()

# L3 benchmarks
file(GLOB L3_BENCHMARK_SOURCES l3/*.cpp)
foreach (source_file ${L3_BENCHMARK_SOURCES})
  get_filename_component(basename ${source_file} NAME_WE)
  add_executable(${basename} ${source_file})
  target_link_libraries(${basename} PRIVATE 
    hpc 
    benchmark::benchmark
    # benchmark::benchmark_main
  )
  set_target_properties(${basename} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
  )
  math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
  set(DEMO_COUNT ${DEMO_COUNT} PARENT_SCOPE)
endforeach()

# Summary
if(DEMO_COUNT GREATER 0)
  print_info("[INFO] Total benchmarks: ${DEMO_COUNT}\n" "92")
  print_info("[INFO] Output directories: ${CMAKE_BINARY_DIR}/bin/benchmarks\n" "92")
else()
  print_info("[WARN] No benchmark executables found\n" "93")
endif()

print_info("◆ END OF BENCHMARKING " "90")
print_info("================================ \n" "90")
print_info("\n" "0")
