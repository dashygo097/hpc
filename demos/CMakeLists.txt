cmake_minimum_required(VERSION 3.21)
project(demos LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Function to create a demo executable
function(add_demo_executable DEMO_NAME DEMO_DIR BACKEND_TYPE)
  # Collect sources based on backend type
  if(BACKEND_TYPE STREQUAL "CUDA")
    file(GLOB_RECURSE DEMO_SOURCES 
      ${DEMO_DIR}/*.c 
      ${DEMO_DIR}/*.cc 
      ${DEMO_DIR}/*.cpp 
      ${DEMO_DIR}/*.cu
    )
    file(GLOB_RECURSE DEMO_HEADERS 
      ${DEMO_DIR}/*.h 
      ${DEMO_DIR}/*.hh 
      ${DEMO_DIR}/*.hpp 
      ${DEMO_DIR}/*.cuh
    )
  else()
    file(GLOB_RECURSE DEMO_SOURCES 
      ${DEMO_DIR}/*.c 
      ${DEMO_DIR}/*.cc 
      ${DEMO_DIR}/*.cpp
    )
    file(GLOB_RECURSE DEMO_HEADERS 
      ${DEMO_DIR}/*.h 
      ${DEMO_DIR}/*.hh 
      ${DEMO_DIR}/*.hpp
    )
  endif()

  # Create executable
  add_executable(${DEMO_NAME} ${DEMO_SOURCES} ${DEMO_HEADERS})
  
  # Link against hpc library
  target_link_libraries(${DEMO_NAME} PRIVATE hpc)
  
  # Set output directory based on backend type
  set_target_properties(${DEMO_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${BACKEND_TYPE}
    FOLDER "Demos/${BACKEND_TYPE}"
  )
  
  # Add backend-specific compile definitions
  if(BACKEND_TYPE STREQUAL "Sequential")
    target_compile_definitions(${DEMO_NAME} PRIVATE USE_SEQUENTIAL_BACKEND)
  elseif(BACKEND_TYPE STREQUAL "OpenMP")
    target_compile_definitions(${DEMO_NAME} PRIVATE USE_OPENMP_BACKEND)
  elseif(BACKEND_TYPE STREQUAL "SIMD")
    target_compile_definitions(${DEMO_NAME} PRIVATE USE_SIMD_BACKEND)
  elseif(BACKEND_TYPE STREQUAL "OpenMP-SIMD")
    target_compile_definitions(${DEMO_NAME} PRIVATE USE_OPENMP_SIMD_BACKEND)
  elseif(BACKEND_TYPE STREQUAL "CUDA")
    target_compile_definitions(${DEMO_NAME} PRIVATE USE_CUDA_BACKEND)
    set_target_properties(${DEMO_NAME} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
  elseif(BACKEND_TYPE STREQUAL "MPI")
    target_compile_definitions(${DEMO_NAME} PRIVATE USE_MPI_BACKEND)
  endif()
  
  # Store in parent scope for reporting
  set(DEMO_COUNT ${DEMO_COUNT} PARENT_SCOPE)
  
  print_info("[TRACE] ${BACKEND_TYPE}: ${DEMO_NAME}" "94")
endfunction()

set(DEMO_COUNT 0)

print_info("―――――――――――――――――DEMOS――――――――――――――――――" "0")

# --- Sequential Demos (always available) ---
file(GLOB SEQUENTIAL_DEMOS ${CMAKE_CURRENT_SOURCE_DIR}/sequential/*)
if(SEQUENTIAL_DEMOS)
  print_info("[INFO] Sequential Backend Demos:" "92")
  foreach(demo_dir ${SEQUENTIAL_DEMOS})
    if(IS_DIRECTORY ${demo_dir})
      get_filename_component(demo_name ${demo_dir} NAME)
      add_demo_executable("${demo_name}_seq" ${demo_dir} "Sequential")
      math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
    endif()
  endforeach()
endif()

# --- OpenMP Demos ---
if(ENABLE_OPENMP AND HAS_OPENMP)
  file(GLOB OPENMP_DEMOS ${CMAKE_CURRENT_SOURCE_DIR}/openmp/*)
  if(OPENMP_DEMOS)
    print_info("[INFO] OpenMP Backend Demos:" "92")
    foreach(demo_dir ${OPENMP_DEMOS})
      if(IS_DIRECTORY ${demo_dir})
        get_filename_component(demo_name ${demo_dir} NAME)
        add_demo_executable("${demo_name}_omp" ${demo_dir} "OpenMP")
        math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
      endif()
    endforeach()
  endif()
endif()

# --- SIMD Demos ---
if(ENABLE_SIMD)
  file(GLOB SIMD_DEMOS ${CMAKE_CURRENT_SOURCE_DIR}/simd/*)
  if(SIMD_DEMOS)
    print_info("[INFO] SIMD Backend Demos:" "92")
    foreach(demo_dir ${SIMD_DEMOS})
      if(IS_DIRECTORY ${demo_dir})
        get_filename_component(demo_name ${demo_dir} NAME)
        add_demo_executable("${demo_name}_simd" ${demo_dir} "SIMD")
        math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
      endif()
    endforeach()
  endif()
endif()

# --- CUDA Demos ---
if(ENABLE_CUDA AND HAS_CUDA)
  file(GLOB CUDA_DEMOS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*)
  if(CUDA_DEMOS)
    print_info("[INFO] CUDA Backend Demos:" "92")
    foreach(demo_dir ${CUDA_DEMOS})
      if(IS_DIRECTORY ${demo_dir})
        get_filename_component(demo_name ${demo_dir} NAME)
        add_demo_executable("${demo_name}_cuda" ${demo_dir} "CUDA")
        math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
      endif()
    endforeach()
  endif()
endif()

# --- MPI Demos ---
if(ENABLE_MPI AND HAS_MPI)
  file(GLOB MPI_DEMOS ${CMAKE_CURRENT_SOURCE_DIR}/mpi/*)
  if(MPI_DEMOS)
    print_info("[INFO] MPI Backend Demos:" "92")
    foreach(demo_dir ${MPI_DEMOS})
      if(IS_DIRECTORY ${demo_dir})
        get_filename_component(demo_name ${demo_dir} NAME)
        add_demo_executable("${demo_name}_mpi" ${demo_dir} "MPI")
        math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
      endif()
    endforeach()
  endif()
endif()

# --- Benchmark Demos (compare all backends) ---
file(GLOB BENCHMARK_DEMOS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*)
if(BENCHMARK_DEMOS)
  print_info("[INFO] Benchmark Demos (Multi-Backend):" "92")
  foreach(demo_dir ${BENCHMARK_DEMOS})
    if(IS_DIRECTORY ${demo_dir})
      get_filename_component(demo_name ${demo_dir} NAME)
      
      # Benchmark demos get compiled with all available backends
      file(GLOB_RECURSE BENCH_SOURCES 
        ${demo_dir}/*.c 
        ${demo_dir}/*.cc 
        ${demo_dir}/*.cpp
      )
      
      if(HAS_CUDA)
        file(GLOB_RECURSE BENCH_CUDA_SOURCES ${demo_dir}/*.cu)
        list(APPEND BENCH_SOURCES ${BENCH_CUDA_SOURCES})
      endif()
      
      file(GLOB_RECURSE BENCH_HEADERS ${demo_dir}/*.h ${demo_dir}/*.hh ${demo_dir}/*.hpp)
      
      add_executable("benchmark_${demo_name}" ${BENCH_SOURCES} ${BENCH_HEADERS})
      target_link_libraries("benchmark_${demo_name}" PRIVATE hpc)
      
      # Enable all available backends
      target_compile_definitions("benchmark_${demo_name}" PRIVATE 
        USE_ALL_BACKENDS
        $<$<BOOL:${HAS_OPENMP}>:HAS_OPENMP>
        $<$<BOOL:${ENABLE_SIMD}>:HAS_SIMD>
        $<$<BOOL:${HAS_CUDA}>:HAS_CUDA>
        $<$<BOOL:${HAS_MPI}>:HAS_MPI>
      )
      
      if(HAS_CUDA)
        set_target_properties("benchmark_${demo_name}" PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
        )
      endif()
      
      print_info("[TRACE] Benchmark: benchmark_${demo_name}" "96")
      math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
    endif()
  endforeach()
endif()

# Final report
print_info("――――――――――――――――――――――――――――――――――――――――――――――" "0")
print_info("[INFO] Total demo executables: ${DEMO_COUNT}" "92")
print_info("[INFO] Output directory: ${CMAKE_BINARY_DIR}/bin/" "92")
print_info("――――――――――――――――――END OF DEMOS――――――――――――――――――" "90")
