cmake_minimum_required(VERSION 3.21)
project(hpc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# test
include(GoogleTest)
enable_testing()

# CUDA

include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAS_CUDA TRUE)

    if(WIN32)
        message(STATUS "Building on Windows")
    elseif(UNIX AND NOT APPLE)
        message(STATUS "Building on Unix-like system")
    endif()

    message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")

    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")

else()
  message(WARNING "CUDA compiler not found; CUDA support will be disabled.")
    set(HAS_CUDA FALSE)
endif()

if(HAS_CUDA)
    if(NOT TARGET CUDA::cudart)
        message(FATAL_ERROR "CUDA::cudart target not found — missing CUDA Toolkit?")
    endif()
else()
    message(STATUS "CUDA support is disabled.")
endif()


# openmp
if(APPLE)
  set(OPENMP_PATH "/opt/homebrew/opt/libomp")
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OPENMP_PATH}/include")
  set(OpenMP_CXX_FLAGS "${OpenMP_C_FLAGS}")
  set(OpenMP_C_LIB_NAMES "omp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "${OPENMP_PATH}/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)


# 3rdparty
add_subdirectory(3rdparty/googletest)
include_directories(3rdparty)

# hpc

file(GLOB_RECURSE HPC_SOURCES src/*.cpp src/*.c src/*.cc)
file(GLOB_RECURSE HPC_HEADERS include/*.h include/*.hpp)

add_library(hpc ${HPC_SOURCES} ${HPC_HEADERS})
target_include_directories(hpc PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(hpc PUBLIC OpenMP::OpenMP_CXX MPI::MPI_C MPI::MPI_CXX)

if (HAS_CUDA)
  file(GLOB_RECURSE HPC_CUDA_SOURCES src/cuda/*.cu)
  target_link_libraries(hpc PUBLIC ${HPC_CUDA_SOURCES})
  
  set_target_properties(hpc PROPERTIES 
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "native"
  )
  target_link_libraries(hpc PUBLIC CUDA::cudart)
endif()


set_target_properties(hpc PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Colored printing
function(print_info MSG COLOR)
  if(CMAKE_COLOR_MAKEFILE)
    execute_process(COMMAND printf "\\033[1;${COLOR}m${MSG}\\033[0m\n")
  else()
    message(STATUS "${MSG}")
  endif()
endfunction()

function(make_paths_relative out_var)
    set(result "")
    foreach(path IN LISTS ${ARGV1})
        file(RELATIVE_PATH rel "${CMAKE_SOURCE_DIR}" "${path}")
        list(APPEND result "${rel}")
    endforeach()
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

function(make_preview_string LIST_VAR N)
  set(preview "")

  list(LENGTH ${LIST_VAR} list_length)

  foreach(item IN LISTS ${LIST_VAR})
    list(APPEND preview "${item}")
    if("${preview}" MATCHES "^.*;.*;.*$")
      break()
    endif()
  endforeach()

  if(list_length GREATER N)
    math(EXPR remaining_items "${list_length} - ${N}")
    list(APPEND preview "; ...(+${remaining_items} more)")
  endif()

  set(PREVIOUS_SCOPE_VAR "${preview}" PARENT_SCOPE)
endfunction()

# stastics
print_info("――――――――――――――――――LIB SCOPE――――――――――――――――――" "0")
print_info("[INFO] Openmp C version: ${OpenMP_C_VERSION}" "90")
print_info("[INFO] Openmp CXX version: ${OpenMP_CXX_VERSION}" "90")
print_info("[INFO] MPI C version: ${MPI_C_VERSION}" "90")
print_info("[INFO] MPI CXX version: ${MPI_CXX_VERSION}" "90")
if (HAS_CUDA)
  print_info("[INFO] CUDA compiler: ${CMAKE_CUDA_COMPILER}" "90")
else()
  print_info("[INFO] CUDA not available" "90")
endif()

make_paths_relative(REL_SOURCES HPC_SOURCES)
make_paths_relative(REL_HEADERS HPC_HEADERS)

if (HAS_CUDA)
  file(GLOB_RECURSE HPC_CUDA_HEADERS include/cuda/*.h include/cuda/*.hpp)
  make_paths_relative(REL_CUDA_HEADERS HPC_CUDA_HEADERS)
  make_paths_relative(REL_CUDA_SOURCES HPC_CUDA_SOURCES)
  make_preview_string(REL_CUDA_HEADERS 3)
  make_preview_string(REL_CUDA_SOURCES 3)
  print_info("[TRACE] CUDA headers: ${REL_CUDA_HEADERS}" "94")
  print_info("[TRACE] CUDA sources: ${REL_CUDA_SOURCES}" "96")
else()
  set(REL_CUDA_HEADERS "")
endif()

make_preview_string(REL_HEADERS 3)
print_info("[TRACE] Header files: ${REL_HEADERS}" "94")
make_preview_string(REL_SOURCES 3)
print_info("[TRACE] Source files: ${REL_SOURCES}" "96")



# Apps
add_subdirectory(tests)
add_subdirectory(demos)

