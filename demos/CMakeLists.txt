cmake_minimum_required(VERSION 3.21)
project(demos LANGUAGES C CXX)

# Enable CUDA if available
if(HPC_HAS_CUDA OR HPC_ENABLE_CUDA)
  enable_language(CUDA)
  set(HPC_HAS_CUDA TRUE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add a single demo executable (auto-detects if it needs CUDA)
function(add_demo_target DEMO_NAME DEMO_DIR OUTPUT_SUBDIR)
  # Collect all source files
  file(GLOB_RECURSE DEMO_SOURCES 
    ${DEMO_DIR}/*.c 
    ${DEMO_DIR}/*.cc 
    ${DEMO_DIR}/*.cpp
  )
  
  file(GLOB_RECURSE DEMO_CUDA_SOURCES ${DEMO_DIR}/*.cu)
  file(GLOB_RECURSE DEMO_HEADERS 
    ${DEMO_DIR}/*.h 
    ${DEMO_DIR}/*.hh 
    ${DEMO_DIR}/*.hpp 
    ${DEMO_DIR}/*.cuh
  )
  
  # Skip if no sources
  if(NOT DEMO_SOURCES AND NOT DEMO_CUDA_SOURCES)
    return()
  endif()
  
  # Determine if this is a CUDA demo
  set(IS_CUDA_DEMO OFF)
  if(DEMO_CUDA_SOURCES)
    if(NOT HPC_HAS_CUDA)
      message(STATUS "  [SKIP] ${DEMO_NAME} (CUDA not available)")
      return()
    endif()
    set(IS_CUDA_DEMO ON)
    list(APPEND DEMO_SOURCES ${DEMO_CUDA_SOURCES})
  endif()
  
  # Create executable
  add_executable(${DEMO_NAME} ${DEMO_SOURCES} ${DEMO_HEADERS})
  target_link_libraries(${DEMO_NAME} PRIVATE hpc)
  
  # Set output directory
  set_target_properties(${DEMO_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${OUTPUT_SUBDIR}
    FOLDER "host/${OUTPUT_SUBDIR}"
  )
  
  # CUDA-specific settings
  if(IS_CUDA_DEMO)
    set_target_properties(${DEMO_NAME} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
  endif()
  
  print_info("[TRACE] Adding Executable: ${DEMO_NAME}\n" "92")
  
  # Increment counter in parent scope
  math(EXPR DEMO_COUNT "${DEMO_COUNT} + 1")
  set(DEMO_COUNT ${DEMO_COUNT} PARENT_SCOPE)
endfunction()

# Build Demos
set(DEMO_COUNT 0)
print_info("◆ DEMOS\n" "0")

# --- General host (mixed backends) ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/host)
  file(GLOB HOST_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/host/*)
  
  if(HOST_DIRS)
    foreach(demo_dir ${HOST_DIRS})
      if(IS_DIRECTORY ${demo_dir})
        get_filename_component(demo_name ${demo_dir} NAME)
        add_demo_target("${demo_name}" ${demo_dir} "host")
        set(DEMO_COUNT ${DEMO_COUNT} PARENT_SCOPE)
      endif()
    endforeach()
  endif()
endif()

# --- CUDA-Specific Demos ---
if(HPC_HAS_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/device)
  file(GLOB CUDA_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/device/*)
  
  if(CUDA_DIRS)
    foreach(demo_dir ${CUDA_DIRS})
      if(IS_DIRECTORY ${demo_dir})
        get_filename_component(demo_name ${demo_dir} NAME)
        add_demo_target("${demo_name}_cuda" ${demo_dir} "device")
        set(DEMO_COUNT ${DEMO_COUNT} PARENT_SCOPE)
      endif()
    endforeach()
  endif()
elseif(NOT HPC_HAS_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/device)
  print_info("[WARN] CUDA demos found but CUDA is not enabled\n" "93")
endif()

# Summary
if(DEMO_COUNT GREATER 0)
  print_info("[INFO] Total executables: ${DEMO_COUNT}\n" "92")
  print_info("[INFO] Output directories:\n" "92")
  print_info("  - CPU(host): ${CMAKE_BINARY_DIR}/bin/host\n" "90")
  if(HPC_HAS_CUDA)
    print_info("  - CUDA(device):     ${CMAKE_BINARY_DIR}/bin/device\n" "90")
  endif()
else()
  print_info("[WARN] No demo executables found\n" "93")
endif()

print_info("◆ END OF DEMOS " "90")
print_info("=======================================\n" "90")
print_info("\n" "0")
