cmake_minimum_required(VERSION 3.21)
project(demos LANGUAGES C CXX)

if (HAS_CUDA) 
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs)

file(GLOB project ./examples/*)
file(GLOB cuda-project ./cuda-examples/*)

print_info("――――――――――――――――――APP SCOPE――――――――――――――――――" "0")

foreach(exec ${project})
  if(IS_DIRECTORY ${exec})
    math(EXPR index "${index} + 1" OUTPUT_FORMAT DECIMAL)
    get_filename_component(exec-name ${exec} NAME)
    file(GLOB_RECURSE ${exec}-src ${exec}/*.c ${exec}/*.cc ${exec}/*.cpp)
    file(GLOB_RECURSE ${exec}-hdr ${exec}/*.h ${exec}/*.hh ${exec}/*.hpp)

    add_executable(${exec-name} ${${exec}-src} ${${exec}-hdr})
    target_link_libraries(${exec-name} PRIVATE hpc)
    
    set_target_properties(${exec-name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs/demos
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs/demos
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs/demos
    )

    print_info("[TRACE] Adding: ${exec-name}" "0")

  endif()
endforeach()

foreach(exec ${cuda-project})
  if(IS_DIRECTORY ${exec} AND HAS_CUDA)
    math(EXPR index "${index} + 1" OUTPUT_FORMAT DECIMAL)
    get_filename_component(exec-name ${exec} NAME)

    file(GLOB_RECURSE ${exec}-src ${exec}/*.c ${exec}/*.cc ${exec}/*.cpp ${exec}/*.cu)
    file(GLOB_RECURSE ${exec}-hdr ${exec}/*.h ${exec}/*.hh ${exec}/*.hpp)

    add_executable(${exec-name} ${${exec}-src} ${${exec}-hdr})
    target_link_libraries(${exec-name} PRIVATE hpc)
    
    set_target_properties(${exec-name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs/cuda-demos
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs/cuda-demos
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/outputs/cuda-demos
    )

    print_info("[TRACE] Adding: ${exec-name}" "0")
  endif()
endforeach()

print_info("[INFO] CMake configuration completed" "92")
print_info("[INFO] Total ${index} exec files will be generated" "92")
print_info("------------END OF CONFIGURATION-------------" "90")
