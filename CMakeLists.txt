cmake_minimum_required(VERSION 3.24)
project(hpc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



# Colored printing
function(print_info text color)
    if(CMAKE_COLOR_MAKEFILE)
        execute_process(COMMAND printf "\\033[1;${color}m${text}\\033[0m\\n")
    else()
        message(STATUS "${text}")
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

print_info("――――――――――――――――――LIB SCOPE――――――――――――――――――" "0")


# hpc
file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.c src/*.cc)
file(GLOB_RECURSE HEADER_FILES include/*.h include/*.hpp)

add_library(hpc ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(hpc PUBLIC include include/hpc)

set_target_properties(hpc PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY src
    LIBRARY_OUTPUT_DIRECTORY src
)

# stastics
make_paths_relative(REL_HEADER_FILES HEADER_FILES)
make_paths_relative(REL_SOURCE_FILES SOURCE_FILES)

print_info("[TRACE] Header files: ${REL_HEADER_FILES}" "95")
print_info("[TRACE] Source files: ${REL_SOURCE_FILES}" "96")

# Apps
enable_testing()
add_subdirectory(tests)
add_subdirectory(demos)
