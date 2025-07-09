cmake_minimum_required(VERSION 3.24)
project(demos LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB projects ./cmu-15-418/*)

print_info("――――――――――――――――――APP SCOPE――――――――――――――――――" "0")

foreach(project ${projects})
  if(IS_DIRECTORY ${project})
    set(EXECUTABLE_OUTPUT_PATH ../outputs/demos)
    math(EXPR index "${index} + 1" OUTPUT_FORMAT DECIMAL)
    get_filename_component(project-name ${project} NAME)
    file(GLOB ${project}-src ${project}/*.cpp ${project}/*.c ${project}/*.cc)
    file(GLOB ${project}-headers ${project}/*.h ${project}/*.hpp)

    add_executable(${project-name} ${${project}-src} ${${project}-headers})

    target_link_libraries(${project-name} PRIVATE hpc)

    print_info("[TRACE] Adding: ${project-name}" "0")

  endif()
endforeach()

print_info("[INFO] CMake configuration completed" "92")
print_info("[INFO] Total ${index} exec files will be generated" "92")
print_info("------------END OF CONFIGURATION-------------" "90")
