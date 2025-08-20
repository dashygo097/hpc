cmake_minimum_required(VERSION 3.21)
project(demos LANGUAGES C CXX)

if (HAS_CUDA) 
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB projects ./cuda-practice/*)

print_info("――――――――――――――――――APP SCOPE――――――――――――――――――" "0")

foreach(project ${projects})
  if(IS_DIRECTORY ${project})
    set(EXECUTABLE_OUTPUT_PATH ../outputs/demos)
    math(EXPR index "${index} + 1" OUTPUT_FORMAT DECIMAL)
    get_filename_component(project-name ${project} NAME)

    if (HAS_CUDA)
      file(GLOB_RECURSE ${project}-src ${project}/*.cpp ${project}/*.c ${project}/*.cc ${project}/*.cu)
    else()
      file(GLOB_RECURSE ${project}-src ${project}/*.cpp ${project}/*.c ${project}/*.cc)
    endif()

    file(GLOB_RECURSE ${project}-hdr ${project}/*.h ${project}/*.hpp)
    # file(GLOB_RECURSE ${project}-obj ${project}/*.o)

    add_executable(${project-name} ${${project}-src} ${${project}-hdr})

    target_link_libraries(${project-name} PRIVATE hpc)

    print_info("[TRACE] Adding: ${project-name}" "0")

  endif()
endforeach()

print_info("[INFO] CMake configuration completed" "92")
print_info("[INFO] Total ${index} exec files will be generated" "92")
print_info("------------END OF CONFIGURATION-------------" "90")
