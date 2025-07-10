print_info("―――――――――――――――――TEST SCOPE――――――――――――――――――" "0")

set(EXECUTABLE_OUTPUT_PATH ../outputs/tests)

file(GLOB_RECURSE TEST_FILES *.cpp *.c *.cc)

add_executable(test_runner ${TEST_FILES})
target_link_libraries(test_runner
    PRIVATE
        gtest_main
        gtest
        hpc
)
gtest_discover_tests(test_runner
  EXTRA_ARGS --gtest_color=yes
  DISCOVERY_TIMEOUT 30
)

# stastics

make_paths_relative(REL_TEST_FILES TEST_FILES)
make_preview_string(REL_TEST_FILES 3)
print_info("[DEBUG] Test files: ${REL_TEST_FILES}" "93")
