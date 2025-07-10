print_info("―――――――――――――――――TEST SCOPE――――――――――――――――――" "0")

set(EXECUTABLE_OUTPUT_PATH ../outputs/tests)

file(GLOB_RECURSE SOURCE_FILES *.cpp *.c *.cc)

# stastics

make_paths_relative(REL_SOURCE_FILES SOURCE_FILES)
print_info("[DEBUG] Test files: ${REL_SOURCE_FILES}" "93")
