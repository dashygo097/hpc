print_info("―――――――――――――――――TEST SCOPE――――――――――――――――――" "0")

set(EXECUTABLE_OUTPUT_PATH ../outputs/tests)

file(GLOB_RECURSE SANITY_TEST_FILES *.cpp *.c *.cc)
file(GLOB_RECURSE TEST_OPENMP_FILES OpenMP_*.cc OpenMP_*.cpp OpenMP_*.c)
file(GLOB_RECURSE TEST_MPI_FILES MPI_*.cc MPI_*.cpp MPI_*.c)
file(GLOB_RECURSE TEST_CUDA_FILES CUDA_*.cc CUDA_*.cpp CUDA_*.c)

list(REMOVE_ITEM SANITY_TEST_FILES ${TEST_OPENMP_FILES} ${TEST_MPI_FILES} ${TEST_CUDA_FILES})

if (SANITY_TEST_FILES)
    add_executable(test_sanity ${SANITY_TEST_FILES})
    target_link_libraries(test_sanity
        PRIVATE
            gtest_main
            gtest
            hpc
    )
    gtest_discover_tests(test_sanity
        EXTRA_ARGS --gtest_color=yes
        DISCOVERY_TIMEOUT 30
    )
endif()
if (TEST_OPENMP_FILES)
    add_executable(test_openmp ${TEST_OPENMP_FILES})
    target_link_libraries(test_openmp
        PRIVATE
            gtest_main
            gtest
            hpc
    )
    gtest_discover_tests(test_openmp
        EXTRA_ARGS --gtest_color=yes
        DISCOVERY_TIMEOUT 30
    )
endif()
if (TEST_MPI_FILES)
    add_executable(test_mpi ${TEST_MPI_FILES})
    target_link_libraries(test_mpi
        PRIVATE
            gtest_main
            gtest
            hpc
            MPI::MPI_CXX
    )
    gtest_discover_tests(test_mpi
        EXTRA_ARGS --gtest_color=yes
        DISCOVERY_TIMEOUT 30
    )
endif()
if (TEST_CUDA_FILES)
    add_executable(test_cuda ${TEST_CUDA_FILES})
    target_link_libraries(test_cuda
        PRIVATE
            gtest_main
            gtest
            hpc
            CUDA::cudart
    )
    gtest_discover_tests(test_cuda
        EXTRA_ARGS --gtest_color=yes
        DISCOVERY_TIMEOUT 30
    )
endif()

# stastics

make_paths_relative(REL_TEST_FILES TEST_FILES)
make_preview_string(REL_TEST_FILES 3)
print_info("[DEBUG] Test files: ${PREVIOUS_SCOPE_VAR}" "93")
