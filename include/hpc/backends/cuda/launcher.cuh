#pragma once

#ifdef ENABLE_CUDA
#include "./err.cuh"
#include "./macros.cuh"
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_CUDA
namespace hpc::cu {

// Launch Macros
#define CUDA_LAUNCH(kernel, grid, block, ...)                                  \
  do {                                                                         \
    kernel<<<grid, block>>>(__VA_ARGS__);                                      \
    CUDA_CHECK_KERNEL();                                                       \
  } while (0)

#define CUDA_LAUNCH_ASYNC(kernel, grid, block, stream, ...)                    \
  do {                                                                         \
    kernel<<<grid, block, 0, stream>>>(__VA_ARGS__);                           \
    CUDA_CHECK_LAST();                                                         \
  } while (0)

} // namespace hpc::cuda
#endif
