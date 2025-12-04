#pragma once

#ifdef ENABLE_CUDA
#include "./config.cuh"
#include "./err. cuh"
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_CUDA
namespace hpc::cuda {

// Simple Launch Macros
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

class KernelLauncher {
public:
  KernelLauncher() = default;

  // Configure grid
  KernelLauncher &grid(dim3 g) {
    _grid = g;
    return *this;
  }
  KernelLauncher &grid(int x, int y = 1, int z = 1) {
    _grid = dim3(x, y, z);
    return *this;
  }

  // Configure block
  KernelLauncher &block(dim3 b) {
    _block = b;
    return *this;
  }
  KernelLauncher &block(int x, int y = 1, int z = 1) {
    _block = dim3(x, y, z);
    return *this;
  }

  // Configure shared memory
  KernelLauncher &shared(size_t bytes) {
    _shared_mem = bytes;
    return *this;
  }

  // Configure stream
  KernelLauncher &stream(cudaStream_t s) {
    _stream = s;
    return *this;
  }

  // Auto-configure for 1D kernel
  KernelLauncher &config1D(int n, int _blocksize = 256) {
    _block = dim3(block_size);
    _grid = dim3((n + block_size - 1) / block_size);
    return *this;
  }

  // Auto-configure for 2D kernel
  KernelLauncher &config2D(int width, int height, int block_x = 16,
                           int block_y = 16) {
    _block = dim3(block_x, block_y);
    _grid =
        dim3((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);
    return *this;
  }

  // Launch kernel
  template <typename Kernel, typename... Args>
  void operator()(Kernel kernel, Args... args) {
    kernel<<<_grid, _block, _shared_mem, stream_>>>(args...);
    CUDA_CHECK_LAST();
  }

private:
  dim3 _grid{1, 1, 1};
  dim3 _block{256, 1, 1};
  size_t _shared_mem = 0;
  cudaStream_t _stream = 0;
};

template <typename Kernel, typename... Args>
void launch1D(Kernel kernel, int n, int _blocksize, Args... args) {
  dim3 grid((n + block_size - 1) / block_size);
  dim3 block(block_size);
  kernel<<<grid, block>>>(args...);
  CUDA_CHECK_KERNEL();
}

template <typename Kernel, typename... Args>
void launch2D(Kernel kernel, int width, int height, int block_x, int block_y,
              Args... args) {
  dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);
  dim3 block(block_x, block_y);
  kernel<<<grid, block>>>(args...);
  CUDA_CHECK_KERNEL();
}

} // namespace hpc::cuda
#endif
