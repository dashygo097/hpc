#pragma once

#include "../constants.hh"

#define cudaCheckLast(msg)                                                     \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define cudaCheckError(call, msg)                                              \
  do {                                                                         \
    cudaError_t __err = call;                                                  \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error: %s (%s at %s:%d)\n", msg,                   \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define cudaCheckKernel(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Kernel launch error: %s (%s at %s:%d)\n", msg,          \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      exit(1);                                                                 \
    }                                                                          \
    __err = cudaDeviceSynchronize();                                           \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Kernel execution error: %s (%s at %s:%d)\n", msg,       \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace hpc::cu {
typedef struct {
  dim3 gridSize;
  dim3 blockSize;

  size_t sharedMemBytes;

  cudaStream_t stream;

  bool synchronous;
  bool profileTime;

  int deviceId;

  bool verbose;
  const char *kernelName;
} CudaKernelConfig;

static inline CudaKernelConfig cudaGetDefaultConfig() {
  CudaKernelConfig config;
  config.gridSize = dim3(1, 1, 1);
  config.blockSize = dim3(hpc::CBLOCK_SIZE_1D, 1, 1);
  config.sharedMemBytes = 0;
  config.stream = 0;
  config.synchronous = true;
  config.profileTime = false;
  config.deviceId = 0;
  config.verbose = false;
  config.kernelName = "unknown_kernel";
  return config;
}

static inline CudaKernelConfig
cudaAutoConfig1D(int numElements, int blockSize = CBLOCK_SIZE_1D) {
  CudaKernelConfig config = cudaGetDefaultConfig();
  config.blockSize = dim3(blockSize, 1, 1);
  config.gridSize = dim3((numElements + blockSize - 1) / blockSize, 1, 1);
  return config;
}

static inline CudaKernelConfig cudaAutoConfig2D(int width, int height,
                                                int blockX = CBLOCK_SIZE_2D,
                                                int blockY = CBLOCK_SIZE_2D) {
  CudaKernelConfig config = cudaGetDefaultConfig();
  config.blockSize = dim3(blockX, blockY, 1);
  config.gridSize =
      dim3((width + blockX - 1) / blockX, (height + blockY - 1) / blockY, 1);
  return config;
}

static inline void cudaPrintDeviceInfo(int deviceId = 0) {
  cudaDeviceProp prop;
  cudaCheckError(cudaGetDeviceProperties(&prop, deviceId),
                 "Get device properties");

  printf("=== CUDA Device %d Info ===\n", deviceId);
  printf("Name: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Total Global Memory: %.2f GB\n",
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Warp Size: %d\n", prop.warpSize);
  printf("Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
  printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
  printf("===========================\n");
}

static inline int cudaGetOptimalBlockSize(const void *func) {
  int minGridSize, blockSize;
  cudaCheckError(
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, 0),
      "Calculate optimal block size");
  return blockSize;
}

template <typename T> static inline T *cudaMallocM(size_t count) {
  T *ptr;
  cudaCheckError(cudaMallocManaged(&ptr, count * sizeof(T)),
                 "Allocate managed memory");
  return ptr;
}

template <typename T> static inline T *cudaMallocD(size_t count) {
  T *ptr;
  cudaCheckError(cudaMalloc(&ptr, count * sizeof(T)), "Allocate device memory");
  return ptr;
}

template <typename T> static inline T *cudaMallocH(size_t count) {
  T *ptr;
  cudaCheckError(cudaMallocHost(&ptr, count * sizeof(T)),
                 "Allocate pinned host memory");
  return ptr;
}
} // namespace hpc::cu
