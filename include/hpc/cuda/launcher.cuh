#pragma once

#include "./runtime.cuh"

#define CUDA_LAUNCH(kernel, grid, block, ...)                                  \
  do {                                                                         \
    kernel<<<grid, block>>>(__VA_ARGS__);                                      \
    cudaCheckKernel(#kernel);                                                  \
  } while (0)

#define CUDA_LAUNCH_SHARED(kernel, grid, block, shared, ...)                   \
  do {                                                                         \
    kernel<<<grid, block, shared>>>(__VA_ARGS__);                              \
    cudaCheckKernel(#kernel);                                                  \
  } while (0)

#define CUDA_LAUNCH_STREAM(kernel, grid, block, stream, ...)                   \
  do {                                                                         \
    kernel<<<grid, block, 0, stream>>>(__VA_ARGS__);                           \
    cudaCheckErrors(#kernel " launch");                                        \
  } while (0)

#define CUDA_LAUNCH_1D(kernel, numElements, blockSize, ...)                    \
  do {                                                                         \
    dim3 grid((numElements + blockSize - 1) / blockSize);                      \
    dim3 block(blockSize);                                                     \
    kernel<<<grid, block>>>(__VA_ARGS__);                                      \
    cudaCheckKernel(#kernel);                                                  \
  } while (0)

namespace hpc::cu {
class CudaKernelLauncher {
public:
  CudaKernelLauncher() : eventsCreated(false) {
    config = cudaGetDefaultConfig();
  }

  explicit CudaKernelLauncher(const CudaKernelConfig &cfg)
      : config(cfg), eventsCreated(false) {}

  ~CudaKernelLauncher() {
    if (eventsCreated) {
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
    }
  }

  CudaKernelLauncher &grid(dim3 grid) {
    config.gridSize = grid;
    return *this;
  }
  CudaKernelLauncher &block(dim3 block) {
    config.blockSize = block;
    return *this;
  }
  CudaKernelLauncher &sharedMem(size_t bytes) {
    config.sharedMemBytes = bytes;
    return *this;
  }
  CudaKernelLauncher &stream(cudaStream_t s) {
    config.stream = s;
    return *this;
  }
  CudaKernelLauncher &synchronous(bool sync) {
    config.synchronous = sync;
    return *this;
  }
  CudaKernelLauncher &profileTime(bool profile) {
    config.profileTime = profile;
    return *this;
  }
  CudaKernelLauncher &device(int device) {
    config.deviceId = device;
    return *this;
  }
  CudaKernelLauncher &verbose(bool v) {
    config.verbose = v;
    return *this;
  }
  CudaKernelLauncher &kernelName(const char *name) {
    config.kernelName = name;
    return *this;
  }

  CudaKernelLauncher &autoConfig1D(int numElements,
                                   int blockSize = CBLOCK_SIZE_1D) {
    config.blockSize = dim3(blockSize, 1, 1);
    config.gridSize = dim3((numElements + blockSize - 1) / blockSize, 1, 1);
    return *this;
  }

  CudaKernelLauncher &autoConfig2D(int width, int height, int blockX = CBLOCK_SIZE_2D,
                                   int blockY = CBLOCK_SIZE_2D) {
    config.blockSize = dim3(blockX, blockY, 1);
    config.gridSize =
        dim3((width + blockX - 1) / blockX, (height + blockY - 1) / blockY, 1);
    return *this;
  }

  template <typename KernelFunc, typename... Args>
  float launch(KernelFunc kernel, Args... args) {
    float elapsedTime = 0.0f;

    if (config.deviceId >= 0) {
      cudaCheckError(cudaSetDevice(config.deviceId), "Set device");
    }

    if (config.profileTime && !eventsCreated) {
      cudaCheckError(cudaEventCreate(&startEvent), "Create start event");
      cudaCheckError(cudaEventCreate(&stopEvent), "Create stop event");
      eventsCreated = true;
    }

    if (config.verbose) {
      printf("Launching kernel '%s':\n", config.kernelName);
      printf("  Grid: (%d, %d, %d)\n", config.gridSize.x, config.gridSize.y,
             config.gridSize.z);
      printf("  Block: (%d, %d, %d)\n", config.blockSize.x, config.blockSize.y,
             config.blockSize.z);
      printf("  Shared Memory: %zu bytes\n", config.sharedMemBytes);
      printf("  Stream: %p\n", (void *)config.stream);
    }

    if (config.profileTime) {
      cudaCheckError(cudaEventRecord(startEvent, config.stream),
                     "Record start event");
    }

    kernel<<<config.gridSize, config.blockSize, config.sharedMemBytes,
             config.stream>>>(args...);

    cudaCheckErrors("Kernel launch");

    if (config.profileTime) {
      cudaCheckError(cudaEventRecord(stopEvent, config.stream),
                     "Record stop event");
    }

    if (config.synchronous) {
      if (config.stream != 0) {
        cudaCheckError(cudaStreamSynchronize(config.stream),
                       "Stream synchronize");
      } else {
        cudaCheckError(cudaDeviceSynchronize(), "Device synchronize");
      }
    }

    if (config.profileTime) {
      cudaCheckError(cudaEventSynchronize(stopEvent), "Event synchronize");
      cudaCheckError(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent),
                     "Calculate elapsed time");

      if (config.verbose) {
        printf("Kernel '%s' execution time: %.3f ms\n", config.kernelName,
               elapsedTime);
      }
    }

    return elapsedTime;
  }

private:
  CudaKernelConfig config;
  cudaEvent_t startEvent, stopEvent;
  bool eventsCreated;
};

} // namespace hpc::cuda
