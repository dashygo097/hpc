#pragma once

#ifdef ENABLE_CUDA
#include "./err.cuh"
#endif

#ifdef ENABLE_CUDA
namespace hpc::cu {

inline void init(int device_id = 0, bool verbose = false) {
  CUDA_CHECK(cudaSetDevice(device_id));
  if (verbose) {
    printDeviceInfo(device_id);
  }
}

inline void sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

inline int getDeviceCount() {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

inline void printDeviceInfo(int device_id = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

  printf("╔════════════════════════════════════════╗\n");
  printf("║  CUDA Device %d Information          ║\n", device_id);
  printf("╠════════════════════════════════════════╣\n");
  printf("║ Name: %-32s ║\n", prop.name);
  printf("║ Compute Capability: %d.%-16d ║\n", prop.major, prop.minor);
  printf("║ Global Memory: %.2f GB%-15s ║\n", prop.totalGlobalMem / 1e9, "");
  printf("║ Shared Memory/Block: %zu KB%-10s ║\n",
         prop.sharedMemPerBlock / 1024, "");
  printf("║ Max Threads/Block: %-18d ║\n", prop.maxThreadsPerBlock);
  printf("║ Warp Size: %-26d ║\n", prop.warpSize);
  printf("║ SMs: %-32d ║\n", prop.multiProcessorCount);
  printf("╚════════════════════════════════════════╝\n");
}

template <typename Kernel> int getOptimalBlockSize(Kernel kernel) {
  int min_grid_size, block_size;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                kernel, 0, 0));
  return block_size;
}

inline int getComputeCapability(int device_id = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  return prop.major * 10 + prop.minor;
}

} // namespace hpc::cu
#endif
