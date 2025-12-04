#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_CUDA
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d\n", __FILE__, __LINE__);            \
      fprintf(stderr, "  %s: %s\n", #call, cudaGetErrorString(err));           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_LAST()                                                      \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Kernel Error at %s:%d\n", __FILE__, __LINE__);     \
      fprintf(stderr, "  %s\n", cudaGetErrorString(err));                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK_LAST();                                                         \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

#define CUDA_SAFE_CALL(call)                                                   \
  hpc::cuda::checkCudaError((call), __FILE__, __LINE__, #call)

#endif

#ifdef ENABLE_CUDA
namespace hpc::cu {

inline void checkCudaError(cudaError_t err, const char *file, int line,
                           const char *msg = nullptr) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error at %s:%d\n", file, line);
    if (msg)
      fprintf(stderr, "  Message: %s\n", msg);
    fprintf(stderr, "  %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

} // namespace hpc::cu

#endif
