#pragma once

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
#include "../../backends/backends.hh"
#endif

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
__global__ void axpy_cuda_kernel(const size_t &n, T *__restrict__ dst,
                                 const T *__restrict__ src, const T &alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += alpha * src[idx];
  }
}

// copy
template <typename T>
__global__ void vcopy_cuda_kernel(const size_t &n, T *__restrict__ dst,
                                  const T *__restrict__ src) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

// Host-side wrapper functions
template <typename T, int BlockSize = 256>
inline void axpy_cuda(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src, const T &alpha) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(axpy_cuda_kernel<T>, grid_size, BlockSize, n, dst, src, alpha);
}

template <typename T, int BlockSize = 256>
inline void vcopy_cuda(const size_t &n, T *__restrict__ dst,
                       const T *__restrict__ src) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vcopy_cuda_kernel<T>, grid_size, BlockSize, n, dst, src);
}

} // namespace details
} // namespace hpc::l1
#endif
