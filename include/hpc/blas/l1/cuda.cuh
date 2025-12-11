#pragma once

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
#include "../../backends/backends.hh"
#endif

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
#define ENABLE_CUDA_VECTOR_SCALAR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, src, scalar);            \
  }
#define ENABLE_CUDA_BINARY_BRANCH(name)                                        \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, src1, src2);             \
  }
#define ENABLE_CUDA_UNARY_BRANCH(name)                                         \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, src);                    \
  }
#define ENABLE_CUDA_SCALAR_BRANCH(name)                                        \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, scalar);                 \
  }
#define ENABLE_CUDA_REDUCE_BRANCH(name)                                        \
  else if constexpr (backend == Backend::CUDA) {                               \
    return details::name##_cuda<T, BackendParams...>(n, src);                  \
  }
#define ENABLE_CUDA_REDUCE2_BRANCH(name)                                       \
  else if constexpr (backend == Backend::CUDA) {                               \
    return details::name##_cuda<T, BackendParams...>(n, src1, src2);           \
  }
#else
#define ENABLE_CUDA_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_CUDA_BINARY_BRANCH(name)
#define ENABLE_CUDA_UNARY_BRANCH(name)
#define ENABLE_CUDA_SCALAR_BRANCH(name)
#define ENABLE_CUDA_REDUCE_BRANCH(name)
#define ENABLE_CUDA_REDUCE2_BRANCH(name)
#endif

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
__global__ void axpy_cuda_kernel(size_t n, T *__restrict__ dst,
                                 const T *__restrict__ src, T alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += alpha * src[idx];
  }
}

// copy
template <typename T>
__global__ void copy_cuda_kernel(size_t n, T *__restrict__ dst,
                                 const T *__restrict__ src) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

// scal
template <typename T>
__global__ void scal_cuda_kernel(size_t n, T *__restrict__ dst, T alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] *= alpha;
  }
}

// dot
template <typename T>
__global__ T dot_cuda_kernel(size_t n, const T *__restrict__ src1,
                             const T *__restrict__ src2) {}

// swap
template <typename T>
__global__ void swap_cuda_kernel(size_t n, T *__restrict__ src1,
                                 T *__restrict__ src2) {}

// asum
template <typename T>
__global__ T asum_cuda_kernel(size_t n, const T *__restrict__ src) {}

// nrm2
template <typename T>
__global__ T nrm2_cuda_kernel(size_t n, const T *__restrict__ src) {}

// iamax
template <typename T>
__global__ size_t iamax_cuda_kernel(size_t n, const T *__restrict__ src) {}

// Host-side wrapper functions
template <typename T, const size_t BlockSize>
inline void axpy_cuda(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src, const T &alpha) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(axpy_cuda_kernel<T>, grid_size, BlockSize, n, dst, src, alpha);
}

template <typename T, const size_t BlockSize>
inline void copy_cuda(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(copy_cuda_kernel<T>, grid_size, BlockSize, n, dst, src);
}

template <typename T, const size_t BlockSize>
inline void scal_cuda(const size_t &n, T *__restrict__ dst, const T &alpha) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(scal_cuda_kernel<T>, grid_size, BlockSize, n, dst, alpha);
}

template <typename T, const size_t BlockSize>
inline T dot_cuda(const size_t &n, const T *__restrict__ src1,
                     const T *__restrict__ src2, T &result) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(dot_cuda_kernel<T>, grid_size, BlockSize, n, src1, src2);
}

template <typename T, const size_t BlockSize>
inline void swap_cuda(const size_t &n, T *__restrict__ src1,
                      T *__restrict__ src2) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(swap_cuda_kernel<T>, grid_size, BlockSize, n, src1, src2);
}

template <typename T, const size_t BlockSize>
inline void asum_cuda(const size_t &n, const T *__restrict__ src, T &result) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(asum_cuda_kernel<T>, grid_size, BlockSize, n, src);
}

template <typename T, const size_t BlockSize>
inline void nrm2_cuda(const size_t &n, const T *__restrict__ src, T &result) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(nrm2_cuda_kernel<T>, grid_size, BlockSize, n, src);
}

template <typename T, const size_t BlockSize>
inline void iamax_cuda(const size_t &n, const T *__restrict__ src,
                       size_t &result) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(iamax_cuda_kernel<T>, grid_size, BlockSize, n, src);
}

} // namespace details
} // namespace hpc::l1
#endif
