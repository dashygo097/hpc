#pragma once

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
#include "../../backends/backends.hh"
#endif

#define DEF_BLAS_L1_CUDA_OP(name, op)                                          \
  template <typename T>                                                        \
  __global__ void name##_cuda_kernel(T *__restrict__ dst,                      \
                                     const T *__restrict__ src, size_t n) {    \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (idx < n) {                                                             \
      dst[idx] op src[idx];                                                    \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  __global__ void name##_cuda_kernel(T *__restrict__ dst, const T &scalar,     \
                                     size_t n) {                               \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (idx < n) {                                                             \
      dst[idx] op scalar;                                                      \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, int BlockSize = 256>                                   \
  inline void name##_cuda(T *__restrict__ dst, const T *__restrict__ src,      \
                          size_t n) {                                          \
    int grid_size = (n + BlockSize - 1) / BlockSize;                           \
    CUDA_LAUNCH(name##_cuda_kernel<T>, grid_size, BlockSize, dst, src, n);     \
  }                                                                            \
                                                                               \
  template <typename T, int BlockSize = 256>                                   \
  inline void name##_cuda(T *__restrict__ dst, const T &scalar, size_t n) {    \
    int grid_size = (n + BlockSize - 1) / BlockSize;                           \
    CUDA_LAUNCH(name##_cuda_kernel<T>, grid_size, BlockSize, dst, scalar, n);  \
  }

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
namespace hpc::l1 {
namespace details {

DEF_BLAS_L1_CUDA_OP(vadd, +=)
DEF_BLAS_L1_CUDA_OP(vsub, -=)
DEF_BLAS_L1_CUDA_OP(vmul, *=)
DEF_BLAS_L1_CUDA_OP(vdiv, /=)

// reduce
template <typename T>
__global__ void vsum_cuda_kernel(T *result, const T *__restrict__ src,
                                 size_t n) {
  // NOTE: Unimpled method
  cudaError_t err = cudaErrorNotSupported;
  printf("Error: vsum_cuda_kernel is not implemented. %s\n",
         cudaGetErrorString(err));
}

// fill
template <typename T>
__global__ void vfill_cuda_kernel(T *__restrict__ dst, const T &scalar,
                                  size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = scalar;
  }
}

// l1

// axpy
template <typename T>
__global__ void axpy_cuda_kernel(T *__restrict__ y, const T *__restrict__ x,
                                 const T &a, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += a * x[idx];
  }
}

// copy
template <typename T>
__global__ void vcopy_cuda_kernel(T *__restrict__ dst,
                                  const T *__restrict__ src, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

// Host-side wrapper functions
template <typename T, int BlockSize = 256>
inline void vsum_cuda(T *result, const T *__restrict__ src, size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vsum_cuda_kernel<T>, grid_size, BlockSize, result, src, n);
}

template <typename T, int BlockSize = 256>
inline void vfill_cuda(T *__restrict__ dst, const T &scalar, size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vfill_cuda_kernel<T>, grid_size, BlockSize, dst, scalar, n);
}

template <typename T, int BlockSize = 256>
inline void axpy_cuda(T *__restrict__ y, const T *__restrict__ x, const T &a,
                      size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(axpy_cuda_kernel<T>, grid_size, BlockSize, y, x, a, n);
}

template <typename T, int BlockSize = 256>
inline void vcopy_cuda(T *__restrict__ dst, const T *__restrict__ src,
                       size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vcopy_cuda_kernel<T>, grid_size, BlockSize, dst, src, n);
}

} // namespace details
} // namespace hpc::l1
#endif
