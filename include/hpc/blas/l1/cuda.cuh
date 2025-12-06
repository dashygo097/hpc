#pragma once

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
#include "../../backends/backends.hh"
#endif

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
namespace hpc::l1 {
namespace details {

// CUDA Kernels
template <typename T>
__global__ void vadd_cuda_kernel(T *__restrict__ dst, const T *__restrict__ src,
                                 size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += src[idx];
  }
}

template <typename T>
__global__ void vadd_scalar_cuda_kernel(T *__restrict__ dst, const T &scalar,
                                        size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += scalar;
  }
}

template <typename T>
__global__ void vsub_cuda_kernel(T *__restrict__ dst, const T *__restrict__ src,
                                 size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] -= src[idx];
  }
}

template <typename T>
__global__ void vsub_scalar_cuda_kernel(T *__restrict__ dst, const T &scalar,
                                        size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] -= scalar;
  }
}

template <typename T>
__global__ void vmul_cuda_kernel(T *__restrict__ dst, const T *__restrict__ src,
                                 size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] *= src[idx];
  }
}

template <typename T>
__global__ void vmul_scalar_cuda_kernel(T *__restrict__ dst, const T &scalar,
                                        size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] *= scalar;
  }
}

template <typename T>
__global__ void vdiv_cuda_kernel(T *__restrict__ dst, const T *__restrict__ src,
                                 size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] /= src[idx];
  }
}

template <typename T>
__global__ void vdiv_scalar_cuda_kernel(T *__restrict__ dst, const T &scalar,
                                        size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] /= scalar;
  }
}

template <typename T>
__global__ void vfill_cuda_kernel(T *__restrict__ dst, const T &scalar,
                                  size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = scalar;
  }
}

template <typename T>
__global__ void vcopy_cuda_kernel(T *__restrict__ dst,
                                  const T *__restrict__ src, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

template <typename T>
__global__ void axpy_cuda_kernel(T *__restrict__ y, const T *__restrict__ x,
                                 const T &a, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += a * x[idx];
  }
}

// Host-side wrapper functions
template <typename T, int BlockSize = 256>
inline void vadd_cuda(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vadd_cuda_kernel<T>, grid_size, BlockSize, dst, src, n);
}

template <typename T, int BlockSize = 256>
inline void vadd_cuda(T *__restrict__ dst, const T &scalar, size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vadd_scalar_cuda_kernel<T>, grid_size, BlockSize, dst, scalar, n);
}

template <typename T, int BlockSize = 256>
inline void vsub_cuda(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vsub_cuda_kernel<T>, grid_size, BlockSize, dst, src, n);
}

template <typename T, int BlockSize = 256>
inline void vsub_cuda(T *__restrict__ dst, const T &scalar, size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vsub_scalar_cuda_kernel<T>, grid_size, BlockSize, dst, scalar, n);
}

template <typename T, int BlockSize = 256>
inline void vmul_cuda(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vmul_cuda_kernel<T>, grid_size, BlockSize, dst, src, n);
}

template <typename T, int BlockSize = 256>
inline void vmul_cuda(T *__restrict__ dst, const T &scalar, size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vmul_scalar_cuda_kernel<T>, grid_size, BlockSize, dst, scalar, n);
}

template <typename T, int BlockSize = 256>
inline void vdiv_cuda(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vdiv_cuda_kernel<T>, grid_size, BlockSize, dst, src, n);
}

template <typename T, int BlockSize = 256>
inline void vdiv_cuda(T *__restrict__ dst, const T &scalar, size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vdiv_scalar_cuda_kernel<T>, grid_size, BlockSize, dst, scalar, n);
}

template <typename T, int BlockSize = 256>
inline void vfill_cuda(T *__restrict__ dst, const T &scalar, size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vfill_cuda_kernel<T>, grid_size, BlockSize, dst, scalar, n);
}

template <typename T, int BlockSize = 256>
inline void vcopy_cuda(T *__restrict__ dst, const T *__restrict__ src,
                       size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(vcopy_cuda_kernel<T>, grid_size, BlockSize, dst, src, n);
}

template <typename T, int BlockSize = 256>
inline void axpy_cuda(T *__restrict__ y, const T *__restrict__ x, const T &a,
                      size_t n) {
  int grid_size = (n + BlockSize - 1) / BlockSize;
  CUDA_LAUNCH(axpy_cuda_kernel<T>, grid_size, BlockSize, y, x, a, n);
}
} // namespace details
} // namespace hpc::l1
#endif
