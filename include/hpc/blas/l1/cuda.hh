#pragma once

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
#include "../../backends/backends.hh"
#endif

#if defined(ENABLE_CUDA) && defined(__CUDACC__)
namespace hpc::l1 {
namespace details {
template <typename T>
__global__ inline void vadd_cuda(T* __restrict__ dst, const T* __restrict__ src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] += src[i];
    }
}

template <typename T>
__global__ inline void vadd_cuda(T* __restrict__ dst, const T& scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] += scalar;
    }
}

template <typename T>
__global__ inline void vsub_cuda(T* __restrict__ dst, const T* __restrict__ src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] -= src[i];
    }   
}

template <typename T>
__global__ inline void vsub_cuda(T* __restrict__ dst, const T& scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] -= scalar;
    }
}

template <typename T>
__global__ inline void vmul_cuda(T* __restrict__ dst, const T* __restrict__ src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] *= src[i];
    }   
}

template <typename T>
__global__ inline void vmul_cuda(T* __restrict__ dst, const T& scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] *= scalar;
    }
}


template <typename T>
__global__ inline void vdiv_cuda(T* __restrict__ dst, const T* __restrict__ src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] /= src[i];
    }
}

template <typename T>
__global__ inline void vdiv_cuda(T* __restrict__ dst, const T& scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] /= scalar;
    }
}

template <typename T>
__global__ inline void vfill_cuda(T* __restrict__ dst, const T& value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = value;
    }
}

template <typename T>
__global__ inline void axpy_cuda(T* __restrict__ y, const T* __restrict__ x, const T& a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        y[i] += a * x[i];
    }
}

} // namespace details
} // namespace hpc::l1
#endif