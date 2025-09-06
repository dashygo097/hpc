#pragma once
#include "hpc/constants.hh"
#include <cuda_fp16.h>

namespace hpc::cuda {
template <const size_t kWarpSize = CWARP_SIZE>
__device__ __forceinline__ float warpReduceSum_fp32(float val);
template <const size_t kWarpSize = CWARP_SIZE>
__device__ __forceinline__ half warpReduceSum_fp16(half val);
template <const size_t kThreadNum = CNUM_THREADS,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp32_kernel(float output, const float *input,
                                             size_t N);
template <const size_t kThreadNum = CNUM_THREADS / 4,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp32x4_kernel(float output, const float *input,
                                               size_t N);
template <const size_t kThreadNum = CNUM_THREADS,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp16_kernel(half output, const half *input,
                                             size_t N);
template <const size_t kThreadNum = CNUM_THREADS / 2,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp16x2_kernel(half output, const half *input,
                                               size_t N);
} // namespace hpc::cuda
