#pragma once
#include <cuda_fp16.h>

#include "hpc/constants.hh"

namespace hpc::cuda {
template <const size_t kWarpSize = CWARP_SIZE>
__device__ __forceinline__ float warpReduceSum_fp32(float val);
template <const size_t kWarpSize = CWARP_SIZE>
__device__ __forceinline__ half warpReduceSum_fp16(half val);
template <const size_t kThreadNum = CTHREAD_NUM>
__global__ void block_reduce_sum_fp32_kernel(float output, const float *input,
                                             size_t N);
} // namespace hpc::cuda
