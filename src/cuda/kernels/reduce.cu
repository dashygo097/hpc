#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/constants.cuh"
#include "hpc/cuda/kernels/reduce.cuh"

namespace hpc::cuda {
template <const size_t kWarpSize = CWARP_SIZE>
__device__ __forceinline__ float warpReduceSum_fp32(float val) {
#pragma unroll
  for (size_t mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const size_t kWarpSize = CWARP_SIZE>
__device__ __forceinline__ half warpReduceSum_fp16(half val) {
#pragma unroll
  for (size_t mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val = __hadd(__shfl_down_sync(0xffffffff, val, mask), val);
  }
  return val;
}

template <const size_t kThreadNum = CTHREAD_NUM>
__global__ void block_reduce_sum_fp32_kernel(float output, const float *input,
                                             size_t N) {}

} // namespace hpc::cuda
