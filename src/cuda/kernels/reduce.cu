#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/cast.cuh"
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

template <const size_t kThreadNum = CNUM_THREADS,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp32_kernel(float *output, const float *input,
                                             size_t N) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * kThreadNum + tid;
  constexpr size_t NUM_WARPS = (kThreadNum + kWarpSize - 1) / kWarpSize;
  __shared__ float warp_sdata[NUM_WARPS];

  float sum = (idx < N) ? input[idx] : 0.0f;
  size_t wid = tid / CWARP_SIZE;
  size_t lid = tid % CWARP_SIZE;

  sum = warpReduceSum_fp32<kWarpSize>(sum);

  if (lid == 0) {
    warp_sdata[wid] = sum;
  }
  __syncthreads();

  sum = (lid < NUM_WARPS) ? warp_sdara[lid] : 0.0f;
  if (wid == 0) {
    sum = warpReduceSum_fp32<kWarpSize>(sum);
  }
  if (tid == 0) {
    atomicAdd(output, sum);
  }
}

template <const size_t kThreadNum = CNUM_THREADS / 4,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp32x4_kernel(float *output, float *input,
                                               size_t N) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * kThreadNum + tid;
  constexpr size_t NUM_WARPS = (kThreadNum + kWarpSize - 1) / kWarpSize;
  __shared__ float warp_sdata[NUM_WARPS];

  float4 in_reg = FLOAT4(input[idx]);

  float sum = (idx < N) ? (in_reg.x + in_reg.y + in_reg.z + in_reg.w) : 0.0f;
  size_t wid = tid / CWARP_SIZE;
  size_t lid = tid % CWARP_SIZE;

  sum = warpReduceSum_fp32<kWarpSize>(sum);

  if (tid == 0) {
    warp_sdata[wid] = sum;
  }
  __syncthreads();

  sum = (lid < NUM_WARPS) ? warp_sdata[lid] : 0.0f;
  if (wid == 0) {
    sum = warpReduceSum_fp32<kWarpSize>(sum);
  }
  if (lid == 0) {
    atomicAdd(output, sum);
  }
}

template <const size_t kThreadNum = CNUM_THREADS,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp16_kernel(half *output, const half *input,
                                             size_t N) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * kThreadNum + tid;
  constexpr size_t NUM_WARPS = (kThreadNum + kWarpSize - 1) / kWarpSize;
  __shared__ half warp_sdata[NUM_WARPS];

  half sum = (idx < N) ? input[idx] : 0.0f;
  size_t wid = tid / CWARP_SIZE;
  size_t lid = tid % CWARP_SIZE;

  sum = warpReduceSum_fp16<kWarpSize>(sum);

  if (lid == 0) {
    warp_sdata[wid] = sum;
  }
  __syncthreads();

  sum = (lid < NUM_WARPS) ? warp_sdara[lid] : 0.0f;
  if (wid == 0) {
    sum = warpReduceSum_fp16<kWarpSize>(sum);
  }
  if (tid == 0) {
    atomicAdd(output, sum);
  }
}

template <const size_t kThreadNum = CNUM_THREADS / 2,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void block_reduce_sum_fp16x2_kernel(half *output, half *input,
                                               size_t N) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * kThreadNum + tid;
  constexpr size_t NUM_WARPS = (kThreadNum + kWarpSize - 1) / kWarpSize;
  __shared__ half warp_sdata[NUM_WARPS];

  half2 in_reg = HALF2(input[idx]);

  half sum = (idx < N) ? (in_reg.x + in_reg.y) : 0.0f;
  size_t wid = tid / CWARP_SIZE;
  size_t lid = tid % CWARP_SIZE;

  sum = warpReduceSum_fp16<kWarpSize>(sum);

  if (tid == 0) {
    warp_sdata[wid] = sum;
  }
  __syncthreads();

  sum = (lid < NUM_WARPS) ? warp_sdata[lid] : 0.0f;
  if (wid == 0) {
    sum = warpReduceSum_fp16<kWarpSize>(sum);
  }
  if (lid == 0) {
    atomicAdd(output, sum);
  }
}

} // namespace hpc::cuda
