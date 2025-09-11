#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/dot_prod.cuh"

namespace hpc::cu {
template <const size_t kBlockSize>
__global__ void dot_prod_fp32_kernel(float *out, float *in1, float *in2,
                                     size_t N) {
  __shared__ float sdata[kBlockSize / CWARP_SIZE];
  size_t tid = threadIdx.x;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t lane = threadIdx.x % CWARP_SIZE;
  size_t wid = threadIdx.x / CWARP_SIZE;
  size_t mask = 0xffffffff;

  float sum = 0.0f;
  while (idx < N) {
    sum += in1[idx] * in2[idx];
    idx += blockDim.x * gridDim.x;
  }

  for (size_t offset = CWARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  if (lane == 0) {
    sdata[wid] = sum;
  }
  __syncthreads();

  if (wid == 0) {
    sum = (tid < blockDim.x / CWARP_SIZE) ? sdata[lane] : 0.0f;
    for (size_t offset = CWARP_SIZE / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
      atomicAdd(out, sum);
    }
  }
}

template <const size_t kBlockSize>
__global__ void dot_prod_fp16_kernel(half *out, half *in1, half *in2,
                                     size_t N) {
  __shared__ half sdata[kBlockSize / CWARP_SIZE];
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t lane = threadIdx.x % CWARP_SIZE;
  size_t wid = threadIdx.x / CWARP_SIZE;
  size_t mask = 0xffffffff;

  half sum = __float2half(0.0f);
  while (tid < N) {
    sum += __hmul(in1[tid], in2[tid]);
    tid += blockDim.x * gridDim.x;
  }
  for (size_t offset = CWARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  if (lane == 0) {
    sdata[wid] = sum;
  }

  __syncthreads();
  if (wid == 0) {
    sum = (threadIdx.x < blockDim.x / CWARP_SIZE) ? sdata[lane] : 0.0f;
    for (size_t offset = CWARP_SIZE / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
      atomicAdd(out, sum);
    }
  }
}

} // namespace hpc::cu
