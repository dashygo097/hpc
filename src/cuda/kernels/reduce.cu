#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/reduce.cuh"

namespace hpc::cuda {

template <const size_t kBlockSize>
__global__ void reduce_sum_fp32_kernel(float *output, const float *input,
                                       size_t N) {
  __shared__ float sdata[kBlockSize / CWARP_SIZE];
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t lane = tid % CWARP_SIZE;
  size_t wid = tid / CWARP_SIZE;
  size_t mask = 0xffffffff;

  float sum = 0.0f;
  while (idx < N) {
    sum += input[idx];
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
    if (tid == 0) {
      atomicAdd(output, sum);
    }
  }
}

template <const size_t kBlockSize>
__global__ void reduce_sum_fp16_kernel(half *output, const half *input,
                                       size_t N) {
  __shared__ half sdata[kBlockSize / CWARP_SIZE];
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t lane = tid % CWARP_SIZE;
  size_t wid = tid / CWARP_SIZE;
  size_t mask = 0xffffffff;
  half sum = __float2half(0.0f);

  while (idx < N) {
    sum += input[idx];
    idx += blockDim.x * gridDim.x;
  };

  for (size_t offset = CWARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  if (lane == 0) {
    sdata[wid] = sum;
  }
  __syncthreads();

  if (wid == 0) {
    sum = (tid < blockDim.x / CWARP_SIZE) ? sdata[lane] : __float2half(0.0f);
    for (size_t offset = CWARP_SIZE / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(mask, sum, offset);
    }
    if (tid == 0) {
      atomicAdd(output, sum);
    }
  }
}

template <const size_t kBlockSize>
torch::Tensor reduce_sum_fp32(torch::Tensor input) {
  return reduce_wrapper<float, 1, kBlockSize>(
      input, torch::kFloat32, reduce_sum_fp32_kernel<kBlockSize>);
}
template <const size_t kBlockSize>
torch::Tensor reduce_sum_fp16(torch::Tensor input) {
  return reduce_wrapper<half, 1, kBlockSize>(
      input, torch::kHalf, reduce_sum_fp16_kernel<kBlockSize>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum_fp32", &reduce_sum_fp32<CBLOCK_SIZE_1D>, "reduce_sum_fp32");
  m.def("reduce_sum_fp16", &reduce_sum_fp16<CBLOCK_SIZE_1D>, "reduce_sum_fp16");
}

} // namespace hpc::cuda
