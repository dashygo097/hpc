#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/relu.cuh"

namespace hpc::cu {
__global__ void relu_fp32_kernel(float *out, float *in, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = fmaxf(0.0f, in[idx]);
  }
}

__global__ void relu_fp16_kernel(half *out, half *in, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = __hmax(__float2half(0.0f), in[idx]);
  }
}

} // namespace hpc::cu
