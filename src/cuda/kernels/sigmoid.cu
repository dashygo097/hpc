#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/sigmoid.cuh"

namespace hpc::cu {
__global__ void sigmoid_fp32_kernel(float *out, float *in, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float in_reg = in[idx];
    in_reg = fminf(fmaxf(in_reg, MIN_EXP_FP32), MAX_EXP_FP32);
    out[idx] = 1.0f / (1.0f + expf(-in_reg));
  }
}

__global__ void sigmoid_fp16_kernel(half *out, half *in, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    half in_reg = in[idx];
    in_reg = __hmin(__hmax(in_reg, MIN_EXP_FP16), MAX_EXP_FP16);
    out[idx] = __hdiv(__float2half(1.0f),
                      __hadd(__float2half(1.0f), hexp(__hneg(in_reg))));
  }
}

} // namespace hpc::cu
