#include "hpc/cuda/kernels/relu/relu.cuh"

namespace hpc::cuda {
#ifdef __CUDACC__
__global__ void relu_fp32(float *output, const float *input, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void relu_fp32x2(float *output, const float *input, size_t N) {
  size_t idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float2 in_reg = FLOAT2(input + idx);
    float2 out_reg;
    out_reg.x = fmaxf(0.0f, in_reg.x);
    out_reg.y = fmaxf(0.0f, in_reg.y);
    FLOAT2(output + idx) = out_reg;
  }
}

__global__ void relu_fp32x4(float *output, const float *input, size_t N) {
  size_t idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 in_reg = FLOAT4(input + idx);
    float4 out_reg;
    out_reg.x = fmaxf(0.0f, in_reg.x);
    out_reg.y = fmaxf(0.0f, in_reg.y);
    out_reg.z = fmaxf(0.0f, in_reg.z);
    out_reg.w = fmaxf(0.0f, in_reg.w);
    FLOAT4(output + idx) = out_reg;
  }
}

__global__ void relu_fp16(half *output, const half *input, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = __hmax(__float2half(0.0f), input[idx]);
  }
}

__global__ void relu_fp16x2(half *output, const half *input, size_t N) {
  size_t idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 in_reg = HALF2(input + idx);
    half2 out_reg;
    out_reg.x = __hmax(__float2half(0.0f), in_reg.x);
    out_reg.y = __hmax(__float2half(0.0f), in_reg.y);
    HALF2(output + idx) = out_reg;
  }
}

__global__ void relu_fp16x8(half *output, const half *input, size_t N) {
  size_t idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 in_reg_0 = HALF2(input + idx);
    half2 out_reg_0;
    out_reg_0.x = __hmax(__float2half(0.0f), in_reg_0.x);
    out_reg_0.y = __hmax(__float2half(0.0f), in_reg_0.y);
    HALF2(output + idx) = out_reg_0;
  }
  if ((idx + 2) < N) {
    half2 in_reg_1 = HALF2(input + idx + 2);
    half2 out_reg_1;
    out_reg_1.x = __hmax(__float2half(0.0f), in_reg_1.x);
    out_reg_1.y = __hmax(__float2half(0.0f), in_reg_1.y);
    HALF2(output + idx + 2) = out_reg_1;
  }
  if ((idx + 4) < N) {
    half2 in_reg_2 = HALF2(input + idx + 4);
    half2 out_reg_2;
    out_reg_2.x = __hmax(__float2half(0.0f), in_reg_2.x);
    out_reg_2.y = __hmax(__float2half(0.0f), in_reg_2.y);
    HALF2(output + idx + 4) = out_reg_2;
  }
  if ((idx + 6) < N) {
    half2 in_reg_3 = HALF2(input + idx + 6);
    half2 out_reg_3;
    out_reg_3.x = __hmax(__float2half(0.0f), in_reg_3.x);
    out_reg_3.y = __hmax(__float2half(0.0f), in_reg_3.y);
    HALF2(output + idx + 6) = out_reg_3;
  }
}
#endif
} // namespace hpc::cuda
