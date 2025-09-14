#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/relu.cuh"

namespace hpc::cu {
__global__ void relu_fp32_kernel(float *out, float *in, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = fmaxf(0.0f, in[idx]);
  }
}

__global__ void relu_fp32x2_kernel(float *out, float *in, size_t N) {
  size_t idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float2 in_reg = FLOAT2(in + idx);
    float2 out_reg;
    out_reg.x = fmaxf(0.0f, in_reg.x);
    out_reg.y = fmaxf(0.0f, in_reg.y);
    FLOAT2(out + idx) = out_reg;
  }
}

__global__ void relu_fp32x4_kernel(float *out, float *in, size_t N) {
  size_t idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 in_reg = FLOAT4(in + idx);
    float4 out_reg;
    out_reg.x = fmaxf(0.0f, in_reg.x);
    out_reg.y = fmaxf(0.0f, in_reg.y);
    out_reg.z = fmaxf(0.0f, in_reg.z);
    out_reg.w = fmaxf(0.0f, in_reg.w);
    FLOAT4(out + idx) = out_reg;
  }
}

__global__ void relu_fp16_kernel(half *out, half *in, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = __hmax(__float2half(0.0f), in[idx]);
  }
}

__global__ void relu_fp16x2_kernel(half *out, half *in, size_t N) {
  size_t idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 in_reg = HALF2(in + idx);
    half2 out_reg;
    out_reg.x = __hmax(__float2half(0.0f), in_reg.x);
    out_reg.y = __hmax(__float2half(0.0f), in_reg.y);
    HALF2(out + idx) = out_reg;
  }
}

__global__ void relu_fp16x8_kernel(half *out, half *in, size_t N) {
  size_t idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 in_reg_0 = HALF2(in + idx);
    half2 out_reg_0;
    out_reg_0.x = __hmax(__float2half(0.0f), in_reg_0.x);
    out_reg_0.y = __hmax(__float2half(0.0f), in_reg_0.y);
    HALF2(out + idx) = out_reg_0;
  }
  if ((idx + 2) < N) {
    half2 in_reg_1 = HALF2(in + idx + 2);
    half2 out_reg_1;
    out_reg_1.x = __hmax(__float2half(0.0f), in_reg_1.x);
    out_reg_1.y = __hmax(__float2half(0.0f), in_reg_1.y);
    HALF2(out + idx + 2) = out_reg_1;
  }
  if ((idx + 4) < N) {
    half2 in_reg_2 = HALF2(in + idx + 4);
    half2 out_reg_2;
    out_reg_2.x = __hmax(__float2half(0.0f), in_reg_2.x);
    out_reg_2.y = __hmax(__float2half(0.0f), in_reg_2.y);
    HALF2(out + idx + 4) = out_reg_2;
  }
  if ((idx + 6) < N) {
    half2 in_reg_3 = HALF2(in + idx + 6);
    half2 out_reg_3;
    out_reg_3.x = __hmax(__float2half(0.0f), in_reg_3.x);
    out_reg_3.y = __hmax(__float2half(0.0f), in_reg_3.y);
    HALF2(out + idx + 6) = out_reg_3;
  }
}

} // namespace hpc::cu
