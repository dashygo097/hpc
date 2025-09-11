#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/constants.cuh"
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
__global__ void sigmoid_fp32x2_kernel(float *out, float *in, size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    float2 in_reg = FLOAT2(in + idx);
    in_reg.x = fminf(fmaxf(in_reg.x, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.y = fminf(fmaxf(in_reg.y, MIN_EXP_FP32), MAX_EXP_FP32);
    float2 out_reg;
    out_reg.x = 1.0f / (1.0f + expf(-in_reg.x));
    out_reg.y = 1.0f / (1.0f + expf(-in_reg.y));
    FLOAT2(out + idx) = out_reg;
  }
}
__global__ void sigmoid_fp32x4_kernel(float *out, float *in, size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 in_reg = FLOAT4(in + idx);
    in_reg.x = fminf(fmaxf(in_reg.x, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.y = fminf(fmaxf(in_reg.y, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.z = fminf(fmaxf(in_reg.z, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.w = fminf(fmaxf(in_reg.w, MIN_EXP_FP32), MAX_EXP_FP32);
    float4 out_reg;
    out_reg.x = 1.0f / (1.0f + expf(-in_reg.x));
    out_reg.y = 1.0f / (1.0f + expf(-in_reg.y));
    out_reg.z = 1.0f / (1.0f + expf(-in_reg.z));
    out_reg.w = 1.0f / (1.0f + expf(-in_reg.w));
    FLOAT4(out + idx) = out_reg;
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
__global__ void sigmoid_fp16x2_kernel(half *out, half *in, size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    half2 in_reg = HALF2(in + idx);
    in_reg.x = __hmin(__hmax(in_reg.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg.y = __hmin(__hmax(in_reg.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg;
    out_reg.x = __hdiv(__float2half(1.0f),
                       __hadd(__float2half(1.0f), hexp(__hneg(in_reg.x))));
    out_reg.y = __hdiv(__float2half(1.0f),
                       __hadd(__float2half(1.0f), hexp(__hneg(in_reg.y))));
    HALF2(out + idx) = out_reg;
  }
}
__global__ void sigmoid_fp16x8_kernel(half *out, half *in, size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx < N) {
    half2 in_reg_0 = HALF2(in + idx);
    in_reg_0.x = __hmin(__hmax(in_reg_0.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_0.y = __hmin(__hmax(in_reg_0.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_0;
    out_reg_0.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_0.x))));
    out_reg_0.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_0.y))));
    HALF2(out + idx) = out_reg_0;
  }
  if ((idx + 2) < N) {
    half2 in_reg_1 = HALF2(in + idx + 2);
    in_reg_1.x = __hmin(__hmax(in_reg_1.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_1.y = __hmin(__hmax(in_reg_1.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_1;
    out_reg_1.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_1.x))));
    out_reg_1.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_1.y))));
    HALF2(out + idx + 2) = out_reg_1;
  }
  if ((idx + 4) < N) {
    half2 in_reg_2 = HALF2(in + idx + 4);
    in_reg_2.x = __hmin(__hmax(in_reg_2.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_2.y = __hmin(__hmax(in_reg_2.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_2;
    out_reg_2.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_2.x))));
    out_reg_2.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_2.y))));
    HALF2(out + idx + 4) = out_reg_2;
  }
  if ((idx + 6) < N) {
    half2 in_reg_3 = HALF2(in + idx + 6);
    in_reg_3.x = __hmin(__hmax(in_reg_3.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_3.y = __hmin(__hmax(in_reg_3.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_3;
    out_reg_3.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_3.x))));
    out_reg_3.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_3.y))));
    HALF2(out + idx + 6) = out_reg_3;
  }
}

} // namespace hpc::cu
