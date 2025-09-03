#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/constants.cuh"
#include "hpc/cuda/kernels/sigmoid.cuh"

namespace hpc::cuda {
__global__ void sigmoid_fp32_kernel(float *output, const float *input,
                                    size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx];
  }
}
__global__ void sigmoid_fp32x2_kernel(float *output, const float *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    float2 in_reg = FLOAT2(input + idx);
    float2 out_reg;
    out_reg.x = in_reg.x;
    out_reg.y = in_reg.y;
    FLOAT2(output + idx) = out_reg;
  }
}
__global__ void sigmoid_fp32x4_kernel(float *output, const float *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 in_reg = FLOAT4(input + idx);
    float4 out_reg;
    out_reg.x = in_reg.x;
    out_reg.y = in_reg.y;
    out_reg.z = in_reg.z;
    out_reg.w = in_reg.w;
    FLOAT4(output + idx) = out_reg;
  }
}
__global__ void sigmoid_fp16_kernel(half *output, const half *input, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx];
  }
}
__global__ void sigmoid_fp16x2_kernel(half *output, const half *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    half2 in_reg = HALF2(input + idx);
    half2 out_reg;
    out_reg.x = in_reg.x;
    out_reg.y = in_reg.y;
    HALF2(output + idx) = out_reg;
  }
}
__global__ void sigmoid_fp16x8_kernel(half *output, const half *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx < N) {
    half2 in_reg_0 = HALF2(input + idx);
    half2 out_reg_0;
    out_reg_0.x = in_reg_0.x;
    out_reg_0.y = in_reg_0.y;
    HALF2(output + idx) = out_reg_0;
  }
  if ((idx + 2) < N) {
    half2 in_reg_1 = HALF2(input + idx + 2);
    half2 out_reg_1;
    out_reg_1.x = in_reg_1.x;
    out_reg_1.y = in_reg_1.y;
    HALF2(output + idx + 2) = out_reg_1;
  }
  if ((idx + 4) < N) {
    half2 in_reg_2 = HALF2(input + idx + 4);
    half2 out_reg_2;
    out_reg_2.x = in_reg_2.x;
    out_reg_2.y = in_reg_2.y;
    HALF2(output + idx + 4) = out_reg_2;
  }
  if ((idx + 6) < N) {
    half2 in_reg_3 = HALF2(input + idx + 6);
    half2 out_reg_3;
    out_reg_3.x = in_reg_3.x;
    out_reg_3.y = in_reg_3.y;
    HALF2(output + idx + 6) = out_reg_3;
  }
}
} // namespace hpc::cuda
