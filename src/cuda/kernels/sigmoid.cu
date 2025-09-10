#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/constants.cuh"
#include "hpc/cuda/kernels/sigmoid.cuh"

namespace hpc::cu {
__global__ void sigmoid_fp32_kernel(float *output, const float *input,
                                    size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float in_reg = input[idx];
    in_reg = fminf(fmaxf(in_reg, MIN_EXP_FP32), MAX_EXP_FP32);
    output[idx] = 1.0f / (1.0f + expf(-in_reg));
  }
}
__global__ void sigmoid_fp32x2_kernel(float *output, const float *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    float2 in_reg = FLOAT2(input + idx);
    in_reg.x = fminf(fmaxf(in_reg.x, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.y = fminf(fmaxf(in_reg.y, MIN_EXP_FP32), MAX_EXP_FP32);
    float2 out_reg;
    out_reg.x = 1.0f / (1.0f + expf(-in_reg.x));
    out_reg.y = 1.0f / (1.0f + expf(-in_reg.y));
    FLOAT2(output + idx) = out_reg;
  }
}
__global__ void sigmoid_fp32x4_kernel(float *output, const float *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 in_reg = FLOAT4(input + idx);
    in_reg.x = fminf(fmaxf(in_reg.x, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.y = fminf(fmaxf(in_reg.y, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.z = fminf(fmaxf(in_reg.z, MIN_EXP_FP32), MAX_EXP_FP32);
    in_reg.w = fminf(fmaxf(in_reg.w, MIN_EXP_FP32), MAX_EXP_FP32);
    float4 out_reg;
    out_reg.x = 1.0f / (1.0f + expf(-in_reg.x));
    out_reg.y = 1.0f / (1.0f + expf(-in_reg.y));
    out_reg.z = 1.0f / (1.0f + expf(-in_reg.z));
    out_reg.w = 1.0f / (1.0f + expf(-in_reg.w));
    FLOAT4(output + idx) = out_reg;
  }
}
__global__ void sigmoid_fp16_kernel(half *output, const half *input, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    half in_reg = input[idx];
    in_reg = __hmin(__hmax(in_reg, MIN_EXP_FP16), MAX_EXP_FP16);
    output[idx] = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg))));
  }
}
__global__ void sigmoid_fp16x2_kernel(half *output, const half *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    half2 in_reg = HALF2(input + idx);
    in_reg.x = __hmin(__hmax(in_reg.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg.y = __hmin(__hmax(in_reg.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg;
    out_reg.x = __hdiv(__float2half(1.0f),
                       __hadd(__float2half(1.0f), hexp(__hneg(in_reg.x))));
    out_reg.y = __hdiv(__float2half(1.0f),
                       __hadd(__float2half(1.0f), hexp(__hneg(in_reg.y))));
    HALF2(output + idx) = out_reg;
  }
}
__global__ void sigmoid_fp16x8_kernel(half *output, const half *input,
                                      size_t N) {
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx < N) {
    half2 in_reg_0 = HALF2(input + idx);
    in_reg_0.x = __hmin(__hmax(in_reg_0.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_0.y = __hmin(__hmax(in_reg_0.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_0;
    out_reg_0.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_0.x))));
    out_reg_0.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_0.y))));
    HALF2(output + idx) = out_reg_0;
  }
  if ((idx + 2) < N) {
    half2 in_reg_1 = HALF2(input + idx + 2);
    in_reg_1.x = __hmin(__hmax(in_reg_1.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_1.y = __hmin(__hmax(in_reg_1.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_1;
    out_reg_1.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_1.x))));
    out_reg_1.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_1.y))));
    HALF2(output + idx + 2) = out_reg_1;
  }
  if ((idx + 4) < N) {
    half2 in_reg_2 = HALF2(input + idx + 4);
    in_reg_2.x = __hmin(__hmax(in_reg_2.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_2.y = __hmin(__hmax(in_reg_2.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_2;
    out_reg_2.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_2.x))));
    out_reg_2.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_2.y))));
    HALF2(output + idx + 4) = out_reg_2;
  }
  if ((idx + 6) < N) {
    half2 in_reg_3 = HALF2(input + idx + 6);
    in_reg_3.x = __hmin(__hmax(in_reg_3.x, MIN_EXP_FP16), MAX_EXP_FP16);
    in_reg_3.y = __hmin(__hmax(in_reg_3.y, MIN_EXP_FP16), MAX_EXP_FP16);
    half2 out_reg_3;
    out_reg_3.x = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_3.x))));
    out_reg_3.y = __hdiv(__float2half(1.0f),
                         __hadd(__float2half(1.0f), hexp(__hneg(in_reg_3.y))));
    HALF2(output + idx + 6) = out_reg_3;
  }
}

template <const size_t kBlockSize>
torch::Tensor sigmoid_fp32(torch::Tensor input) {
  return act_wrapper<float, 1, kBlockSize>(input, torch::kFloat32,
                                           sigmoid_fp32_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp32x2(torch::Tensor input) {
  return act_wrapper<float, 2, kBlockSize>(input, torch::kFloat32,
                                           sigmoid_fp32x2_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp32x4(torch::Tensor input) {
  return act_wrapper<float, 4, kBlockSize>(input, torch::kFloat32,
                                           sigmoid_fp32x4_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp16(torch::Tensor input) {
  return act_wrapper<half, 1, kBlockSize>(input, torch::kHalf,
                                          sigmoid_fp16_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp16x2(torch::Tensor input) {
  return act_wrapper<half, 2, kBlockSize>(input, torch::kHalf,
                                          sigmoid_fp16x2_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp16x8(torch::Tensor input) {
  return act_wrapper<half, 8, kBlockSize>(input, torch::kHalf,
                                          sigmoid_fp16x8_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_fp32", &sigmoid_fp32<CBLOCK_SIZE_1D>, "sigmoid_fp32");
  m.def("sigmoid_fp32x2", &sigmoid_fp32x2<CBLOCK_SIZE_1D>, "sigmoid_fp32x2");
  m.def("sigmoid_fp32x4", &sigmoid_fp32x4<CBLOCK_SIZE_1D>, "sigmoid_fp32x4");
  m.def("sigmoid_fp16", &sigmoid_fp16<CBLOCK_SIZE_1D>, "sigmoid_fp16");
  m.def("sigmoid_fp16x2", &sigmoid_fp16x2<CBLOCK_SIZE_1D>, "sigmoid_fp16x2");
  m.def("sigmoid_fp16x8", &sigmoid_fp16x8<CBLOCK_SIZE_1D>, "sigmoid_fp16x8");
}

} // namespace hpc::cuda
