#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/relu.cuh"

namespace hpc::cuda {
__global__ void relu_fp32_kernel(float *output, const float *input, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void relu_fp32x2_kernel(float *output, const float *input,
                                   size_t N) {
  size_t idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float2 in_reg = FLOAT2(input + idx);
    float2 out_reg;
    out_reg.x = fmaxf(0.0f, in_reg.x);
    out_reg.y = fmaxf(0.0f, in_reg.y);
    FLOAT2(output + idx) = out_reg;
  }
}

__global__ void relu_fp32x4_kernel(float *output, const float *input,
                                   size_t N) {
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

__global__ void relu_fp16_kernel(half *output, const half *input, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = __hmax(__float2half(0.0f), input[idx]);
  }
}

__global__ void relu_fp16x2_kernel(half *output, const half *input, size_t N) {
  size_t idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 in_reg = HALF2(input + idx);
    half2 out_reg;
    out_reg.x = __hmax(__float2half(0.0f), in_reg.x);
    out_reg.y = __hmax(__float2half(0.0f), in_reg.y);
    HALF2(output + idx) = out_reg;
  }
}

__global__ void relu_fp16x8_kernel(half *output, const half *input, size_t N) {
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

template <const size_t kBlockSize>
torch::Tensor relu_fp32(torch::Tensor input) {
  return act_wrapper<float, 1, kBlockSize>(input, torch::kFloat32,
                                           relu_fp32_kernel);
}
template <const size_t kBlockSize>
torch::Tensor relu_fp32x2(torch::Tensor input) {
  return act_wrapper<float, 2, kBlockSize>(input, torch::kFloat32,
                                           relu_fp32x2_kernel);
}
template <const size_t kBlockSize>
torch::Tensor relu_fp32x4(torch::Tensor input) {
  return act_wrapper<float, 4, kBlockSize>(input, torch::kFloat32,
                                           relu_fp32x4_kernel);
}
template <const size_t kBlockSize>
torch::Tensor relu_fp16(torch::Tensor input) {
  return act_wrapper<half, 1, kBlockSize>(input, torch::kHalf,
                                          relu_fp16_kernel);
}
template <const size_t kBlockSize>
torch::Tensor relu_fp16x2(torch::Tensor input) {
  return act_wrapper<half, 2, kBlockSize>(input, torch::kHalf,
                                          relu_fp16x2_kernel);
}
template <const size_t kBlockSize>
torch::Tensor relu_fp16x8(torch::Tensor input) {
  return act_wrapper<half, 8, kBlockSize>(input, torch::kHalf,
                                          relu_fp16x8_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_fp32", &relu_fp32<CBLOCK_SIZE_1D>, "relu_fp32");
  m.def("relu_fp32x2", &relu_fp32x2<CBLOCK_SIZE_1D>, "relu_fp32x2");
  m.def("relu_fp32x4", &relu_fp32x4<CBLOCK_SIZE_1D>, "relu_fp32x4");
  m.def("relu_fp16", &relu_fp16<CBLOCK_SIZE_1D>, "relu_fp16");
  m.def("relu_fp16x2", &relu_fp16x2<CBLOCK_SIZE_1D>, "relu_fp16x2");
  m.def("relu_fp16x8", &relu_fp16x8<CBLOCK_SIZE_1D>, "relu_fp16x8");
}

} // namespace hpc::cuda
