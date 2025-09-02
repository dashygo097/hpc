#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/relu/relu.cuh"
#include <torch/extension.h>
#include <torch/types.h>

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

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)     \
  void relu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      relu_##packed_type##_kernel<<<grid, block>>>(                            \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_RELU(fp32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(fp32x2, torch::kFloat32, float, 2)
TORCH_BINDING_RELU(fp32x4, torch::kFloat32, float, 4)
TORCH_BINDING_RELU(fp16, torch::kHalf, half, 1)
TORCH_BINDING_RELU(fp16x2, torch::kHalf, half, 2)
TORCH_BINDING_RELU(fp16x8, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(relu_fp32)
  TORCH_BINDING_COMMON_EXTENSION(relu_fp32x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_fp32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_fp16)
  TORCH_BINDING_COMMON_EXTENSION(relu_fp16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_fp16x8)
}

} // namespace hpc::cuda
