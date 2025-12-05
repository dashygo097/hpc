#include <hpc.hh>

namespace hpc::nn {
__global__ void relu_f32_kernel(float *output, const float *input, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void relu_f16_kernel(__half *output, const __half *input, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] =
        __hgt(input[idx], __float2half(0.0f)) ? input[idx] : __float2half(0.0f);
  }
}

::torch::Tensor relu_f32(::torch::Tensor input) {
  return hpc::bindings::torch::elem_wrapper<float, 1, 256>(
      relu_f32_kernel, input, ::torch::kFloat32);
}
::torch::Tensor relu_f16(::torch::Tensor input) {
  return hpc::bindings::torch::elem_wrapper<__half, 1, 256>(
      relu_f16_kernel, input, ::torch::kFloat16);
}
} // namespace hpc::nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_f32", &hpc::nn::relu_f32,
        "ReLU activation function for float32 tensors");
  m.def("relu_f16", &hpc::nn::relu_f16,
        "ReLU activation function for float16 tensors");
}
