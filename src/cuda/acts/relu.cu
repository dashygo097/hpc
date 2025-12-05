#include <hpc.hh>

namespace hpc::nn {
__global__ void relu_f32_kernel(float* output, const float* input,
                                size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

::torch::Tensor relu_f32(::torch::Tensor input) {
    return bindings::torch::elem_wrapper<float, 1, 256>(relu_f32_kernel, input,
                                                ::torch::kFloat32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_f32", &relu_f32, "ReLU activation function for float32 tensors");
}

} // namespace hpc::nn

