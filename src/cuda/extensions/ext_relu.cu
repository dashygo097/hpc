#include "hpc/cuda/bindings.cuh"
#include "hpc/cuda/kernels/relu.cuh"

namespace hpc::cu {
template <const size_t kBlockSize> torch::Tensor relu_fp32(torch::Tensor in) {
  return act_wrapper<float, 1, kBlockSize>(in, torch::kFloat32,
                                           relu_fp32_kernel);
}
template <const size_t kBlockSize> torch::Tensor relu_fp16(torch::Tensor in) {
  return act_wrapper<half, 1, kBlockSize>(in, torch::kHalf, relu_fp16_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_fp32", &relu_fp32<CBLOCK_SIZE_1D>, "relu_fp32");
  m.def("relu_fp16", &relu_fp16<CBLOCK_SIZE_1D>, "relu_fp16");
}
} // namespace hpc::cu
