#include "hpc/cuda/bindings.cuh"
#include "hpc/cuda/kernels/sigmoid.cuh"

namespace hpc::cu {
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp32(torch::Tensor in) {
  return act_wrapper<float, 1, kBlockSize>(in, torch::kFloat32,
                                           sigmoid_fp32_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp16(torch::Tensor in) {
  return act_wrapper<half, 1, kBlockSize>(in, torch::kHalf,
                                          sigmoid_fp16_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_fp32", &sigmoid_fp32<CBLOCK_SIZE_1D>, "sigmoid_fp32");
  m.def("sigmoid_fp16", &sigmoid_fp16<CBLOCK_SIZE_1D>, "sigmoid_fp16");
}

} // namespace hpc::cu
