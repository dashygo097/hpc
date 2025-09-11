#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/kernels/sigmoid.cuh"

namespace hpc::cu {
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp32(torch::Tensor in) {
  return act_wrapper<float, 1, kBlockSize>(in, torch::kFloat32,
                                           sigmoid_fp32_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp32x2(torch::Tensor in) {
  return act_wrapper<float, 2, kBlockSize>(in, torch::kFloat32,
                                           sigmoid_fp32x2_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp32x4(torch::Tensor in) {
  return act_wrapper<float, 4, kBlockSize>(in, torch::kFloat32,
                                           sigmoid_fp32x4_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp16(torch::Tensor in) {
  return act_wrapper<half, 1, kBlockSize>(in, torch::kHalf,
                                          sigmoid_fp16_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp16x2(torch::Tensor in) {
  return act_wrapper<half, 2, kBlockSize>(in, torch::kHalf,
                                          sigmoid_fp16x2_kernel);
}
template <const size_t kBlockSize>
torch::Tensor sigmoid_fp16x8(torch::Tensor in) {
  return act_wrapper<half, 8, kBlockSize>(in, torch::kHalf,
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

} // namespace hpc::cu
