#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/kernels/reduce.cuh"

namespace hpc::cu {
template <const size_t kBlockSize>
torch::Tensor reduce_sum_fp32(torch::Tensor in) {
  return reduce_wrapper<float, 1, kBlockSize>(
      in, torch::kFloat32, reduce_sum_fp32_kernel<kBlockSize>);
}
template <const size_t kBlockSize>
torch::Tensor reduce_sum_fp16(torch::Tensor in) {
  return reduce_wrapper<half, 1, kBlockSize>(
      in, torch::kHalf, reduce_sum_fp16_kernel<kBlockSize>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum_fp32", &reduce_sum_fp32<CBLOCK_SIZE_1D>, "reduce_sum_fp32");
  m.def("reduce_sum_fp16", &reduce_sum_fp16<CBLOCK_SIZE_1D>, "reduce_sum_fp16");
}
} // namespace hpc::cu
