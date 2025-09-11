#include "hpc/cuda/binding.cuh"
#include "hpc/cuda/kernels/dot_prod.cuh"

namespace hpc::cu {
template <const size_t kBlockSize>
torch::Tensor dot_prod_fp32(torch::Tensor in) {
  return reduce_wrapper<float, 1, kBlockSize>(in, torch::kFloat32,
                                              dot_prod_fp32_kernel<kBlockSize>);
}
template <const size_t kBlockSize>
torch::Tensor dot_prod_fp16(torch::Tensor in) {
  return reduce_wrapper<half, 1, kBlockSize>(in, torch::kHalf,
                                             dot_prod_fp16_kernel<kBlockSize>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dot_prod_fp32", &dot_prod_fp32<CBLOCK_SIZE_1D>, "dot_prod_fp32");
  m.def("dot_prod_fp16", &dot_prod_fp16<CBLOCK_SIZE_1D>, "dot_prod_fp16");
}
} // namespace hpc::cu
