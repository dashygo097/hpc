#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/sigmoid.cuh"

namespace hpc::cuda {
__global__ void sigmoid_fp32_kernel(float *output, const float *input,
                                    size_t N) {}
__global__ void sigmoid_fp32x2_kernel(float *output, const float *input,
                                      size_t N) {}
__global__ void sigmoid_fp32x4_kernel(float *output, const float *input,
                                      size_t N) {}
__global__ void sigmoid_fp16_kernel(half *output, const half *input, size_t N) {
}
__global__ void sigmoid_fp16x2_kernel(half *output, const half *input,
                                      size_t N) {}
__global__ void sigmoid_fp16x8_kernel(half *output, const half *input,
                                      size_t N) {}
} // namespace hpc::cuda
