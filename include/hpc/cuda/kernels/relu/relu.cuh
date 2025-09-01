#pragma once

namespace hpc::cuda {
#ifdef ENABLE_CUDA
__global__ void relu_fp32(float *output, const float *input, size_t N);
__global__ void relu_fp32x2(float *output, const float *input, size_t N);
__global__ void relu_fp32x4(float *output, const float *input, size_t N);
__global__ void relu_fp16(half *output, const half *input, size_t N);
__global__ void relu_fp16x2(half *output, const half *input, size_t N);
__global__ void relu_fp16x8(half *output, const half *input, size_t N);
#endif
} // namespace hpc::cuda
