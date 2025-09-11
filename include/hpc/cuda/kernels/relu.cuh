#pragma once

#include <cuda_fp16.h>

namespace hpc::cu {
__global__ void relu_fp32_kernel(float *output, float *input, size_t N);
__global__ void relu_fp32x2_kernel(float *output, float *input, size_t N);
__global__ void relu_fp32x4_kernel(float *output, float *input, size_t N);
__global__ void relu_fp16_kernel(half *output, half *input, size_t N);
__global__ void relu_fp16x2_kernel(half *output, half *input, size_t N);
__global__ void relu_fp16x8_kernel(half *output, half *input, size_t N);
} // namespace hpc::cu
