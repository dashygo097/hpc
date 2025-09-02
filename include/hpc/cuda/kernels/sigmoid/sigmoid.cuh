#pragma once
#include <cuda_fp16.h>

namespace hpc::cuda {
__global__ void sigmoid_fp32(float *output, const float *input, size_t N);
__global__ void sigmoid_fp32x4(float *output, const float *input, size_t N);
__global__ void sigmoid_fp16(half *output, const half *input, size_t N);
__global__ void sigmoid_fp16x2(half *output, const half *input, size_t N);
__global__ void sigmoid_fp16x8(half *output, const half *input, size_t N);
} // namespace hpc::cuda
