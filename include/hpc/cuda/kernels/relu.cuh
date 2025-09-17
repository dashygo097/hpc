#pragma once

#include <cuda_fp16.h>

namespace hpc::cu {
__global__ void relu_fp32_kernel(float *out, float *in, size_t N);
__global__ void relu_fp16_kernel(half *out, half *in, size_t N);
} // namespace hpc::cu
