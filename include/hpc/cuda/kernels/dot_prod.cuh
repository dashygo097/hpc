#pragma once

#include "./reduce.cuh"
#include <cuda_fp16.h>

namespace hpc::cu {
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void dot_prod_fp32_kernel(float *out, float *in1, float *in2,
                                     size_t N);
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void dot_prod_fp16_kernel(half *out, half *in1, half *in2, size_t N);
} // namespace hpc::cu
