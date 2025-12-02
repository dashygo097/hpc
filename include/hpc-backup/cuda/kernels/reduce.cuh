#pragma once

#include "../../constants.hh"
#include <cuda_fp16.h>

namespace hpc::cu {
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void reduce_sum_fp32_kernel(float *out, float *in, size_t N);
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void reduce_sum_fp16_kernel(half *out, half *in, size_t N);
} // namespace hpc::cu
