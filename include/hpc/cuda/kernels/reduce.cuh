#pragma once

#include "hpc/constants.hh"
#include <cuda_fp16.h>

namespace hpc::cu {
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void reduce_sum_fp32_kernel(float *output, float *input,
                                       size_t N);
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void reduce_sum_fp16_kernel(half *output, half *input,
                                       size_t N);

} // namespace hpc::cu
