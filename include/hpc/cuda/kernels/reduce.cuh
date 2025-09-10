#pragma once

#include "hpc/constants.hh"
#include <cuda_fp16.h>

namespace hpc::cuda {
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void reduce_sum_fp32_kernel(float *output, const float *input,
                                       size_t N);
template <const size_t kBlockSize = CBLOCK_SIZE_1D>
__global__ void reduce_sum_fp16_kernel(half *output, const half *input,
                                       size_t N);

} // namespace hpc::cuda
