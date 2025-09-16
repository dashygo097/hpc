#pragma once

#include "../../constants.hh"

namespace hpc::cu {
__global__ void sgemm_naive_kernel(float *C, float *A, float *B, size_t M,
                                   size_t K, size_t N, float alpha, float beta);
template <const size_t kBlockSize = CGEMM_SBLOCK_SIZE,
          const size_t kWarpSize = CWARP_SIZE>
__global__ void sgemm_smem_kernel(float *C, float *A, float *B, size_t M,
                                  size_t K, size_t N, float alpha, float beta);
} // namespace hpc::cu
