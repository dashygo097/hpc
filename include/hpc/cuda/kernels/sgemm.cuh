#pragma once

#include "../../constants.hh"

namespace hpc::cu {
__global__ void sgemm_naive_kernel(float *C, float *A, float *B, size_t M,
                                   size_t K, size_t N, float alpha, float beta);
template <const size_t kBlockSizeM = CGEMM_SLICE_SIZE_M,
          const size_t kBlockSizeK = CGEMM_SLICE_SIZE_K,
          const size_t kBlockSizeN = CGEMM_SLICE_SIZE_N>
__global__ void sgemm_sliced_kernel(float *C, float *A, float *B, size_t M,
                                    size_t K, size_t N, float alpha,
                                    float beta);
} // namespace hpc::cu
