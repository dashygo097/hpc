#include "hpc/cuda/kernels/sgemm.cuh"

namespace hpc::cu {
void __global__ sgemm_naive_kernel(float *C, float *A, float *B, size_t M,
                                   size_t K, size_t N, float alpha,
                                   float beta) {
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idy < M && idx < N) {
    float value = 0.0f;
    for (size_t k = 0; k < K; ++k) {
      value += A[idy * K + k] * B[k * N + idx];
    }
    C[idy * N + idx] = alpha * value + beta * C[idy * N + idx];
  }
}

template <const size_t kBlockSizeM, const size_t kBlockSizeK,
          const size_t kBlockSizeN, const size_t kWarpSize>
void __global__ sgemm_smem_kernel(float *C, float *A, float *B, size_t M,
                                  size_t K, size_t N, float alpha, float beta) {
  __shared__ float As[kBlockSizeM][kBlockSizeK];
  __shared__ float Bs[kBlockSizeK][kBlockSizeN];

  size_t kblock_num = (K + kBlockSizeK - 1) / kBlockSizeK;

  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tidy = threadIdx.y;
  size_t tidx = threadIdx.x;

  if (idx < N && idy < M) {
    float value = 0.0f;
    for (size_t block_idx = 0; block_idx < (K + kBlockSizeK - 1) / kBlockSizeK;
         ++block_idx) {
      if (block_idx * kBlockSizeK + threadIdx.x < K) {
        As[tidy][tidx] = A[idy * K + block_idx * kBlockSizeK + threadIdx.x];
      } else {
        As[tidy][tidx] = 0.0f;
      }
      if (block_idx * kBlockSizeK + threadIdx.y < K) {
        Bs[tidy][tidx] = B[(block_idx * kBlockSizeK + threadIdx.y) * N + idx];
      } else {
        Bs[tidy][tidx] = 0.0f;
      }
      __syncthreads();

      for (size_t k = 0; k < kBlockSizeK; ++k) {
        value += As[tidy][k] * Bs[k][tidx];
      }
      __syncthreads();
    }

    C[idy * N + idx] = alpha * value + beta * C[idy * N + idx];
  }
}

} // namespace hpc::cu
