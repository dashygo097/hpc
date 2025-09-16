#include "hpc/cuda/kernels/sgemm.cuh"

namespace hpc::cu {
__global__ void sgemm_naive_kernel(float *C, float *A, float *B, size_t M,
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

template <const size_t kBlockSize, const size_t kWarpSize>
__global__ void sgemm_smem_kernel(float *C, float *A, float *B, size_t M,
                                  size_t K, size_t N, float alpha, float beta) {
  __shared__ float As[kBlockSize][kBlockSize];
  __shared__ float Bs[kBlockSize][kBlockSize];

  size_t idy = blockIdx.y * kBlockSize + threadIdx.y;
  size_t idx = blockIdx.x * kBlockSize + threadIdx.x;
  size_t tidy = threadIdx.y;
  size_t tidx = threadIdx.x;

  if (idx < N && idy < M) {
    float value = 0.0f;
    for (size_t block_idx = 0; block_idx < (K + kBlockSize - 1) / kBlockSize;
         ++block_idx) {
      if (block_idx * kBlockSize + threadIdx.x < K) {
        As[tidy][tidx] = A[idy * K + block_idx * kBlockSize + threadIdx.x];
      } else {
        As[tidy][tidx] = 0.0f;
      }
      if (block_idx * kBlockSize + threadIdx.y < K) {
        Bs[tidy][tidx] = B[(block_idx * kBlockSize + threadIdx.y) * N + idx];
      } else {
        Bs[tidy][tidx] = 0.0f;
      }
      __syncthreads();

      for (size_t k = 0; k < kBlockSize; ++k) {
        value += As[tidy][k] * Bs[k][tidx];
      }
      __syncthreads();
    }

    C[idy * N + idx] = alpha * value + beta * C[idy * N + idx];
  }
}


template __global__ void sgemm_smem_kernel<CGEMM_SBLOCK_SIZE, CWARP_SIZE>(
    float *C, float *A, float *B, size_t M, size_t K, size_t N, float alpha,
    float beta);
template __global__ void sgemm_smem_kernel<CGEMM_SBLOCK_SIZE / 2, CWARP_SIZE>(
    float *C, float *A, float *B, size_t M, size_t K, size_t N, float alpha,
    float beta);
template __global__ void sgemm_smem_kernel<CGEMM_SBLOCK_SIZE / 4, CWARP_SIZE>(
    float *C, float *A, float *B, size_t M, size_t K, size_t N, float alpha,
    float beta);
} // namespace hpc::cu
