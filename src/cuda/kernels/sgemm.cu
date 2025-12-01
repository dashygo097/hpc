#include "hpc/cuda/kernels/sgemm.cuh"

namespace hpc::cu {
__global__ void sgemm_naive_kernel(float *C, float *A, float *B, size_t M,
                                   size_t K, size_t N, float alpha,
                                   float beta) {
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idy < M && idx < N) {
    float c_val = 0.0f;
    #pragma unroll
    for (size_t k = 0; k < K; ++k) {
      c_val += A[idy * K + k] * B[k * N + idx];
    }
    C[idy * N + idx] = alpha * c_val + beta * C[idy * N + idx];
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
  
  float c_val = 0.0f;

  if (idx < N && idy < M) {
    for (size_t block_idx = 0; block_idx < (K + kBlockSize - 1) / kBlockSize;
         ++block_idx) {
      size_t a_col = block_idx * kBlockSize + threadIdx.x;
      size_t b_row = block_idx * kBlockSize + threadIdx.y;
      As[tidy][tidx] = (a_col < K) ? A[idy * K + a_col] : 0.0f;
      Bs[tidy][tidx] = (b_row < K) ? B[b_row * N + idx] : 0.0f;
      __syncthreads();

      #pragma unroll
      for (size_t k = 0; k < kBlockSize; ++k) {
        c_val += As[tidy][k] * Bs[k][tidx];
      }
      __syncthreads();
    }
  }
  
  if (idx < N && idy < M) {
    if (beta != 0.0f) {
      c_val = alpha * c_val + beta * C[idy * N + idx];
    } else {
      c_val = alpha * c_val;
    }
    C[idy * N + idx] = c_val;
  }
}

template <const size_t kBlockSize, const size_t kTileSize, const size_t kWarpSize>
__global__ void sgemm_tiled_kernel(float *C, float *A, float *B, size_t M, 
                                    size_t K, size_t N, float alpha, float beta) {
  __shared__ float As[kBlockSize][kTileSize];
  __shared__ float Bs[kTileSize][kBlockSize];
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
