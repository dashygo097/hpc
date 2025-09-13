#include "hpc/cuda/kernels/sgemm.cuh"

namespace hpc::cu {
void __global__ sgemm_naive_kernel(float* C, float* A, float* B, size_t M, size_t K, size_t N, float alpha, float beta) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * value + beta * C[row * N + col];
    }
}

template <const size_t kBlockSizeM, const size_t kBlockSizeK, const size_t kBlockSizeN>
void __global__ sgemm_sliced_kernel(float* C, float* A, float* B, size_t M, size_t K, size_t N, float alpha, float beta) {\}

} // namespace hpc::cu