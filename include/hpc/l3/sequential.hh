#pragma once

#include <cstddef>

namespace hpc::l3 {
namespace details {

template <typename T>
inline void naive_mmul_seq(T *C, const T *A, const T *B, const size_t &M,
                           const size_t &K, const size_t &N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T sum = T{};
      for (size_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

template <typename T>
inline void naive_gemm_seq(T *C, const T *A, const T *B, const size_t &M,
                           const size_t &K, const size_t &N, const T &alpha,
                           const T &beta) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T sum = T{};
      for (size_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
}

} // namespace details
} // namespace hpc::l3
