#pragma once

#include "./pch.hh"

namespace hpc::serial {
template <typename T>
void naive_mmul(T *C, T *A, T *B, const size_t &M, const size_t &K,
                const size_t &N) {
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
} // namespace hpc::serial

namespace hpc::openmp {
template <typename T>
void naive_mmul_impl(T *C, const T *A, const T *B, const size_t &M,
                     const size_t &K, const size_t &N) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; ++i) {
    C[i] = T{};
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      T a_ik = A[i * K + k];
      for (size_t j = 0; j < N; ++j) {
        C[i * N + j] += a_ik * B[k * N + j];
      }
    }
  }
}

template <typename T>
void tiled_mmul_impl(T *C, const T *A, const T *B, const size_t &M,
                     const size_t &K, const size_t &N,
                     const size_t &tile_size) {

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; ++i) {
    C[i] = T{};
  }

#pragma omp parallel for schedule(static)
  for (size_t ii = 0; ii < M; ii += tile_size) {
    for (size_t jj = 0; jj < N; jj += tile_size) {
      for (size_t kk = 0; kk < K; kk += tile_size) {
        size_t i_end = std::min(ii + tile_size, M);
        size_t j_end = std::min(jj + tile_size, N);
        size_t k_end = std::min(kk + tile_size, K);
        for (size_t i = ii; i < i_end; ++i) {
          for (size_t k = kk; k < k_end; ++k) {
            T a_ik = A[i * K + k];
            for (size_t j = jj; j < j_end; ++j) {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

} // namespace hpc::openmp
