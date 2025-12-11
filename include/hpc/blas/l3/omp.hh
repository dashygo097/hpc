#pragma once

#ifdef ENABLE_OPENMP
#include "../../backends/backends.hh"
#include <algorithm>
#endif

#ifdef ENABLE_OPENMP
#define ENABLE_GEMM_OPENMP_BRANCH(name)                                        \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(M, K, N, C, A, B, alpha, beta);   \
  }
#else
#define ENABLE_GEMM_OPENMP_BRANCH(name)
#endif

#ifdef ENABLE_OPENMP
namespace hpc::l3 {
namespace details {

// gemm
template <typename T, const size_t TileSize, const size_t Alignment>
inline void gemm_omp(const size_t &M, const size_t &K, const size_t &N,
                     T *__restrict__ C, const T *__restrict__ A,
                     const T *__restrict__ B, const T &alpha = T{1},
                     const T &beta = T{0}) {

  // scale C by beta
  if (beta != T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M * N; ++i) {
      C[i] *= beta;
    }
  } else if (beta == T{0}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M * N; ++i) {
      C[i] = T{0};
    }
  }

  if (alpha == T{0}) {
    return;
  }
#pragma omp parallel
  {
    // local buffer
    alignas(Alignment) T localA[TileSize * TileSize];
    alignas(Alignment) T localB[TileSize * TileSize];
    alignas(Alignment) T localC[TileSize * TileSize];

#pragma omp for schedule(static)
    for (size_t ii = 0; ii < M; ii += TileSize) {
      const size_t i_end = std::min(ii + TileSize, M);
      const size_t tile_m = i_end - ii;

      for (size_t jj = 0; jj < N; jj += TileSize) {
        const size_t j_end = std::min(jj + TileSize, N);
        const size_t tile_n = j_end - jj;

        // init localC
        for (size_t idx = 0; idx < TileSize * TileSize; ++idx) {
          localC[idx] = T{0};
        }

        for (size_t kk = 0; kk < K; kk += TileSize) {
          const size_t k_end = std::min(kk + TileSize, K);
          const size_t tile_k = k_end - kk;

          // load localA
          for (size_t i = 0; i < tile_m; ++i) {
            for (size_t k = 0; k < tile_k; ++k) {
              localA[i * TileSize + k] = A[(ii + i) * K + (kk + k)];
            }
            for (size_t k = tile_k; k < TileSize; ++k) {
              localA[i * TileSize + k] = T{0};
            }
          }
          for (size_t i = tile_m; i < TileSize; ++i) {
            for (size_t k = 0; k < TileSize; ++k) {
              localA[i * TileSize + k] = T{0};
            }
          }

          // load localB
          for (size_t k = 0; k < tile_k; ++k) {
            for (size_t j = 0; j < tile_n; ++j) {
              localB[k * TileSize + j] = B[(kk + k) * N + (jj + j)];
            }
            for (size_t j = tile_n; j < TileSize; ++j) {
              localB[k * TileSize + j] = T{0};
            }
          }
          for (size_t k = tile_k; k < TileSize; ++k) {
            for (size_t j = 0; j < TileSize; ++j) {
              localB[k * TileSize + j] = T{0};
            }
          }

          // compute
          for (size_t i = 0; i < TileSize; ++i) {
            for (size_t k = 0; k < TileSize; ++k) {
              const T a_ik = localA[i * TileSize + k];
              for (size_t j = 0; j < TileSize; ++j) {
                localC[i * TileSize + j] += a_ik * localB[k * TileSize + j];
              }
            }
          }
        }

        // copy
        for (size_t i = 0; i < tile_m; ++i) {
          for (size_t j = 0; j < tile_n; ++j) {
            C[(ii + i) * N + (jj + j)] += alpha * localC[i * TileSize + j];
          }
        }
      }
    }
  }
}

} // namespace details
} // namespace hpc::l3
#endif
