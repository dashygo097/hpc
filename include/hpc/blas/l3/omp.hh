#pragma once

#ifdef ENABLE_OPENMP
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_OPENMP
namespace hpc::l3 {
namespace details {

template <typename T, const size_t TileSize, const size_t Alignment>
void mmul_omp(T *__restrict__ C, const T *__restrict__ A,
              const T *__restrict__ B, const size_t &M, const size_t &K,
              const size_t &N) {
  // init C
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; ++i) {
    C[i] = T{};
  }

#pragma omp parallel
  {
    // local buffer
    alignas(Alignment) T localA[TileSize * TileSize];
    alignas(Alignment) T localB[TileSize * TileSize];
    alignas(Alignment) T localC[TileSize * TileSize];

#pragma omp for schedule(static)
    for (size_t ii = 0; ii < M; ii += TileSize) {
      size_t i_end = std::min(ii + TileSize, M);
      size_t tile_m = i_end - ii;

      for (size_t jj = 0; jj < N; jj += TileSize) {
        size_t j_end = std::min(jj + TileSize, N);
        size_t tile_n = j_end - jj;

        // init localC
        for (size_t idx = 0; idx < TileSize * TileSize; ++idx) {
          localC[idx] = T{};
        }

        for (size_t kk = 0; kk < K; kk += TileSize) {
          size_t k_end = std::min(kk + TileSize, K);
          size_t tile_k = k_end - kk;

          // load localA
          for (size_t i = 0; i < tile_m; ++i) {
            for (size_t k = 0; k < tile_k; ++k) {
              localA[i * TileSize + k] = A[(ii + i) * K + (kk + k)];
            }
            for (size_t k = tile_k; k < TileSize; ++k) {
              localA[i * TileSize + k] = T{};
            }
          }
          for (size_t i = tile_m; i < TileSize; ++i) {
            for (size_t k = 0; k < TileSize; ++k) {
              localA[i * TileSize + k] = T{};
            }
          }

          // load localB
          for (size_t k = 0; k < tile_k; ++k) {
            for (size_t j = 0; j < tile_n; ++j) {
              localB[k * TileSize + j] = B[(kk + k) * N + (jj + j)];
            }
            for (size_t j = tile_n; j < TileSize; ++j) {
              localB[k * TileSize + j] = T{};
            }
          }
          for (size_t k = tile_k; k < TileSize; ++k) {
            for (size_t j = 0; j < TileSize; ++j) {
              localB[k * TileSize + j] = T{};
            }
          }

          // compute
          for (size_t i = 0; i < TileSize; ++i) {
            for (size_t k = 0; k < TileSize; ++k) {
              T a_ik = localA[i * TileSize + k];
              for (size_t j = 0; j < TileSize; ++j) {
                localC[i * TileSize + j] += a_ik * localB[k * TileSize + j];
              }
            }
          }
        }

        // copy
        for (size_t i = 0; i < tile_m; ++i) {
          for (size_t j = 0; j < tile_n; ++j) {
            C[(ii + i) * N + (jj + j)] = localC[i * TileSize + j];
          }
        }
      }
    }
  }
}

template <typename T, const size_t TileSize, const size_t Alignment>
inline void gemm_omp(T *__restrict__ C, const T *__restrict__ A,
                     const T *__restrict__ B, const size_t &M, const size_t &K,
                     const size_t &N, const T &alpha, const T &beta) {

// scale C by beta
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; ++i) {
    C[i] *= beta;
  }

#pragma omp parallel
  {
    // local buffer
    alignas(Alignment) T localA[TileSize * TileSize];
    alignas(Alignment) T localB[TileSize * TileSize];
    alignas(Alignment) T localC[TileSize * TileSize];

#pragma omp for schedule(static)
    for (size_t ii = 0; ii < M; ii += TileSize) {
      size_t i_end = std::min(ii + TileSize, M);
      size_t tile_m = i_end - ii;

      for (size_t jj = 0; jj < N; jj += TileSize) {
        size_t j_end = std::min(jj + TileSize, N);
        size_t tile_n = j_end - jj;

        // init localC
        for (size_t idx = 0; idx < TileSize * TileSize; ++idx) {
          localC[idx] = T{};
        }

        for (size_t kk = 0; kk < K; kk += TileSize) {
          size_t k_end = std::min(kk + TileSize, K);
          size_t tile_k = k_end - kk;

          // load localA
          for (size_t i = 0; i < tile_m; ++i) {
            for (size_t k = 0; k < tile_k; ++k) {
              localA[i * TileSize + k] = A[(ii + i) * K + (kk + k)];
            }
            for (size_t k = tile_k; k < TileSize; ++k) {
              localA[i * TileSize + k] = T{};
            }
          }
          for (size_t i = tile_m; i < TileSize; ++i) {
            for (size_t k = 0; k < TileSize; ++k) {
              localA[i * TileSize + k] = T{};
            }
          }

          // load localB
          for (size_t k = 0; k < tile_k; ++k) {
            for (size_t j = 0; j < tile_n; ++j) {
              localB[k * TileSize + j] = B[(kk + k) * N + (jj + j)];
            }
            for (size_t j = tile_n; j < TileSize; ++j) {
              localB[k * TileSize + j] = T{};
            }
          }
          for (size_t k = tile_k; k < TileSize; ++k) {
            for (size_t j = 0; j < TileSize; ++j) {
              localB[k * TileSize + j] = T{};
            }
          }

          // compute
          for (size_t i = 0; i < TileSize; ++i) {
            for (size_t k = 0; k < TileSize; ++k) {
              T a_ik = localA[i * TileSize + k];
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
