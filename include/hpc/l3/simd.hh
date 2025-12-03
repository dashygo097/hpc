#pragma once

#include "../backends/backends.hh"

#ifdef ENABLE_SIMD
namespace hpc::l3 {
namespace details {

template <typename T, const size_t TileSize, const size_t SimdWidth,
          const size_t Alignment>
void tiled_mmul_simd(T *__restrict__ C, const T *__restrict__ A,
                     const T *__restrict__ B, const size_t &M, const size_t &K,
                     const size_t &N) {
  static_assert(TileSize % SimdWidth == 0,
                "TileSize must be multiple of SimdWidth");

  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  simd_t zero_vec = SIMD_DUP(traits, T{});
  size_t total_size = M * N;

  // init C
  size_t i = 0;
  for (size_t i = 0; i + SimdWidth < total_size; i += SimdWidth) {
    SIMD_STORE(traits, C + i, zero_vec);
  }
  for (; i < total_size; ++i) {
    C[i] = T{};
  }

  // local buffer
  alignas(Alignment) T localA[TileSize * TileSize];
  alignas(Alignment) T localB[TileSize * TileSize];
  alignas(Alignment) T localC[TileSize * TileSize];

  for (size_t ii = 0; ii < M; ii += TileSize) {
    size_t i_end = std::min(ii + TileSize, M);
    size_t tile_m = i_end - ii;

    for (size_t jj = 0; jj < N; jj += TileSize) {
      size_t j_end = std::min(jj + TileSize, N);
      size_t tile_n = j_end - jj;

      // init localC
      for (size_t idx = 0; idx < TileSize * TileSize; idx += SimdWidth) {
        SIMD_STORE(traits, localC + idx, zero_vec);
      }

      for (size_t kk = 0; kk < K; kk += TileSize) {
        const size_t tile_k = std::min(TileSize, K - kk);

        // load localA
        for (size_t i = 0; i < tile_m; ++i) {
          std::memcpy(localA + i * TileSize, A + (ii + i) * K + kk,
                      tile_k * sizeof(T));
          std::memset(localA + i * TileSize + tile_k, 0,
                      (TileSize - tile_k) * sizeof(T));
        }
        std::memset(localA + tile_m * TileSize, 0,
                    (TileSize - tile_m) * TileSize * sizeof(T));

        // load localB
        for (size_t k = 0; k < tile_k; ++k) {
          std::memcpy(localB + k * TileSize, B + (kk + k) * N + jj,
                      tile_n * sizeof(T));
          std::memset(localB + k * TileSize + tile_n, 0,
                      (TileSize - tile_n) * sizeof(T));
        }
        std::memset(localB + tile_k * TileSize, 0,
                    (TileSize - tile_k) * TileSize * sizeof(T));

        // compute
        for (size_t i = 0; i < TileSize; ++i) {
          T *c_row = localC + i * TileSize;
          const T *a_row = localA + i * TileSize;

          for (size_t k = 0; k < TileSize; ++k) {
            const T a_ik = a_row[k];
            const T *b_row = localB + k * TileSize;

            size_t j = 0;
            for (; j + SimdWidth <= TileSize; j += SimdWidth) {
              simd_t b_vec = SIMD_LOAD(traits, b_row + j);
              simd_t c_vec = SIMD_LOAD(traits, c_row + j);
              *(simd_t *)(c_row + j) = SIMD_FMA(traits, a_ik, b_vec, c_vec);
            }

            for (; j < TileSize; ++j) {
              c_row[j] += a_ik * b_row[j];
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

template <typename T, const size_t TileSize, const size_t SimdWidth,
          const size_t Alignment>
inline void tiled_gemm_simd(T *__restrict__ C, const T *__restrict__ A,
                            const T *__restrict__ B, const size_t &M,
                            const size_t &K, const size_t &N, const T &alpha,
                            const T &beta) {
  static_assert(TileSize % SimdWidth == 0,
                "TileSize must be multiple of SimdWidth");
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  simd_t zero_vec = SIMD_DUP(traits, T{});
  simd_t alpha_vec = SIMD_DUP(traits, alpha);
  simd_t beta_vec = SIMD_DUP(traits, beta);
  size_t total_size = M * N;

  // scale c by beta
  size_t i = 0;
  for (size_t i = 0; i + SimdWidth < total_size; i += SimdWidth) {
    simd_t c_vec = SIMD_LOAD(traits, C + i);
    SIMD_STORE(traits, C + i, SIMD_MUL(traits, beta_vec, c_vec));
  }
  for (; i < total_size; ++i) {
    C[i] *= beta;
  }

  // local buffer
  alignas(Alignment) T localA[TileSize * TileSize];
  alignas(Alignment) T localB[TileSize * TileSize];
  alignas(Alignment) T localC[TileSize * TileSize];

  for (size_t ii = 0; ii < M; ii += TileSize) {
    size_t i_end = std::min(ii + TileSize, M);
    size_t tile_m = i_end - ii;

    for (size_t jj = 0; jj < N; jj += TileSize) {
      size_t j_end = std::min(jj + TileSize, N);
      size_t tile_n = j_end - jj;

      // init localC
      for (size_t idx = 0; idx < TileSize * TileSize; idx += SimdWidth) {
        SIMD_STORE(traits, localC + idx, zero_vec);
      }

      for (size_t kk = 0; kk < K; kk += TileSize) {
        const size_t tile_k = std::min(TileSize, K - kk);

        // load localA
        for (size_t i = 0; i < tile_m; ++i) {
          std::memcpy(localA + i * TileSize, A + (ii + i) * K + kk,
                      tile_k * sizeof(T));
          std::memset(localA + i * TileSize + tile_k, 0,
                      (TileSize - tile_k) * sizeof(T));
        }
        std::memset(localA + tile_m * TileSize, 0,
                    (TileSize - tile_m) * TileSize * sizeof(T));

        // load localB
        for (size_t k = 0; k < tile_k; ++k) {
          std::memcpy(localB + k * TileSize, B + (kk + k) * N + jj,
                      tile_n * sizeof(T));
          std::memset(localB + k * TileSize + tile_n, 0,
                      (TileSize - tile_n) * sizeof(T));
        }
        std::memset(localB + tile_k * TileSize, 0,
                    (TileSize - tile_k) * TileSize * sizeof(T));

        // compute
        for (size_t i = 0; i < TileSize; ++i) {
          T *c_row = localC + i * TileSize;
          const T *a_row = localA + i * TileSize;

          for (size_t k = 0; k < TileSize; ++k) {
            const T a_ik = a_row[k];
            const T *b_row = localB + k * TileSize;

            size_t j = 0;
            for (; j + SimdWidth <= TileSize; j += SimdWidth) {
              simd_t b_vec = SIMD_LOAD(traits, b_row + j);
              simd_t c_vec = SIMD_LOAD(traits, c_row + j);
              SIMD_STORE(traits, c_row + j,
                         SIMD_FMA(traits, a_ik, b_vec, c_vec));
            }
            for (; j < TileSize; ++j) {
              c_row[j] += a_ik * b_row[j];
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

} // namespace details
} // namespace hpc::l3
#endif
