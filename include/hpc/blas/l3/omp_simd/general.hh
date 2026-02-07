#pragma once

#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
#include "../../../backends/backends.hh"
#include <algorithm>
#endif

#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
namespace hpc::l3 {
namespace details {

// gemm
template <typename T, const size_t TileSize, const size_t SimdWidth,
          const size_t Alignment>
inline void gemm_omp_simd(const size_t &M, const size_t &K, const size_t &N,
                          T *__restrict__ C, const T *__restrict__ A,
                          const T *__restrict__ B, const T &alpha = T{1},
                          const T &beta = T{0}) {
  static_assert(TileSize % SimdWidth == 0,
                "TileSize must be multiple of SimdWidth");
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  simd_t zero_vec = SIMD_DUP(traits, T{0});
  simd_t alpha_vec = SIMD_DUP(traits, alpha);
  simd_t beta_vec = SIMD_DUP(traits, beta);
  size_t simd_total_size = (M * N - SimdWidth + 1) / SimdWidth * SimdWidth;

  // scale C by beta
  if (beta != T{1}) {
    size_t i = 0;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_total_size; i += SimdWidth) {
      simd_t c_vec = SIMD_LOAD(traits, C + i);
      SIMD_STORE(traits, C + i, SIMD_MUL(traits, beta_vec, c_vec));
    }
    for (; i < M * N; ++i) {
      C[i] *= beta;
    }
  } else if (beta == T{0}) {
    // init C
    size_t i = 0;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_total_size; i += SimdWidth) {
      SIMD_STORE(traits, C + i, zero_vec);
    }
    for (; i < M * N; ++i) {
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
        for (size_t idx = 0; idx < TileSize * TileSize; idx += SimdWidth) {
          SIMD_STORE(traits, localC + idx, zero_vec);
        }

        for (size_t kk = 0; kk < K; kk += TileSize) {
          const size_t tile_k = std::min(TileSize, K - kk);

          // load localA
          for (size_t i = 0; i < tile_m; ++i) {
            memcpy(localA + i * TileSize, A + (ii + i) * K + kk,
                   tile_k * sizeof(T));
            memset(localA + i * TileSize + tile_k, 0,
                   (TileSize - tile_k) * sizeof(T));
          }
          memset(localA + tile_m * TileSize, 0,
                 (TileSize - tile_m) * TileSize * sizeof(T));

          // load localB
          for (size_t k = 0; k < tile_k; ++k) {
            memcpy(localB + k * TileSize, B + (kk + k) * N + jj,
                   tile_n * sizeof(T));
            memset(localB + k * TileSize + tile_n, 0,
                   (TileSize - tile_n) * sizeof(T));
          }
          memset(localB + tile_k * TileSize, 0,
                 (TileSize - tile_k) * TileSize * sizeof(T));

          // compute
          for (size_t i = 0; i < TileSize; ++i) {
            T *c_row = localC + i * TileSize;
            const T *a_row = localA + i * TileSize;

            for (size_t k = 0; k < TileSize; ++k) {
              const T a_ik = a_row[k];
              const simd_t a_ik_vec = SIMD_DUP(traits, a_ik);
              const T *b_row = localB + k * TileSize;

              size_t j = 0;
              for (; j + SimdWidth <= TileSize; j += SimdWidth) {
                simd_t b_vec = SIMD_LOAD(traits, b_row + j);
                simd_t c_vec = SIMD_LOAD(traits, c_row + j);
                SIMD_STORE(traits, c_row + j,
                           SIMD_FMA(traits, a_ik_vec, b_vec, c_vec));
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
}

} // namespace details
} // namespace hpc::l3
#endif
