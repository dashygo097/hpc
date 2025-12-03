#pragma once

#include "../backends/backends.hh"

#ifdef ENABLE_SIMD
namespace hpc::l3 {
namespace details {

template <typename T, const size_t TileSize, const size_t SimdWidth,
          const size_t Alignment>
void tiled_mmul_1xk_simd(T *__restrict__ C, const T *__restrict__ A,
                         const T *__restrict__ B, const size_t &M,
                         const size_t &K, const size_t &N) {
  std::runtime_error(TileSize % SimdWidth != 0
                         ? "TileSize must be multiple of SimdWidth"
                         : "");

  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t N_simd = N - N % SimdWidth;
  size_t is_remain = N % SimdWidth;

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

      for (size_t idx = 0; idx < TileSize * TileSize; idx += SimdWidth) {
        localC[idx] = T{};
      }
      for (size_t kk = 0; kk < K; kk += TileSize) {
        size_t k_end = std::min(kk + TileSize, K);
        size_t tile_k = k_end - kk;

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
            // compute
            for (size_t i = 0; i < TileSize; i++) {
              for (size_t k = 0; k < TileSize; ++k) {
                T a_ik = localA[i * TileSize + k];
                simd_t a_ik_vec = traits::duplicate(a_ik);
                for (size_t j = 0; j < tile_n; j += SimdWidth) {
                  if (j + SimdWidth <= tile_n) {
                    simd_t b_kj = traits::load(localB + k * TileSize + j);
                    simd_t c_ij = traits::load(localC + i * TileSize + j);
                    traits::store(
                        localC + i * TileSize + j,
                        traits::add(c_ij, traits::mul(a_ik_vec, b_kj)));
                  } else {
                    for (size_t jj = j; jj < tile_n; ++jj) {
                      localC[i * TileSize + jj] +=
                          a_ik * localB[k * TileSize + jj];
                    }
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
  }
}

} // namespace details
} // namespace hpc::l3
#endif
