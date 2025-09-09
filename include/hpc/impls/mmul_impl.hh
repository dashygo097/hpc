#pragma once

#include "../pch.hh"

#ifdef ENABLE_SIMD
#include "./simd_impl.hh"
#endif

namespace hpc::serial {
template <typename T>
void mmul_baseline(T *C, T *A, T *B, const size_t &M, const size_t &K,
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

template <typename T, const size_t kTileSize>
void tiled_mmul_impl(T *C, const T *A, const T *B, const size_t &M,
                     const size_t &K, const size_t &N) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; ++i) {
    C[i] = T{};
  }

#pragma omp parallel
  {
    // local buffer
    alignas(64) T localA[kTileSize * kTileSize];
    alignas(64) T localB[kTileSize * kTileSize];
    alignas(64) T localC[kTileSize * kTileSize];
#pragma omp for schedule(static)
    for (size_t ii = 0; ii < M; ii += kTileSize) {
      for (size_t jj = 0; jj < N; jj += kTileSize) {
        size_t i_end = std::min(ii + kTileSize, M);
        size_t j_end = std::min(jj + kTileSize, N);
        size_t tile_m = i_end - ii;
        size_t tile_n = j_end - jj;

        for (size_t idx = 0; idx < kTileSize * kTileSize; ++idx) {
          localC[idx] = T{};
        }

        for (size_t kk = 0; kk < K; kk += kTileSize) {
          size_t k_end = std::min(kk + kTileSize, K);
          size_t tile_k = k_end - kk;

          for (size_t i = 0; i < tile_m; ++i) {
            for (size_t k = 0; k < tile_k; ++k) {
              localA[i * kTileSize + k] = A[(ii + i) * K + (kk + k)];
            }
            for (size_t k = tile_k; k < kTileSize; ++k) {
              localA[i * kTileSize + k] = T{};
            }
          }
          for (size_t i = tile_m; i < kTileSize; ++i) {
            for (size_t k = 0; k < kTileSize; ++k) {
              localA[i * kTileSize + k] = T{};
            }
          }

          for (size_t k = 0; k < tile_k; ++k) {
            for (size_t j = 0; j < tile_n; ++j) {
              localB[k * kTileSize + j] = B[(kk + k) * N + (jj + j)];
            }
            for (size_t j = tile_n; j < kTileSize; ++j) {
              localB[k * kTileSize + j] = T{};
            }
          }
          for (size_t k = tile_k; k < kTileSize; ++k) {
            for (size_t j = 0; j < kTileSize; ++j) {
              localB[k * kTileSize + j] = T{};
            }
          }

          // compute
          for (size_t i = 0; i < kTileSize; ++i) {
            for (size_t k = 0; k < kTileSize; ++k) {
              T a_ik = localA[i * kTileSize + k];
              for (size_t j = 0; j < kTileSize; ++j) {
                localC[i * kTileSize + j] += a_ik * localB[k * kTileSize + j];
              }
            }
          }
        }

        // copy
        for (size_t i = 0; i < tile_m; ++i) {
          for (size_t j = 0; j < tile_n; ++j) {
            C[(ii + i) * N + (jj + j)] = localC[i * kTileSize + j];
          }
        }
      }
    }
  }
}

#ifdef ENABLE_SIMD
template <typename T, const size_t kSimdWidth>
void naive_mmul_simd_1xk_impl(T *C, const T *A, const T *B, size_t M, size_t K,
                              size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  size_t N_simd = N - N % kSimdWidth;
  size_t is_remain = N % kSimdWidth;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; i += kSimdWidth) {
    *((simd_t *)(C + i)) = simd_t{};
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      T a_ik = A[i * K + k];
      for (size_t j = 0; j < N_simd; j += kSimdWidth) {
        simd_t b_kj = *((simd_t *)(B + k * N + j));
        *((simd_t *)(C + i * N + j)) += a_ik * b_kj;
      }
      if (is_remain) {
        for (size_t j = N_simd; j < N; ++j) {
          C[i * N + j] += a_ik * B[k * N + j];
        }
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  size_t N_simd = N - N % kSimdWidth;
  size_t is_remain = N % kSimdWidth;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; i += kSimdWidth) {
    *(simd_t *)(C + i) = simd_t{};
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      T a_ik = A[i * K + k];
      for (size_t j = 0; j < N; j += kSimdWidth) {
        simd_t a_ik_vec = traits::duplicate(a_ik);
        simd_t b_kj = traits::load(B + k * N + j);
        simd_t c_ij = traits::load(C + i * N + j);
        traits::store(C + i * N + j,
                      traits::add(c_ij, traits::mul(a_ik_vec, b_kj)));
      }
      if (is_remain) {
        for (size_t j = N_simd; j < N; ++j) {
          C[i * N + j] += a_ik * B[k * N + j];
        }
      }
    }
  }

#else

  std::cerr << "Not Implement SIMD for `naive_mmul_simd_1xk_impl` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kTileSize, const size_t kSimdWidth>
void tiled_mmul_simd_1xk_impl(T *C, const T *A, const T *B, size_t M, size_t K,
                              size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  size_t N_simd = N - N % kSimdWidth;
  size_t is_remain = N % kSimdWidth;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < M * N; i += kSimdWidth) {
    *((simd_t *)(C + i)) = simd_t{};
  }

#pragma omp parallel
  {
    // local buffer
    alignas(64) T localA[kTileSize * kTileSize];
    alignas(64) T localB[kTileSize * kTileSize];
    alignas(64) T localC[kTileSize * kTileSize];
#pragma omp for schedule(static)
    for (size_t ii = 0; ii < M; ii += kTileSize) {
      for (size_t jj = 0; jj < N; jj += kTileSize) {
        size_t i_end = std::min(ii + kTileSize, M);
        size_t j_end = std::min(jj + kTileSize, N);
        size_t tile_m = i_end - ii;
        size_t tile_n = j_end - jj;
        for (size_t idx = 0; idx < kTileSize * kTileSize; ++idx) {
          localC[idx] = T{};
        }
        for (size_t kk = 0; kk < K; kk += kTileSize) {
          size_t k_end = std::min(kk + kTileSize, K);
          size_t tile_k = k_end - kk;
          for (size_t i = 0; i < tile_m; ++i) {
            for (size_t k = 0; k < tile_k; ++k) {
              localA[i * kTileSize + k] = A[(ii + i) * K + (kk + k)];
            }
            for (size_t k = tile_k; k < kTileSize; ++k) {
              localA[i * kTileSize + k] = T{};
            }
          }
          for (size_t i = tile_m; i < kTileSize; ++i) {
            for (size_t k = 0; k < kTileSize; ++k) {
              localA[i * kTileSize + k] = T{};
            }
          }
          for (size_t k = 0; k < tile_k; ++k) {
            for (size_t j = 0; j < tile_n; ++j) {
              localB[k * kTileSize + j] = B[(kk + k) * N + (jj + j)];
            }
            for (size_t j = tile_n; j < kTileSize; ++j) {
              localB[k * kTileSize + j] = T{};
            }
          }
          for (size_t k = tile_k; k < kTileSize; ++k) {
            for (size_t j = 0; j < kTileSize; ++j) {
              localB[k * kTileSize + j] = T{};
            }
          }

          // compute
          for (size_t i = 0; i < kTileSize; i++) {
            for (size_t k = 0; k < kTileSize; ++k) {
              T a_ik = localA[i * kTileSize + k];
              for (size_t j = 0; j < tile_n; j += kSimdWidth) {
                if (j + kSimdWidth <= tile_n) {
                  simd_t b_kj = *((simd_t *)(localB + k * kTileSize + j));
                  simd_t c_ij = *((simd_t *)(localC + i * kTileSize + j));
                  *((simd_t *)(localC + i * kTileSize + j)) =
                      c_ij + a_ik * b_kj;
                } else {
                  for (size_t jj = j; jj < tile_n; ++jj) {
                    localC[i * kTileSize + jj] +=
                        a_ik * localB[k * kTileSize + jj];
                  }
                }
              }
            }
          }

          // copy
          for (size_t i = 0; i < tile_m; ++i) {
            for (size_t j = 0; j < tile_n; ++j) {
              C[(ii + i) * N + (jj + j)] = localC[i * kTileSize + j];
            }
          }
        }
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `tiled_mmul_simd_1xk_impl` function for "
               "this platform."
            << std::endl;
#endif
}

#endif
} // namespace hpc::openmp
