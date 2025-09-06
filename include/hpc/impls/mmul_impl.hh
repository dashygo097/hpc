#pragma once

#include "../pch.hh"

#ifdef ENABLE_SIMD
#include "./simd_impl.hh"
#endif

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

#ifdef ENABLE_SIMD

template <typename T, const size_t kSimdWidth = SIMD_WIDTH>
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

  std::cerr << "Not Implement SIMD for `assign` function for "
               "this platform."
            << std::endl;
#endif
}

#endif

} // namespace hpc::openmp
