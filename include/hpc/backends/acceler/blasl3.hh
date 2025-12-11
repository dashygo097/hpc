#pragma once

#ifdef ENABLE_ACCELERATE
#include <Accelerate/Accelerate.h>
#include <cstddef>
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::acceler {
template <typename T> struct blasl3_traits;

template <> struct blasl3_traits<float> {
  using type = float;

  __attribute__((always_inline)) static inline void
    gemm(const size_t &M, const size_t &K, const size_t &N,
         float *__restrict__ C, const float *__restrict__ A,
         const float *__restrict__ B, const float &alpha,
         const float &beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K,
                B, N, beta, C, N);
  }
};

template <> struct blasl3_traits<double> {
  using type = double;

  __attribute__((always_inline)) static inline void
    gemm(const size_t &M, const size_t &K, const size_t &N,
         double *__restrict__ C, const double *__restrict__ A,
         const double *__restrict__ B, const double &alpha,
         const double &beta) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K,
                B, N, beta, C, N);
  }
};

} // namespace hpc::acceler

#endif // ENABLE_ACCELERATE
