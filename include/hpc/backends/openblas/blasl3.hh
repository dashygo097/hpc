#pragma once

#ifdef ENABLE_OPENBLAS
#include <cstddef>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#endif

#ifdef ENABLE_OPENBLAS
namespace hpc::openblas {
template <typename T> struct blasl3_traits;

template <> struct blasl3_traits<float> {
  // gemm
  __attribute__((always_inline)) static inline void
  gemm(const size_t &M, const size_t &N, const size_t &K, float *C,
       const float *A, const float *B, const float &alpha, const float &beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), alpha, A,
                static_cast<int>(K), B, static_cast<int>(N), beta, C,
                static_cast<int>(N));
  }
};

template <> struct blasl3_traits<double> {
  // gemm
  __attribute__((always_inline)) static inline void
  gemm(const size_t &M, const size_t &N, const size_t &K, double *C,
       const double *A, const double *B, const double &alpha,
       const double &beta) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), alpha, A,
                static_cast<int>(K), B, static_cast<int>(N), beta, C,
                static_cast<int>(N));
  }
};
} // namespace hpc::openblas
#endif
