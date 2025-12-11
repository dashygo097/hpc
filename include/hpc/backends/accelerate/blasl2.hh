#pragma once

#ifdef ENABLE_ACCELERATE
#include <Accelerate/Accelerate.h>
#include <cstddef>
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::accelerate {
template <typename T> struct blasl2_traits;

template <> struct blasl2_traits<float> {
  using type = float;

  // gemv
  __attribute__((always_inline)) static inline void
  gemv(const size_t &M, const size_t &N, float *y, const float *A,
       const float *x, const float &alpha, const float &beta) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), alpha, A, static_cast<int>(N), x, 1, beta,
                y, 1);
  }
};

template <> struct blasl2_traits<double> {
  using type = double;

  // gemv
  __attribute__((always_inline)) static inline void
  gemv(const size_t &M, const size_t &N, double *y, const double *A,
       const double *x, const double &alpha, const double &beta) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), alpha, A, static_cast<int>(N), x, 1, beta,
                y, 1);
  }
};

} // namespace hpc::accelerate

#endif
