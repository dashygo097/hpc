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
template <typename T> struct blasl1_traits;

template <> struct blasl1_traits<float> {
  // axpy
  __attribute__((always_inline)) static inline void
  axpy(const size_t &n, float *y, const float *x, const float &alpha) {
    cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
  }

  // copy
  __attribute__((always_inline)) static inline void
  copy(const size_t &n, float *dst, const float *src) {
    cblas_scopy(static_cast<int>(n), src, 1, dst, 1);
  }

  // scal
  __attribute__((always_inline)) static inline void
  scal(const size_t &n, float *x, const float &alpha) {
    cblas_sscal(static_cast<int>(n), alpha, x, 1);
  }

  // dot
  __attribute__((always_inline)) static inline float
  dot(const size_t &n, const float *x, const float *y) {
    return cblas_sdot(static_cast<int>(n), x, 1, y, 1);
  }
};

template <> struct blasl1_traits<double> {
  // axpy
  __attribute__((always_inline)) static inline void
  axpy(const size_t &n, double *y, const double *x, const double &alpha) {
    cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
  }

  // copy
  __attribute__((always_inline)) static inline void
  copy(const size_t &n, double *dst, const double *src) {
    cblas_dcopy(static_cast<int>(n), src, 1, dst, 1);
  }

  // scal
  __attribute__((always_inline)) static inline void
  scal(const size_t &n, double *x, const double &alpha) {
    cblas_dscal(static_cast<int>(n), alpha, x, 1);
  }

  // dot
  __attribute__((always_inline)) static inline double
  dot(const size_t &n, const double *x, const double *y) {
    return cblas_ddot(static_cast<int>(n), x, 1, y, 1);
  }
};

} // namespace hpc::openblas
#endif
