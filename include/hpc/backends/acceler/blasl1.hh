#pragma once

#ifdef ENABLE_ACCELERATE
#include <Accelerate/Accelerate.h>
#include <cstddef>
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::acceler {
template <typename T> struct blasl1_traits;

template <> struct blasl1_traits<float> {
  using type = float;

  __attribute__((always_inline)) static inline void
  copy(const size_t &n, float *dst, const float *src) {
    cblas_scopy(static_cast<int>(n), src, 1, dst, 1);
  }

  __attribute__((always_inline)) static inline void
  axpy(size_t n, float *y, const float *x, const float &alpha) {
    cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
  }

  __attribute__((always_inline)) static inline void
  scal(const size_t &n, float *x, const float &alpha) {
    cblas_sscal(static_cast<int>(n), alpha, x, 1);
  }

  __attribute__((always_inline)) static inline type
  dot(size_t n, const float *x, const float *y) {
    return cblas_sdot(static_cast<int>(n), x, 1, y, 1);
  }
};

template <> struct blasl1_traits<double> {
  using type = double;

  __attribute__((always_inline)) static inline void
  copy(const size_t &n, double *dst, const double *src) {
    cblas_dcopy(static_cast<int>(n), src, 1, dst, 1);
  }

  __attribute__((always_inline)) static inline void
  axpy(const size_t &n, double *y, const double *x, const double &alpha) {
    cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
  }

  __attribute__((always_inline)) static inline void
  scal(const size_t &n, double *x, const double &alpha) {
    cblas_dscal(static_cast<int>(n), alpha, x, 1);
  }

  __attribute__((always_inline)) static inline type
  dot(const size_t &n, const double *x, const double *y) {
    return cblas_ddot(static_cast<int>(n), x, 1, y, 1);
  }
};

} // namespace hpc::acceler

#endif
