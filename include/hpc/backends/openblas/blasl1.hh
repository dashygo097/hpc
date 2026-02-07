#pragma once

#ifdef HPC_ENABLE_OPENBLAS
#include <cstddef>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#endif

#ifdef HPC_ENABLE_OPENBLAS
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

  // swap
  __attribute__((always_inline)) static inline void swap(const size_t &n,
                                                         float *x, float *y) {
    cblas_sswap(static_cast<int>(n), x, 1, y, 1);
  }

  // asum
  __attribute__((always_inline)) static inline float asum(const size_t &n,
                                                          const float *x) {
    return cblas_sasum(static_cast<int>(n), x, 1);
  }

  // nrm2
  __attribute__((always_inline)) static inline float nrm2(const size_t &n,
                                                          const float *x) {
    return cblas_snrm2(static_cast<int>(n), x, 1);
  }

  // iamax
  __attribute__((always_inline)) static inline size_t iamax(const size_t &n,
                                                            const float *x) {
    return static_cast<size_t>(cblas_isamax(static_cast<int>(n), x, 1));
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

  // swap
  __attribute__((always_inline)) static inline void swap(const size_t &n,
                                                         double *x, double *y) {
    cblas_dswap(static_cast<int>(n), x, 1, y, 1);
  }

  // asum
  __attribute__((always_inline)) static inline double asum(const size_t &n,
                                                           const double *x) {
    return cblas_dasum(static_cast<int>(n), x, 1);
  }

  // nrm2
  __attribute__((always_inline)) static inline double nrm2(const size_t &n,
                                                           const double *x) {
    return cblas_dnrm2(static_cast<int>(n), x, 1);
  }

  // iamax
  __attribute__((always_inline)) static inline size_t iamax(const size_t &n,
                                                            const double *x) {
    return static_cast<size_t>(cblas_idamax(static_cast<int>(n), x, 1));
  }
};

} // namespace hpc::openblas
#endif
