#pragma once

#ifdef ENABLE_ACCELERATE
#include <Accelerate/Accelerate.h>
#include <cstddef>
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::acceler {
template <typename T> struct acceler_traits;

template <> struct acceler_traits<float> {
  using type = float;

  static inline void vsadd(const float *src, float scalar, float *dst,
                           size_t n) {
    vDSP_vsadd(src, 1, &scalar, dst, 1, n);
  }

  static inline void vsmul(const float *src, float scalar, float *dst,
                           size_t n) {
    vDSP_vsmul(src, 1, &scalar, dst, 1, n);
  }

  static inline void vsdiv(const float *src, float scalar, float *dst,
                           size_t n) {
    vDSP_vsdiv(src, 1, &scalar, dst, 1, n);
  }

  static inline void vadd(const float *a, const float *b, float *dst,
                          size_t n) {
    vDSP_vadd(a, 1, b, 1, dst, 1, n);
  }

  static inline void vsub(const float *a, const float *b, float *dst,
                          size_t n) {
    vDSP_vsub(b, 1, a, 1, dst, 1, n); // dst = a - b
  }

  static inline void vmul(const float *a, const float *b, float *dst,
                          size_t n) {
    vDSP_vmul(a, 1, b, 1, dst, 1, n);
  }

  static inline void vdiv(const float *a, const float *b, float *dst,
                          size_t n) {
    vDSP_vdiv(b, 1, a, 1, dst, 1, n); // dst = a / b
  }

  static inline void vfill(float value, float *dst, size_t n) {
    vDSP_vfill(&value, dst, 1, n);
  }

  static inline void axpy(size_t n, float alpha, const float *x, float *y) {
    cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
  }
};

template <> struct acceler_traits<double> {
  using type = double;

  static inline void vsadd(const double *src, double scalar, double *dst,
                           size_t n) {
    vDSP_vsaddD(src, 1, &scalar, dst, 1, n);
  }

  static inline void vsmul(const double *src, double scalar, double *dst,
                           size_t n) {
    vDSP_vsmulD(src, 1, &scalar, dst, 1, n);
  }

  static inline void vsdiv(const double *src, double scalar, double *dst,
                           size_t n) {
    vDSP_vsdivD(src, 1, &scalar, dst, 1, n);
  }

  static inline void vadd(const double *a, const double *b, double *dst,
                          size_t n) {
    vDSP_vaddD(a, 1, b, 1, dst, 1, n);
  }

  static inline void vsub(const double *a, const double *b, double *dst,
                          size_t n) {
    vDSP_vsubD(b, 1, a, 1, dst, 1, n);
  }

  static inline void vmul(const double *a, const double *b, double *dst,
                          size_t n) {
    vDSP_vmulD(a, 1, b, 1, dst, 1, n);
  }

  static inline void vdiv(const double *a, const double *b, double *dst,
                          size_t n) {
    vDSP_vdivD(b, 1, a, 1, dst, 1, n);
  }

  static inline void vfill(double value, double *dst, size_t n) {
    vDSP_vfillD(&value, dst, 1, n);
  }

  static inline void axpy(size_t n, double alpha, const double *x, double *y) {
    cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
  }
};

} // namespace hpc::acceler

#endif // ENABLE_ACCELERATE
