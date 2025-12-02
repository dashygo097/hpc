#pragma once

#include "../../backends/backends.hh"

#ifdef ENABLE_OPENMP
namespace hpc::op {
namespace details {

template <typename T, size_t BlockSize>
inline void vadd_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] += scalar;
  }
}

template <typename T, size_t BlockSize>
inline void vadd_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] += src[i];
  }
}

template <typename T, size_t BlockSize>
inline void vsub_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] -= scalar;
}

template <typename T, size_t BlockSize>
inline void vsub_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] -= src[i];
}

template <typename T, size_t BlockSize>
inline void vmul_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] *= scalar;
}

template <typename T, size_t BlockSize>
inline void vmul_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] *= src[i];
}

template <typename T, size_t BlockSize>
inline void vdiv_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] /= scalar;
}

template <typename T, size_t BlockSize>
inline void vdiv_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] /= src[i];
}

template <typename T, size_t BlockSize>
inline void vfill_omp(T *__restrict__ dst, const T &value, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] = value;
}

} // namespace details
} // namespace hpc::op
#endif
