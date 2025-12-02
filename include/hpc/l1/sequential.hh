#pragma once

#include "../backends/backends.hh"

namespace hpc::l1 {
namespace details {

template <typename T>
inline void vadd_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] += scalar;
  }
}

template <typename T>
inline void vadd_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] += src[i];
  }
}

template <typename T>
inline void vsub_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] -= scalar;
  }
}

template <typename T>
inline void vsub_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] -= src[i];
  }
}

template <typename T>
inline void vmul_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] *= scalar;
  }
}

template <typename T>
inline void vmul_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] *= src[i];
  }
}

template <typename T>
inline void vdiv_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] /= scalar;
  }
}

template <typename T>
inline void vfill_seq(T *__restrict__ dst, const T &value, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = value;
  }
}

// l1
template <typename T>
inline void axpy_seq(T *__restrict__ y, const T *__restrict__ x, const T &a,
                     size_t n) {
  for (size_t i = 0; i < n; ++i) {
    y += a * x[i];
  }
}

} // namespace details

} // namespace hpc::l1
