#pragma once

#include "../../backends/backends.hh"

namespace hpc::op {
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

} // namespace details

} // namespace hpc::op
