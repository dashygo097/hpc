#pragma once

#ifdef ENABLE_SIMD
#include "./simd_impl.hh"
#endif

namespace hpc::serial {
template <typename T> void vadd(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] += value;
  }
}
template <typename T> void vadd(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] += B[i];
  }
}

template <typename T> void vsub(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] -= value;
  }
}
template <typename T> void vsub(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] -= B[i];
  }
}

template <typename T> void vmul(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] *= value;
  }
}
template <typename T> void vmul(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] *= B[i];
  }
}

template <typename T> void vdiv(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] /= value;
  }
}
template <typename T> void vdiv(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] /= B[i];
  }
}

} // namespace hpc::serial
