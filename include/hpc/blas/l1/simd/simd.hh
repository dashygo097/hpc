#pragma once

#ifdef ENABLE_SIMD
#include "./general.hh"
#endif

#ifdef ENABLE_SIMD
#define ENABLE_SIMD_VECTOR_SCALAR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, src, scalar);            \
  }
#define ENABLE_SIMD_BINARY_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, src1, src2);             \
  }
#define ENABLE_SIMD_UNARY_BRANCH(name)                                         \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, src);                    \
  }
#define ENABLE_SIMD_SCALAR_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, scalar);                 \
  }
#define ENABLE_SIMD_REDUCE_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    return details::name##_simd<T, BackendParams...>(n, src);                  \
  }
#define ENABLE_SIMD_REDUCE2_BRANCH(name)                                       \
  else if constexpr (backend == Backend::SIMD) {                               \
    return details::name##_simd<T, BackendParams...>(n, src1, src2);           \
  }
#else
#define ENABLE_SIMD_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_SIMD_BINARY_BRANCH(name)
#define ENABLE_SIMD_UNARY_BRANCH(name)
#define ENABLE_SIMD_SCALAR_BRANCH(name)
#define ENABLE_SIMD_REDUCE_BRANCH(name)
#define ENABLE_SIMD_REDUCE2_BRANCH(name)
#endif
