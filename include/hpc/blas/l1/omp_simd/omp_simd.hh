#pragma once

#if defined(ENABLE_SIMD) && defined(ENABLE_OPENMP)
#include "./general.hh"
#include <cstring>
#endif

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
#define ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)                          \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src, scalar);        \
  }
#define ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src1, src2);         \
  }
#define ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)                                  \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src);                \
  }
#define ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, scalar);             \
  }
#define ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    return details::name##_omp_simd<T, BackendParams...>(n, src);              \
  }
#define ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)                                \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    return details::name##_omp_simd<T, BackendParams...>(n, src1, src2);       \
  }
#else
#define ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)
#define ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)
#define ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)
#define ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)
#endif
