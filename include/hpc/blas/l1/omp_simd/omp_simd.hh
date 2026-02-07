#pragma once

#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
#include "./general.hh"
#include <cstring>
#endif

#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
#define HPC_ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)                      \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src, scalar);        \
  }
#define HPC_ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)                             \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src1, src2);         \
  }
#define HPC_ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)                              \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src);                \
  }
#define HPC_ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)                             \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, scalar);             \
  }
#define HPC_ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)                             \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    return details::name##_omp_simd<T, BackendParams...>(n, src);              \
  }
#define HPC_ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)                            \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    return details::name##_omp_simd<T, BackendParams...>(n, src1, src2);       \
  }
#else
#define HPC_ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)
#define HPC_ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)
#define HPC_ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)
#define HPC_ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)
#define HPC_ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)
#define HPC_ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)
#endif
