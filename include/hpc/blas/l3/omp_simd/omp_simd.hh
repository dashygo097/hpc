#pragma once

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
#include "./general.hh"
#endif

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
#define ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)                                   \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(M, K, N, C, A, B, alpha,     \
                                                  beta);                       \
  }
#else
#define ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)
#endif
