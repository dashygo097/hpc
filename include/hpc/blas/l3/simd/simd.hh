#pragma once

#ifdef ENABLE_SIMD
#include "./general.hh"
#endif

#ifdef ENABLE_SIMD
#define ENABLE_GEMM_SIMD_BRANCH(name)                                          \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(M, K, N, C, A, B, alpha, beta);  \
  }
#else
#define ENABLE_GEMM_SIMD_BRANCH(name)
#endif
