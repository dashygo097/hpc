#pragma once

#ifdef HPC_ENABLE_SIMD
#include "./general.hh"
#endif

#ifdef HPC_ENABLE_SIMD
#define HPC_ENABLE_GEMM_SIMD_BRANCH(name)                                      \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(M, K, N, C, A, B, alpha, beta);  \
  }
#else
#define HPC_ENABLE_GEMM_SIMD_BRANCH(name)
#endif
