#pragma once

#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
#include "./general.hh"
#endif

#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
#define HPC_ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)                               \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(M, K, N, C, A, B, alpha,     \
                                                  beta);                       \
  }
#else
#define HPC_ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)
#endif
