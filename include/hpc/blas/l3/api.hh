#pragma once

#include "../../backends/backends.hh"
#include "./acceler.hh"
#include "./cuda.cuh"
#include "./omp.hh"
#include "./omp_simd.hh"
#include "./sequential.hh"
#include "./simd.hh"
#include <cstddef>

#define L3_GEMM_FACTORY(name)                                                  \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(const T &M, const T &K, const T &N, T *__restrict__ C,      \
                   const T *__restrict__ A, const T *__restrict__ B,           \
                   const T &alpha = T{1}, const T &beta = T{0}) {              \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(M, K, N, C, A, B, alpha, beta); \
    }                                                                          \
    ENABLE_GEMM_SIMD_BRANCH(name)                                              \
    ENABLE_GEMM_OPENMP_BRANCH(name)                                            \
    ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)                                       \
    ENABLE_GEMM_ACCEL_BRANCH(name)                                             \
  }

#ifdef ENABLE_SIMD
#define ENABLE_GEMM_SIMD_BRANCH(name)                                          \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(M, K, N, C, A, B, alpha, beta);  \
  }
#else
#define ENABLE_GEMM_SIMD_BRANCH(name)
#endif

#ifdef ENABLE_OPENMP
#define ENABLE_GEMM_OPENMP_BRANCH(name)                                        \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(M, K, N, C, A, B, alpha, beta);   \
  }
#else
#define ENABLE_GEMM_OPENMP_BRANCH(name)
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

#ifdef ENABLE_CUDA
#define ENABLE_GEMM_CUDA_BRANCH(name)
#else
#define ENABLE_GEMM_CUDA_BRANCH(name)
#endif

#ifdef ENABLE_ACCELERATE
#define ENABLE_GEMM_ACCEL_BRANCH(name)                                         \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(M, K, N, C, A, B, alpha,      \
                                                 beta);                        \
  }
#else
#define ENABLE_GEMM_ACCEL_BRANCH(name)
#endif

namespace hpc::l3 {
// Public API
L3_GEMM_FACTORY(gemm)

} // namespace hpc::l3
