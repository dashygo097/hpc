#pragma once

#include "../../backends/backends.hh"
#include "./acceler.hh"
#include "./omp.hh"
#include "./omp_simd.hh"
#include "./sequential.hh"
#include "./simd.hh"
#include <cstddef>

#define L3_MMUL_FACTORY(name)                                                  \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(T *__restrict__ C, const T *__restrict__ A,                 \
                   const T *__restrict__ B, const T &M, const T &K,            \
                   const T &N) {                                               \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(C, A, B, M, K, N);              \
    }                                                                          \
    ENABLE_MMUL_SIMD_BRANCH(name)                                              \
    ENABLE_MMUL_OPENMP_BRANCH(name)                                            \
    ENABLE_MMUL_OPENMP_SIMD_BRANCH(name)                                       \
    ENABLE_MMUL_ACCEL_BRANCH(name)                                             \
  }

#define L3_GEMM_FACTORY(name)                                                  \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(T *__restrict__ C, const T *__restrict__ A,                 \
                   const T *__restrict__ B, const T &M, const T &K,            \
                   const T &N, const T &alpha, const T &beta) {                \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(C, A, B, M, K, N, alpha, beta); \
    }                                                                          \
    ENABLE_GEMM_SIMD_BRANCH(name)                                              \
    ENABLE_GEMM_OPENMP_BRANCH(name)                                            \
    ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)                                       \
    ENABLE_GEMM_ACCEL_BRANCH(name)                                             \
  }

#ifdef ENABLE_SIMD
#define ENABLE_MMUL_SIMD_BRANCH(name)                                          \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(C, A, B, M, K, N);               \
  }
#define ENABLE_GEMM_SIMD_BRANCH(name)                                          \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(C, A, B, M, K, N, alpha, beta);  \
  }
#else
#define ENABLE_MMUL_SIMD_BRANCH(name)
#define ENABLE_GEMM_SIMD_BRANCH(name)
#endif

#ifdef ENABLE_OPENMP
#define ENABLE_MMUL_OPENMP_BRANCH(name)                                        \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(C, A, B, M, K, N);                \
  }
#define ENABLE_GEMM_OPENMP_BRANCH(name)                                        \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(C, A, B, M, K, N, alpha, beta);   \
  }
#else
#define ENABLE_MMUL_OPENMP_BRANCH(name)
#define ENABLE_GEMM_OPENMP_BRANCH(name)
#endif

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
#define ENABLE_MMUL_OPENMP_SIMD_BRANCH(name)                                   \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(C, A, B, M, K, N);           \
  }
#define ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)                                   \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(C, A, B, M, K, N, alpha,     \
                                                  beta);                       \
  }
#else
#define ENABLE_MMUL_OPENMP_SIMD_BRANCH(name)
#define ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)
#endif

#ifdef ENABLE_ACCELERATE
#define ENABLE_MMUL_ACCEL_BRANCH(name)                                         \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T>(C, A, B, M, K, N);                              \
  }
#define ENABLE_GEMM_ACCEL_BRANCH(name)                                         \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T>(C, A, B, M, K, N, alpha, beta);                 \
  }
#else
#define ENABLE_MMUL_ACCEL_BRANCH(name)
#define ENABLE_GEMM_ACCEL_BRANCH(name)
#endif

namespace hpc::l3 {
L3_MMUL_FACTORY(mmul)
L3_GEMM_FACTORY(gemm)
} // namespace hpc::l3
