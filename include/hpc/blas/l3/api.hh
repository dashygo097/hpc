#pragma once

#include "../../backends/backends.hh"
#include "./accelerate.hh"
#include "./cuda.cuh"
#include "./omp.hh"
#include "./omp_simd/omp_simd.hh"
#include "./openblas.hh"
#include "./sequential.hh"
#include "./simd/simd.hh"
#include <cstddef>

#define L3_GEMM_FACTORY(name)                                                  \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(const T &M, const T &K, const T &N, T *__restrict__ C,      \
                   const T *__restrict__ A, const T *__restrict__ B,           \
                   const T &alpha = T{1}, const T &beta = T{0}) {              \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(M, K, N, C, A, B, alpha, beta); \
    }                                                                          \
    HPC_ENABLE_GEMM_SIMD_BRANCH(name)                                          \
    HPC_ENABLE_GEMM_OPENMP_BRANCH(name)                                        \
    HPC_ENABLE_GEMM_OPENMP_SIMD_BRANCH(name)                                   \
    HPC_ENABLE_GEMM_OPENBLAS_BRANCH(name)                                      \
    HPC_ENABLE_GEMM_ACCELERATE_BRANCH(name)                                    \
  }

namespace hpc::l3 {
// Public API
L3_GEMM_FACTORY(gemm)

} // namespace hpc::l3
