#pragma once

#include "../../backends/backends.hh"
#include "./acceler.hh"
#include "./cuda.cuh"
#include "./omp.hh"
#include "./omp_simd.hh"
#include "./sequential.hh"
#include "./simd.hh"
#include <cstddef>

#ifdef ENABLE_OPENMP
#define ENABLE_OPENMP_VECTOR_SCALAR_BRANCH(name)                               \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src, scalar);             \
  }
#define ENABLE_OPENMP_BINARY_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src1, src2);              \
  }
#define ENABLE_OPENMP_UNARY_BRANCH(name)                                       \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src);                     \
  }
#define ENABLE_OPENMP_SCALAR_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, scalar);                  \
  }
#define ENABLE_OPENMP_REDUCE_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    return details::name##_omp<T, BackendParams...>(n, src);                   \
  }
#define ENABLE_OPENMP_REDUCE2_BRANCH(name)                                     \
  else if constexpr (backend == Backend::OPENMP) {                             \
    return details::name##_omp<T, BackendParams...>(n, x, y);                  \
  }
#else
#define ENABLE_OPENMP_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_BINARY_BRANCH(name)
#define ENABLE_OPENMP_UNARY_BRANCH(name)
#define ENABLE_OPENMP_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_REDUCE_BRANCH(name)
#define ENABLE_OPENMP_REDUCE2_BRANCH(name)
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
    return details::name##_simd<T, BackendParams...>(n, x, y);                 \
  }
#else
#define ENABLE_SIMD_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_SIMD_BINARY_BRANCH(name)
#define ENABLE_SIMD_UNARY_BRANCH(name)
#define ENABLE_SIMD_SCALAR_BRANCH(name)
#define ENABLE_SIMD_REDUCE_BRANCH(name)
#define ENABLE_SIMD_REDUCE2_BRANCH(name)
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
    return details::name##_omp_simd<T, BackendParams...>(n, x, y);             \
  }
#else
#define ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)
#define ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)
#define ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)
#define ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)
#endif

#ifdef ENABLE_CUDA
#define ENABLE_CUDA_VECTOR_SCALAR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, src, scalar);            \
  }
#define ENABLE_CUDA_BINARY_BRANCH(name)                                        \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, src1, src2);             \
  }
#define ENABLE_CUDA_UNARY_BRANCH(name)                                         \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, src);                    \
  }
#define ENABLE_CUDA_SCALAR_BRANCH(name)                                        \
  else if constexpr (backend == Backend::CUDA) {                               \
    details::name##_cuda<T, BackendParams...>(n, dst, scalar);                 \
  }
#define ENABLE_CUDA_REDUCE_BRANCH(name)                                        \
  else if constexpr (backend == Backend::CUDA) {                               \
    return details::name##_cuda<T, BackendParams...>(n, src);                  \
  }
#define ENABLE_CUDA_REDUCE2_BRANCH(name)                                       \
  else if constexpr (backend == Backend::CUDA) {                               \
    return details::name##_cuda<T, BackendParams...>(n, x, y);                 \
  }
#else
#define ENABLE_CUDA_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_CUDA_BINARY_BRANCH(name)
#define ENABLE_CUDA_UNARY_BRANCH(name)
#define ENABLE_CUDA_SCALAR_BRANCH(name)
#define ENABLE_CUDA_REDUCE_BRANCH(name)
#define ENABLE_CUDA_REDUCE2_BRANCH(name)
#endif

#ifdef ENABLE_ACCELERATE
#define ENABLE_ACCELERATE_VECTOR_SCALAR_BRANCH(name)                           \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, src, scalar);         \
  }
#define ENABLE_ACCELERATE_BINARY_BRANCH(name)                                  \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, src1, src2);          \
  }
#define ENABLE_ACCELERATE_UNARY_BRANCH(name)                                   \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, src);                 \
  }
#define ENABLE_ACCELERATE_SCALAR_BRANCH(name)                                  \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, scalar);              \
  }
#define ENABLE_ACCELERATE_REDUCE_BRANCH(name)                                  \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    return details::name##_acceler<T, BackendParams...>(n, src);               \
  }
#define ENABLE_ACCELERATE_REDUCE2_BRANCH(name)                                 \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    return details::name##_acceler<T, BackendParams...>(n, x, y);              \
  }
#else
#define ENABLE_ACCELERATE_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_ACCELERATE_BINARY_BRANCH(name)
#define ENABLE_ACCELERATE_UNARY_BRANCH(name)
#define ENABLE_ACCELERATE_SCALAR_BRANCH(name)
#define ENABLE_ACCELERATE_REDUCE_BRANCH(name)
#define ENABLE_ACCELERATE_REDUCE2_BRANCH(name)
#endif

// L1 Factory Macros
#define L1_FACTORY_VECTOR_SCALAR(name)                                         \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(const size_t &n, T *__restrict__ dst,                       \
                   const T *__restrict__ src, const T &scalar) {               \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(n, dst, src, scalar);           \
    }                                                                          \
    ENABLE_OPENMP_VECTOR_SCALAR_BRANCH(name)                                   \
    ENABLE_SIMD_VECTOR_SCALAR_BRANCH(name)                                     \
    ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)                              \
    ENABLE_CUDA_VECTOR_SCALAR_BRANCH(name)                                     \
    ENABLE_ACCELERATE_VECTOR_SCALAR_BRANCH(name)                               \
  }

#define L1_FACTORY_BINARY(name)                                                \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(const size_t &n, T *__restrict__ dst,                       \
                   const T *__restrict__ src1, const T *__restrict__ src2) {   \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(n, dst, src1, src2);            \
    }                                                                          \
    ENABLE_OPENMP_BINARY_BRANCH(name)                                          \
    ENABLE_SIMD_BINARY_BRANCH(name)                                            \
    ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)                                     \
    ENABLE_CUDA_BINARY_BRANCH(name)                                            \
    ENABLE_ACCELERATE_BINARY_BRANCH(name)                                      \
  }

#define L1_FACTORY_UNARY(name)                                                 \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(const size_t &n, T *__restrict__ dst,                       \
                   const T *__restrict__ src) {                                \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(n, dst, src);                   \
    }                                                                          \
    ENABLE_OPENMP_UNARY_BRANCH(name)                                           \
    ENABLE_SIMD_UNARY_BRANCH(name)                                             \
    ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)                                      \
    ENABLE_CUDA_UNARY_BRANCH(name)                                             \
    ENABLE_ACCELERATE_UNARY_BRANCH(name)                                       \
  }

#define L1_FACTORY_SCALAR(name)                                                \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(const size_t &n, T *__restrict__ dst, const T &scalar) {    \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(n, dst, scalar);                \
    }                                                                          \
    ENABLE_OPENMP_SCALAR_BRANCH(name)                                          \
    ENABLE_SIMD_SCALAR_BRANCH(name)                                            \
    ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)                                     \
    ENABLE_CUDA_SCALAR_BRANCH(name)                                            \
    ENABLE_ACCELERATE_SCALAR_BRANCH(name)                                      \
  }

#define L1_FACTORY_REDUCE(name)                                                \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline T name(const size_t &n, const T *__restrict__ src) {                  \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      return details::name##_seq<T, BackendParams...>(n, src);                 \
    }                                                                          \
    ENABLE_OPENMP_REDUCE_BRANCH(name)                                          \
    ENABLE_SIMD_REDUCE_BRANCH(name)                                            \
    ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)                                     \
    ENABLE_CUDA_REDUCE_BRANCH(name)                                            \
    ENABLE_ACCELERATE_REDUCE_BRANCH(name)                                      \
  }

#define L1_FACTORY_REDUCE2(name)                                               \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline T name(const size_t &n, const T *__restrict__ src1,                   \
                const T *__restrict__ src2) {                                  \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      return details::name##_seq<T, BackendParams...>(n, x, y);                \
    }                                                                          \
    ENABLE_OPENMP_REDUCE2_BRANCH(name)                                         \
    ENABLE_SIMD_REDUCE2_BRANCH(name)                                           \
    ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)                                    \
    ENABLE_CUDA_REDUCE2_BRANCH(name)                                           \
    ENABLE_ACCELERATE_REDUCE2_BRANCH(name)                                     \
  }

namespace hpc::l1 {
// Public API
L1_FACTORY_VECTOR_SCALAR(axpy)
L1_FACTORY_UNARY(copy)

} // namespace hpc::l1

// Cleanup Macros
