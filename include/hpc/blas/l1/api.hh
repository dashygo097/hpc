#pragma once

#include "../../backends/backends.hh"
#include "./omp.hh"
#include "./omp_simd.hh"
#include "./sequential.hh"
#include "./simd.hh"
#include <cstddef>

#define L1_FACTORY(name)                                                       \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(T *__restrict__ dst, const T &scalar, size_t n) {           \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(dst, scalar, n);                \
    }                                                                          \
    ENABLE_SIMD_SCALAR_BRANCH(name)                                            \
    ENABLE_OPENMP_SCALAR_BRANCH(name)                                          \
    ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)                                     \
  }                                                                            \
                                                                               \
  template <typename T, Backend backend, auto... BackendParams>                \
  inline void name(T *__restrict__ dst, const T *__restrict__ src, size_t n) { \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T, BackendParams...>(dst, src, n);                   \
    }                                                                          \
    ENABLE_SIMD_VECTOR_BRANCH(name)                                            \
    ENABLE_OPENMP_VECTOR_BRANCH(name)                                          \
    ENABLE_OPENMP_SIMD_VECTOR_BRANCH(name)                                     \
  }

#ifdef ENABLE_SIMD
#define ENABLE_SIMD_SCALAR_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(dst, scalar, n);                 \
  }
#define ENABLE_SIMD_VECTOR_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(dst, src, n);                    \
  }
#else
#define ENABLE_SIMD_SCALAR_BRANCH(name)
#define ENABLE_SIMD_VECTOR_BRANCH(name)
#endif

#ifdef ENABLE_OPENMP
#define ENABLE_OPENMP_SCALAR_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T>(dst, scalar, n);                                    \
  }
#define ENABLE_OPENMP_VECTOR_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T>(dst, src, n);                                       \
  }

#else
#define ENABLE_OPENMP_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_VECTOR_BRANCH(name)
#endif

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
#define ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(dst, scalar, n);             \
  }
#define ENABLE_OPENMP_SIMD_VECTOR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(dst, src, n);                \
  }
#else
#define ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_SIMD_VECTOR_BRANCH(name)
#endif

namespace hpc::l1 {

// Public API
L1_FACTORY(vadd)
L1_FACTORY(vsub)
L1_FACTORY(vmul)
L1_FACTORY(vdiv)
L1_FACTORY(vfill)
L1_FACTORY(axpy)

#undef L1_FACTORY
#undef ENABLE_SIMD_SCALAR_BRANCH
#undef ENABLE_SIMD_VECTOR_BRANCH
#undef ENABLE_OPENMP_SCALAR_BRANCH
#undef ENABLE_OPENMP_VECTOR_BRANCH
#undef ENABLE_OPENMP_SIMD_SCALAR_BRANCH
#undef ENABLE_OPENMP_SIMD_VECTOR_BRANCH

} // namespace hpc::l1
