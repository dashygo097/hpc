#pragma once

#include "../backends/backends.hh"
#include "./omp.hh"
#include "./omp_simd.hh"
#include "./sequential.hh"
#include "./simd.hh"
#include <cstddef>

#define L1_FACTORY(name)                                                       \
  template <typename T, Backend backend, typename BackendConfig>               \
  inline void name(T *__restrict__ dst, const T &scaler, size_t n) {           \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T>(dst, scaler, n);                                  \
    } else if constexpr (backend == Backend::SIMD) {                           \
      details::name##_simd<T, BackendConfig::simd_width>(dst, scaler, n);      \
    } else if constexpr (backend == Backend::OPENMP) {                         \
      details::name##_omp<T, BackendConfig::tile_size>(dst, scaler, n);        \
    } else if constexpr (backend == Backend::OPENMP_SIMD) {                    \
      details::name##_omp_simd<T, BackendConfig::tile_size,                    \
                               BackendConfig::simd_width>(dst, scaler, n);     \
    } else {                                                                   \
      static_assert(false, "Unsupported backend for operation " #name);        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, Backend backend, typename BackendConfig>               \
  inline void name(T *__restrict__ dst, const T *__restrict__ src, size_t n) { \
    if constexpr (backend == Backend::SEQUENTIAL) {                            \
      details::name##_seq<T>(dst, src, n);                                     \
    } else if constexpr (backend == Backend::SIMD) {                           \
      details::name##_simd<T, BackendConfig::simd_width>(dst, src, n);         \
    } else if constexpr (backend == Backend::OPENMP) {                         \
      details::name##_omp<T, BackendConfig::tile_size>(dst, src, n);           \
    } else if constexpr (backend == Backend::OPENMP_SIMD) {                    \
      details::name##_omp_simd<T, BackendConfig::tile_size,                    \
                               BackendConfig::simd_width>(dst, src, n);        \
    } else {                                                                   \
      static_assert(false, "Unsupported backend for operation " #name);        \
    }                                                                          \
  }

namespace hpc::l1 {
// Public API
L1_FACTORY(vadd)
L1_FACTORY(vsub)
L1_FACTORY(vmul)
L1_FACTORY(vdiv)
L1_FACTORY(vfill)
L1_FACTORY(axpy)

} // namespace hpc::l1
