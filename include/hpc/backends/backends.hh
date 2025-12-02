#pragma once

// SIMD
#include "./simd/apple_simd_traits.hh"
#include "./simd/arm_neon_traits.hh"
#include "./simd/avx_traits.hh"
#include "./simd/sse_traits.hh"

// MPI

// CUDA

namespace hpc {
enum class Backend {
  // Single backend
  SEQUENTIAL,
#ifdef ENABLE_SIMD
  SIMD,
#endif
#ifdef ENABLE_OPENMP
  OPENMP,
#endif
#ifdef ENABLE_MPI
  MPI,
#endif
#ifdef ENABLE_CUDA
  CUDA,
#endif

// Multi backend (combinations)
#if defined(ENABLE_SIMD) && defined(ENABLE_OPENMP)
  OPENMP_SIMD,
#endif
};

inline const char *backend_name(Backend b) {
  switch (b) {
    // Single backend
  case Backend::SEQUENTIAL:
    return "Sequential";
#if ENABLE_SIMD
  case Backend::SIMD:
    return "SIMD";
#endif
#if ENABLE_OPENMP
  case Backend::OPENMP:
    return "OpenMP";
#endif
#if ENABLE_MPI
  case Backend::MPI:
    return "MPI";
#endif
#if ENABLE_CUDA
  case Backend::CUDA:
    return "CUDA";
#endif

// Multi backend (combinations)
#if defined(ENABLE_SIMD) && defined(ENABLE_OPENMP)
  case Backend::OPENMP_SIMD:
    return "OpenMP + SIMD";
#endif

  default:
    return "Unknown";
  }
}

} // namespace hpc
