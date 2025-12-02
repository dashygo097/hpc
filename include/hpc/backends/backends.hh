#pragma once

// SIMD
#include "./simd/apple_simd_traits.hh"
#include "./simd/arm_neon_traits.hh"
#include "./simd/avx_traits.hh"
#include "./simd/sse_traits.hh"

// MPI
#include "./mpi/comm.hh"
#include "./mpi/env.hh"

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

} // namespace hpc
