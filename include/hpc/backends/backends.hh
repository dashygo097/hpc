#pragma once

// SIMD
#ifdef ENABLE_SIMD
#include "./simd/apple_simd_traits.hh"
#include "./simd/arm_neon_traits.hh"
#include "./simd/avx_traits.hh"
#include "./simd/sse_traits.hh"
#endif

// MPI
#ifdef ENABLE_MPI
#include "./mpi/comm.hh"
#include "./mpi/env.hh"
#endif

// CUDA
#ifdef ENABLE_CUDA
#include "./cuda/device.cuh"
#include "./cuda/err.cuh"
#include "./cuda/launcher.cuh"
#include "./cuda/macros.cuh"
#include "./cuda/memory.cuh"
#endif

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
