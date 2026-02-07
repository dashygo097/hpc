#pragma once

// SIMD
#ifdef HPC_ENABLE_SIMD
#include "./simd/apple_simd_traits.hh"
#include "./simd/arm_neon_traits.hh"
#include "./simd/avx_traits.hh"
#include "./simd/sse_traits.hh"
#endif

// MPI
#ifdef HPC_ENABLE_MPI
#include "./mpi/comm.hh"
#include "./mpi/env.hh"
#endif

// CUDA
#ifdef HPC_ENABLE_CUDA
#include "./cuda/device.cuh"
#include "./cuda/err.cuh"
#include "./cuda/launcher.cuh"
#include "./cuda/macros.cuh"
#include "./cuda/memory.cuh"
#endif

// CUBLAS
#if defined(HPC_ENABLE_CUDA) && defined(ENABLE_CUBLAS)
#include "./cublas/blasl1.cuh"
#include "./cublas/blasl2.cuh"
#include "./cublas/blasl3.cuh"
#endif

// OpenBLAS
#ifdef HPC_ENABLE_OPENBLAS
#include "./openblas/blasl1.hh"
#include "./openblas/blasl2.hh"
#include "./openblas/blasl3.hh"
#endif

// Apple Accelerate
#ifdef HPC_ENABLE_ACCELERATE
#include "./accelerate/blasl1.hh"
#include "./accelerate/blasl2.hh"
#include "./accelerate/blasl3.hh"
#endif

// Prefetch API
#include "./prefetch/prefetch.hh"

namespace hpc {
enum class Backend {
  // Single backend
  SEQUENTIAL,
#ifdef HPC_ENABLE_SIMD
  SIMD,
#endif
#ifdef HPC_ENABLE_OPENMP
  OPENMP,
#endif
#ifdef HPC_ENABLE_MPI
  MPI,
#endif
#ifdef HPC_ENABLE_CUDA
  CUDA,
#endif
#if defined(HPC_ENABLE_CUDA) && defined(HPC_ENABLE_CUBLAS)
  CUBLAS,
#endif
#ifdef HPC_ENABLE_OPENBLAS
  OPENBLAS,
#endif
#ifdef HPC_ENABLE_ACCELERATE
  ACCELERATE,
#endif

// Multi backend (combinations)
#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
  OPENMP_SIMD,
#endif
};

} // namespace hpc
