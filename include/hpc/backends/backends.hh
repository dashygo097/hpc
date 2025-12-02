#pragma once

#include "./cuda/cuda.hh"
#include "./mpi/mpi.hh"
#include "./openmp/openmp.hh"
#include "./simd/simd.hh"

namespace hpc {
enum class Backend {
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
};

inline const char *backend_name(Backend b) {
  switch (b) {
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
  default:
    return "Unknown";
  }
}

} // namespace hpc
