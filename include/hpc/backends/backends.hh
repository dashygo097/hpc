#pragma once

#include "./cuda/cuda.hh"
#include "./mpi/mpi.hh"
#include "./openmp/openmp.hh"
#include "./simd/simd.hh"

namespace hpc {
enum class Backend {
  SEQUENTIAL,
#if defined(ENABLE_SIMD)
  SIMD,
#endif
#if defined(ENABLE_OPENMP)
  OPENMP,
#endif
#if defined(ENABLE_MPI)
  MPI,
#endif
#if defined(ENABLE_CUDA)
  CUDA,
#endif
};

inline const char *backend_name(Backend b) {
  switch (b) {
  case Backend::SEQUENTIAL:
    return "Sequential";
#if defined(ENABLE_SIMD) && defined(HAS_SIMD)
  case Backend::SIMD:
    return "SIMD";
#endif
#if defined(ENABLE_OPENMP) && defined(HAS_OPENMP)
  case Backend::OPENMP:
    return "OpenMP";
#endif
#if defined(ENABLE_MPI) && defined(HAS_MPI)
  case Backend::MPI:
    return "MPI";
#endif
#if defined(ENABLE_CUDA) && defined(HAS_CUDA)
  case Backend::CUDA:
    return "CUDA";
#endif
  default:
    return "Unknown";
  }
}

} // namespace hpc
