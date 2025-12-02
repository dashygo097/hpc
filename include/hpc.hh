#pragma once

#include "hpc/constants.hh"
#include "hpc/impl.hh"
#include "hpc/timer.hh"

#ifdef ENABLE_OPENMP
#include "hpc/openmp/openmp.hh"
#endif

#ifdef ENABLE_MPI
#include "hpc/mpi/mpi.hh"
#endif

#ifdef __CUDACC__
#include "hpc/cuda/cuda.cuh"
#endif
