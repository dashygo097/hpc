#pragma once

#include "hpc/constants.hh"
#include "hpc/mmul_impl.hh"
#include "hpc/pch.hh"
#include "hpc/timer.hh"

#ifdef ENABLE_OPENMP
#include "hpc/openmp/openmp.hh"
#endif

#ifdef ENABLE_MPI
#include "hpc/mpi/mpi.hh"
#endif

#ifdef ENABLE_SIMD
#include "hpc/simd.hh"
#endif
