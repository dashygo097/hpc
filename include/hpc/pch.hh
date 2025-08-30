#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

#include "./constants.hh"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_SIMD

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <simd/simd.h>
#endif

#endif

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif
