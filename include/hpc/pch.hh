#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_SIMD
#include "./simd.hh"
#endif

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif
