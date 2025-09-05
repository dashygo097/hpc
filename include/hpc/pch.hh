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

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#ifdef ENABLE_SIMD

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <simd/simd.h>

#elif defined(__ARM_NEON)
#include <arm_neon.h>

#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#endif
