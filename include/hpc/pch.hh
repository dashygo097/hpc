#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <omp.h>
#include <string>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <simd/simd.h>

#elif defined(__ARM_NEON)
#include <arm_neon.h>

#elif defined(__AVX2__)
#include <immintrin.h>
#endif
