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

#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#endif

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#ifdef ENABLE_CUDA
#include "./cuda/utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#endif
