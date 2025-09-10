#pragma once

#include <memory>

namespace hpc {
constexpr size_t PARALLEL_THRESHOLD_1D = 102400;
constexpr size_t PARALLEL_THRESHOLD_2D = 8192;
constexpr size_t BLOCK_SIZE = 16;
constexpr size_t GEMM_TILE_SIZE = 64;
constexpr size_t SIMD_WIDTH = 4;

// CUDA constants
constexpr size_t CWARP_SIZE = 32;
constexpr size_t CBLOCK_SIZE = 256;
constexpr size_t CBLOCK_SIZE_MAXIMUM = 1024;

static_assert(BLOCK_SIZE % SIMD_WIDTH == 0,
              "BLOCK_SIZE must be a multiple of SIMD_WIDTH");

enum class Backend {
  SERIAL,
  OPENMP,
  OPENMPI,
  CUDA,
};

}; // namespace hpc
