#pragma once

#include <memory>

namespace hpc {
constexpr size_t PARALLEL_THRESHOLD_1D = 102400;
constexpr size_t PARALLEL_THRESHOLD_2D = 8192;
constexpr size_t BLOCK_DIM = 64;
constexpr size_t GEMM_TILE_SIZE = 64;
constexpr size_t SIMD_WIDTH = 4;

// CUDA constants
constexpr size_t WARP_SIZE = 32;

static_assert(BLOCK_DIM % SIMD_WIDTH == 0,
              "BLOCK_SIZE must be a multiple of SIMD_WIDTH");

enum class Backend {
  SERIAL,
  OPENMP,
  OPENMPI,
  CUDA,
};
}; // namespace hpc
