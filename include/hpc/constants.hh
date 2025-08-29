#pragma once

namespace hpc {
constexpr size_t PARALLEL_THRESHOLD_1D = 102400;
constexpr size_t PARALLEL_THRESHOLD_2D = 8192;
constexpr size_t BLOCK_SIZE = 64;
constexpr size_t GEMM_TILE_SIZE = 64;
constexpr size_t SIMD_WIDTH = 4;

enum class Backend {
  SERIAL,
  OPENMP,
  OPENMPI,
  CUDA,
};
}; // namespace hpc
