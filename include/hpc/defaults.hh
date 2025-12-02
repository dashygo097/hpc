#pragma once

#include <cstddef>

namespace hpc::defaults {
#ifdef ENABLE_SIMD
constexpr size_t SIMD_WIDTH = 4;
#endif

#ifdef ENABLE_OPENMP
constexpr size_t OMP_THRESHOLD_1D = 102400;
#endif

} // namespace hpc::defaults
