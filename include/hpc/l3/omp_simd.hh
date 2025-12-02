#pragma once

#include "../backends/backends.hh"

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
namespace hpc::l3 {
namespace details {}
} // namespace hpc::l3
#endif
