#pragma once

#ifdef ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_acceler(const size_t &n, T *__restrict__ dst,
                         const T *__restrict__ src, const T &alpha) {
  using traits = acceler::blasl1_traits<T>;
  traits::axpy(n, dst, src, alpha);
}

// copy
template <typename T>
inline void copy_acceler(const size_t &n, T *__restrict__ dst,
                         const T *__restrict__ src) {
  using traits = acceler::blasl1_traits<T>;
  traits::copy(n, dst, src);
}

} // namespace details
} // namespace hpc::l1

#endif
