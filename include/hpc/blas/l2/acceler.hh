#pragma once

#ifdef ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::l2 {
namespace details {

// gemv
template <typename T>
inline void gemv_accelerate(const size_t &M, const size_t &N, T *__restrict__ y,
                            const T *__restrict__ A, const T *__restrict__ x,
                            const T &alpha, const T &beta) {
  using traits = accelerate::blasl2_traits<T>;
  traits::gemv(M, N, y, A, x, alpha, beta);
}

} // namespace details
} // namespace hpc::l2
#endif
