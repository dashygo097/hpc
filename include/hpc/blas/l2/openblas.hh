#pragma once

#ifdef ENABLE_OPENBLAS
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_OPENBLAS
namespace hpc::l2 {
namespace details {

// gemv
template <typename T>
inline void gemv_openblas(const size_t &M, const size_t &N, T *__restrict__ y,
                          const T *__restrict__ A, const T *__restrict__ x,
                          const T &alpha, const T &beta) {
  using traits = openblas::blasl2_traits<T>;
  traits::gemv(M, N, y, A, x, alpha, beta);
}

} // namespace details
} // namespace hpc::l2
#endif
