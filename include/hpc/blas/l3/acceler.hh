#pragma once

#ifdef ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_ACCELERATE
#define ENABLE_GEMM_ACCEL_BRANCH(name)                                         \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(M, K, N, C, A, B, alpha,      \
                                                 beta);                        \
  }
#else
#define ENABLE_GEMM_ACCEL_BRANCH(name)
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::l3 {
namespace details {

// gemm
template <typename T>
inline void gemm_acceler(const size_t &M, const size_t &K, const size_t &N,
                         T *__restrict__ C, const T *__restrict__ A,
                         const T *__restrict__ B, const T &alpha = T{1},
                         const T &beta = T{0}) {
  using traits = acceler::blasl3_traits<T>;
  traits::gemm(M, K, N, C, A, B, alpha, beta);
}

} // namespace details
} // namespace hpc::l3
#endif
