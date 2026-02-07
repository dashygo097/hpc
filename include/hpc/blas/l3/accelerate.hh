#pragma once

#ifdef HPC_ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef HPC_ENABLE_ACCELERATE
#define HPC_ENABLE_GEMM_ACCELERATE_BRANCH(name)                                \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_accelerate<T, BackendParams...>(M, K, N, C, A, B, alpha,   \
                                                    beta);                     \
  }
#else
#define HPC_ENABLE_GEMM_ACCELERATE_BRANCH(name)
#endif

#ifdef HPC_ENABLE_ACCELERATE
namespace hpc::l3 {
namespace details {

// gemm
template <typename T>
inline void gemm_accelerate(const size_t &M, const size_t &K, const size_t &N,
                            T *__restrict__ C, const T *__restrict__ A,
                            const T *__restrict__ B, const T &alpha = T{1},
                            const T &beta = T{0}) {
  using traits = accelerate::blasl3_traits<T>;
  traits::gemm(M, K, N, C, A, B, alpha, beta);
}

} // namespace details
} // namespace hpc::l3
#endif
