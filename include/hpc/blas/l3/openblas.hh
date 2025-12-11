#pragma once

#ifdef ENABLE_OPENBLAS
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_OPENBLAS
#define ENABLE_GEMM_OPENBLAS_BRANCH(name)                                   \
  else if constexpr (backend == Backend::OPENBLAS) {                         \
    details::name##_openblas<T, BackendParams...>(M, K, N, C, A, B, alpha, beta); \
  }
#else
#define ENABLE_GEMM_OPENBLAS_BRANCH(name)
#endif

#ifdef ENABLE_OPENBLAS
namespace hpc::l3 {
namespace details {
template <typename T, typename... BackendParams>
inline void gemm_openblas(const size_t &M, const size_t &K, const size_t &N,
                            T *__restrict__ C, const T *__restrict__ A,
                            const T *__restrict__ B, const T &alpha = T{1},
                            const T &beta = T{0}) {
    using traits = openblas::blasl3_traits<T>;
    traits::gemm(M, N, K, C, A, B, alpha, beta);
                            }


    } // namespace details
    } // namespace hpc::l3
#endif