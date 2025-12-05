#pragma once

#ifdef ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::l3 {
namespace details {

template <typename T>
inline void mmul_acceler(T *__restrict__ C, const T *__restrict__ A,
                         const T *__restrict__ B, size_t M, size_t K,
                         size_t N) {
  using traits = acceler::blasl3_traits<T>;
  traits::gemm_simple(M, N, K, A, B, C);
}

template <typename T>
inline void gemm_acceler(T *__restrict__ C, const T *__restrict__ A,
                         const T *__restrict__ B, size_t M, size_t K, size_t N,
                         const T &alpha, const T &beta) {
  using traits = acceler::blasl3_traits<T>;
  traits::gemm_scaled(M, N, K, alpha, A, B, beta, C);
}

template <typename T>
inline void gemm_acceler(T *__restrict__ C, const T *__restrict__ A,
                         bool transA, const T *__restrict__ B, bool transB,
                         size_t M, size_t K, size_t N, const T &alpha,
                         const T &beta) {
  using traits = acceler::blasl3_traits<T>;

  CBLAS_TRANSPOSE cblas_transA = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE cblas_transB = transB ? CblasTrans : CblasNoTrans;

  size_t lda = transA ? M : K;
  size_t ldb = transB ? K : N;
  size_t ldc = N;

  traits::gemm(CblasRowMajor, cblas_transA, cblas_transB, M, N, K, alpha, A,
               lda, B, ldb, beta, C, ldc);
}

} // namespace details
} // namespace hpc::l3
#endif
