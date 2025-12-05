#pragma once

#ifdef ENABLE_ACCELERATE
#include <Accelerate/Accelerate.h>
#include <cstddef>
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::acceler {
template <typename T> struct blasl3_traits;

template <> struct blasl3_traits<float> {
  using type = float;

  __attribute__((always_inline)) static inline void
  gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
       size_t M, size_t N, size_t K, float alpha, const float *A, size_t lda,
       const float *B, size_t ldb, float beta, float *C, size_t ldc) {

    cblas_sgemm(order, transA, transB, static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(K), alpha, A, static_cast<int>(lda), B,
                static_cast<int>(ldb), beta, C, static_cast<int>(ldc));
  }

  __attribute__((always_inline)) static inline void
  gemm_simple(size_t M, size_t N, size_t K, const float *A, const float *B,
              float *C) {

    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
         1.0f, // alpha
         A, K, // lda = K (row-major, no transpose)
         B, N, // ldb = N
         0.0f, // beta (overwrite C)
         C, N  // ldc = N
    );
  }

  __attribute__((always_inline)) static inline void
  gemm_scaled(size_t M, size_t N, size_t K, float alpha, const float *A,
              const float *B, float beta, float *C) {

    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N,
         beta, C, N);
  }
};

template <> struct blasl3_traits<double> {
  using type = double;

  __attribute__((always_inline)) static inline void
  gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
       size_t M, size_t N, size_t K, double alpha, const double *A, size_t lda,
       const double *B, size_t ldb, double beta, double *C, size_t ldc) {

    cblas_dgemm(order, transA, transB, static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(K), alpha, A, static_cast<int>(lda), B,
                static_cast<int>(ldb), beta, C, static_cast<int>(ldc));
  }

  __attribute__((always_inline)) static inline void
  gemm_simple(size_t M, size_t N, size_t K, const double *A, const double *B,
              double *C) {

    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N,
         0.0, C, N);
  }

  __attribute__((always_inline)) static inline void
  gemm_scaled(size_t M, size_t N, size_t K, double alpha, const double *A,
              const double *B, double beta, double *C) {

    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N,
         beta, C, N);
  }
};

} // namespace hpc::acceler

#endif // ENABLE_ACCELERATE
