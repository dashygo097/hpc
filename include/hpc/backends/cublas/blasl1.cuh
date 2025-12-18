#pragma once

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS)
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#endif

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__);          \
      fprintf(stderr, "  %s: status code %d\n", #call, status);                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace hpc::cublas {

inline cublasHandle_t get_handle() {
  static thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }
  return handle;
}

template <typename T> struct blasl1_traits;

template <> struct blasl1_traits<float> {
  // axpy: y = alpha * x + y
  __attribute__((always_inline)) static inline void
  axpy(const size_t &n, float *y, const float *x, const float &alpha) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasSaxpy(handle, static_cast<int>(n), &alpha, x, 1, y, 1));
  }

  // copy: dst = src
  __attribute__((always_inline)) static inline void
  copy(const size_t &n, float *dst, const float *src) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasScopy(handle, static_cast<int>(n), src, 1, dst, 1));
  }

  // scal: x = alpha * x
  __attribute__((always_inline)) static inline void
  scal(const size_t &n, float *x, const float &alpha) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasSscal(handle, static_cast<int>(n), &alpha, x, 1));
  }

  // dot: return x^T * y
  __attribute__((always_inline)) static inline float
  dot(const size_t &n, const float *x, const float *y) {
    cublasHandle_t handle = get_handle();
    float result;
    CUBLAS_CHECK(cublasSdot(handle, static_cast<int>(n), x, 1, y, 1, &result));
    return result;
  }

  // swap: exchange x and y
  __attribute__((always_inline)) static inline void swap(const size_t &n,
                                                         float *x, float *y) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasSswap(handle, static_cast<int>(n), x, 1, y, 1));
  }

  // asum: return sum of absolute values
  __attribute__((always_inline)) static inline float asum(const size_t &n,
                                                          const float *x) {
    cublasHandle_t handle = get_handle();
    float result;
    CUBLAS_CHECK(cublasSasum(handle, static_cast<int>(n), x, 1, &result));
    return result;
  }

  // nrm2: return Euclidean norm
  __attribute__((always_inline)) static inline float nrm2(const size_t &n,
                                                          const float *x) {
    cublasHandle_t handle = get_handle();
    float result;
    CUBLAS_CHECK(cublasSnrm2(handle, static_cast<int>(n), x, 1, &result));
    return result;
  }

  // iamax: return index of element with maximum absolute value
  __attribute__((always_inline)) static inline size_t iamax(const size_t &n,
                                                            const float *x) {
    cublasHandle_t handle = get_handle();
    int result;
    CUBLAS_CHECK(cublasIsamax(handle, static_cast<int>(n), x, 1, &result));
    // cuBLAS returns 1-based index, convert to 0-based
    return static_cast<size_t>(result - 1);
  }
};

template <> struct blasl1_traits<double> {
  // axpy: y = alpha * x + y
  __attribute__((always_inline)) static inline void
  axpy(const size_t &n, double *y, const double *x, const double &alpha) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasDaxpy(handle, static_cast<int>(n), &alpha, x, 1, y, 1));
  }

  // copy: dst = src
  __attribute__((always_inline)) static inline void
  copy(const size_t &n, double *dst, const double *src) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasDcopy(handle, static_cast<int>(n), src, 1, dst, 1));
  }

  // scal: x = alpha * x
  __attribute__((always_inline)) static inline void
  scal(const size_t &n, double *x, const double &alpha) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasDscal(handle, static_cast<int>(n), &alpha, x, 1));
  }

  // dot: return x^T * y
  __attribute__((always_inline)) static inline double
  dot(const size_t &n, const double *x, const double *y) {
    cublasHandle_t handle = get_handle();
    double result;
    CUBLAS_CHECK(cublasDdot(handle, static_cast<int>(n), x, 1, y, 1, &result));
    return result;
  }

  // swap: exchange x and y
  __attribute__((always_inline)) static inline void swap(const size_t &n,
                                                         double *x, double *y) {
    cublasHandle_t handle = get_handle();
    CUBLAS_CHECK(cublasDswap(handle, static_cast<int>(n), x, 1, y, 1));
  }

  // asum: return sum of absolute values
  __attribute__((always_inline)) static inline double asum(const size_t &n,
                                                           const double *x) {
    cublasHandle_t handle = get_handle();
    double result;
    CUBLAS_CHECK(cublasDasum(handle, static_cast<int>(n), x, 1, &result));
    return result;
  }

  // nrm2: return Euclidean norm
  __attribute__((always_inline)) static inline double nrm2(const size_t &n,
                                                           const double *x) {
    cublasHandle_t handle = get_handle();
    double result;
    CUBLAS_CHECK(cublasDnrm2(handle, static_cast<int>(n), x, 1, &result));
    return result;
  }

  // iamax: return index of element with maximum absolute value
  __attribute__((always_inline)) static inline size_t iamax(const size_t &n,
                                                            const double *x) {
    cublasHandle_t handle = get_handle();
    int result;
    CUBLAS_CHECK(cublasIdamax(handle, static_cast<int>(n), x, 1, &result));
    // cuBLAS returns 1-based index, convert to 0-based
    return static_cast<size_t>(result - 1);
  }
};

} // namespace hpc::cublas
#endif
