#pragma once

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS) && defined(__CUDACC__)
#include "../../backends/backends.hh"
#endif

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS) && defined(__CUDACC__)
 #define ENABLE_CUBLAS_VECTOR_SCALAR_BRANCH(name) \
   else if constexpr (backend == Backend::CUBLAS) { \
     details::name##_cublas<T, BackendParams...>(n, dst, src, scalar); \
   }
 #define ENABLE_CUBLAS_BINARY_BRANCH(name) \
   else if constexpr (backend == Backend::CUBLAS) { \
     details::name##_cublas<T, BackendParams...>(n, dst, src1, src2); \
   }
 #define ENABLE_CUBLAS_UNARY_BRANCH(name) \
   else if constexpr (backend == Backend::CUBLAS) { \
     details::name##_cublas<T, BackendParams...>(n, dst, src); \
   }
 #define ENABLE_CUBLAS_SCALAR_BRANCH(name) \
   else if constexpr (backend == Backend::CUBLAS) { \
     details::name##_cublas<T, BackendParams...>(n, dst, scalar); \
   }
#define ENABLE_CUBLAS_REDUCE_BRANCH(name) \
   else if constexpr (backend == Backend::CUBLAS) { \
    return details::name##_cublas<T, BackendParams...>(n, src); \
  }
#define ENABLE_CUBLAS_REDUCE2_BRANCH(name) \
   else if constexpr (backend == Backend::CUBLAS) { \
     return details::name##_cublas<T, BackendParams...>(n, src1, src2); \
   }
#else
#define ENABLE_CUBLAS_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_CUBLAS_BINARY_BRANCH(name)
#define ENABLE_CUBLAS_UNARY_BRANCH(name)
#define ENABLE_CUBLAS_SCALAR_BRANCH(name)
#define ENABLE_CUBLAS_REDUCE_BRANCH(name)
#define ENABLE_CUBLAS_REDUCE2_BRANCH(name)
#endif

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS) && defined(__CUDACC__)
namespace hpc::l1 {
namespace details {
// axpy: dst = alpha * src + dst
template <typename T, typename... BackendParams>
inline void axpy_cublas(const size_t &n, T *__restrict__ dst,
                        const T *__restrict__ src, const T &alpha) {
  using traits = cublas::blasl1_traits<T>;
  traits::axpy(n, dst, src, alpha);
}

// copy: dst = src
template <typename T, typename... BackendParams>
inline void copy_cublas(const size_t &n, T *__restrict__ dst,
                        const T *__restrict__ src) {
  using traits = cublas::blasl1_traits<T>;
  traits::copy(n, dst, src);
}

// scal: dst = alpha * dst
template <typename T, typename... BackendParams>
inline void scal_cublas(const size_t &n, T *__restrict__ dst,
                        const T &alpha) {
  using traits = cublas::blasl1_traits<T>;
  traits::scal(n, dst, alpha);
}

// dot: return src1^T * src2
template <typename T, typename... BackendParams>
inline T dot_cublas(const size_t &n, const T *__restrict__ src1,
                    const T *__restrict__ src2) {
  using traits = cublas::blasl1_traits<T>;
  return traits::dot(n, src1, src2);
}

// swap: exchange dst and src
template <typename T, typename... BackendParams>
inline void swap_cublas(const size_t &n, T *__restrict__ dst,
                        T *__restrict__ src) {
  using traits = cublas::blasl1_traits<T>;
  traits::swap(n, dst, src);
}

// asum: return sum of absolute values
template <typename T, typename... BackendParams>
inline T asum_cublas(const size_t &n, const T *__restrict__ src) {
  using traits = cublas::blasl1_traits<T>;
  return traits::asum(n, src);
}

// nrm2: return Euclidean norm
template <typename T, typename... BackendParams>
inline T nrm2_cublas(const size_t &n, const T *__restrict__ src) {
  using traits = cublas::blasl1_traits<T>;
  return traits::nrm2(n, src);
}

// iamax: return index of element with maximum absolute value
template <typename T, typename... BackendParams>
inline size_t iamax_cublas(const size_t &n, const T *__restrict__ src) {
  using traits = cublas::blasl1_traits<T>;
  return traits::iamax(n, src);
}

}
} // namespace hpc::l1
#endif
