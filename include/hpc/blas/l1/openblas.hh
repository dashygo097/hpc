#pragma once

#ifdef ENABLE_OPENBLAS
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_OPENBLAS
#define ENABLE_OPENBLAS_VECTOR_SCALAR_BRANCH(name)                             \
  else if constexpr (backend == Backend::OPENBLAS) {                           \
    details::name##_openblas<T, BackendParams...>(n, dst, src, scalar);        \
  }
#define ENABLE_OPENBLAS_BINARY_BRANCH(name)                                    \
  else if constexpr (backend == Backend::OPENBLAS) {                           \
    details::name##_openblas<T, BackendParams...>(n, dst, src1, src2);         \
  }
#define ENABLE_OPENBLAS_UNARY_BRANCH(name)                                     \
  else if constexpr (backend == Backend::OPENBLAS) {                           \
    details::name##_openblas<T, BackendParams...>(n, dst, src);                \
  }
#define ENABLE_OPENBLAS_SCALAR_BRANCH(name)                                    \
  else if constexpr (backend == Backend::OPENBLAS) {                           \
    details::name##_openblas<T, BackendParams...>(n, dst, scalar);             \
  }
#define ENABLE_OPENBLAS_REDUCE_BRANCH(name)                                    \
  else if constexpr (backend == Backend::OPENBLAS) {                           \
    return details::name##_openblas<T, BackendParams...>(n, src);              \
  }
#define ENABLE_OPENBLAS_REDUCE2_BRANCH(name)                                   \
  else if constexpr (backend == Backend::OPENBLAS) {                           \
    return details::name##_openblas<T, BackendParams...>(n, src1, src2);       \
  }
#else
#define ENABLE_OPENBLAS_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_OPENBLAS_BINARY_BRANCH(name)
#define ENABLE_OPENBLAS_UNARY_BRANCH(name)
#define ENABLE_OPENBLAS_SCALAR_BRANCH(name)
#define ENABLE_OPENBLAS_REDUCE_BRANCH(name)
#define ENABLE_OPENBLAS_REDUCE2_BRANCH(name)
#endif

#ifdef ENABLE_OPENBLAS
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_openblas(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src, const T &alpha) {
  using traits = openblas::blasl1_traits<T>;
  traits::axpy(n, dst, src, alpha);
}

// copy
template <typename T>
inline void copy_openblas(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src) {
  using traits = openblas::blasl1_traits<T>;
  traits::copy(n, dst, src);
}

// scal
template <typename T>
inline void scal_openblas(const size_t &n, T *__restrict__ dst,
                          const T &alpha) {
  using traits = openblas::blasl1_traits<T>;
  traits::scal(n, dst, alpha);
}

// dot
template <typename T>
inline void dot_openblas(const size_t &n, const T *__restrict__ src1,
                         const T *__restrict__ src2) {
  using traits = openblas::blasl1_traits<T>;
  return traits::dot(n, src1, src2);
}

} // namespace details
} // namespace hpc::l1

#endif
