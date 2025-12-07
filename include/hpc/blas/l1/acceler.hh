#pragma once

#ifdef ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_ACCELERATE
#define ENABLE_ACCELERATE_VECTOR_SCALAR_BRANCH(name)                           \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, src, scalar);         \
  }
#define ENABLE_ACCELERATE_BINARY_BRANCH(name)                                  \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, src1, src2);          \
  }
#define ENABLE_ACCELERATE_UNARY_BRANCH(name)                                   \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, src);                 \
  }
#define ENABLE_ACCELERATE_SCALAR_BRANCH(name)                                  \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_acceler<T, BackendParams...>(n, dst, scalar);              \
  }
#define ENABLE_ACCELERATE_REDUCE_BRANCH(name)                                  \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    return details::name##_acceler<T, BackendParams...>(n, src);               \
  }
#define ENABLE_ACCELERATE_REDUCE2_BRANCH(name)                                 \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    return details::name##_acceler<T, BackendParams...>(n, x, y);              \
  }
#else
#define ENABLE_ACCELERATE_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_ACCELERATE_BINARY_BRANCH(name)
#define ENABLE_ACCELERATE_UNARY_BRANCH(name)
#define ENABLE_ACCELERATE_SCALAR_BRANCH(name)
#define ENABLE_ACCELERATE_REDUCE_BRANCH(name)
#define ENABLE_ACCELERATE_REDUCE2_BRANCH(name)
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_acceler(const size_t &n, T *__restrict__ dst,
                         const T *__restrict__ src, const T &alpha) {
  using traits = acceler::blasl1_traits<T>;
  traits::axpy(n, dst, src, alpha);
}

// copy
template <typename T>
inline void copy_acceler(const size_t &n, T *__restrict__ dst,
                         const T *__restrict__ src) {
  using traits = acceler::blasl1_traits<T>;
  traits::copy(n, dst, src);
}

// scal
template <typename T>
inline void scal_acceler(const size_t &n, T *__restrict__ dst, const T &alpha) {
  using traits = acceler::blasl1_traits<T>;
  traits::scal(n, dst, alpha);
}

} // namespace details
} // namespace hpc::l1

#endif
