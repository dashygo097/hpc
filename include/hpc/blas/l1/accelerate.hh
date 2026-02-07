#pragma once

#ifdef HPC_ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef HPC_ENABLE_ACCELERATE
#define HPC_ENABLE_ACCELERATE_VECTOR_SCALAR_BRANCH(name)                       \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_accelerate<T, BackendParams...>(n, dst, src, scalar);      \
  }
#define HPC_ENABLE_ACCELERATE_BINARY_BRANCH(name)                              \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_accelerate<T, BackendParams...>(n, dst, src1, src2);       \
  }
#define HPC_ENABLE_ACCELERATE_UNARY_BRANCH(name)                               \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_accelerate<T, BackendParams...>(n, dst, src);              \
  }
#define HPC_ENABLE_ACCELERATE_SCALAR_BRANCH(name)                              \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    details::name##_accelerate<T, BackendParams...>(n, dst, scalar);           \
  }
#define HPC_ENABLE_ACCELERATE_REDUCE_BRANCH(name)                              \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    return details::name##_accelerate<T, BackendParams...>(n, src);            \
  }
#define HPC_ENABLE_ACCELERATE_REDUCE2_BRANCH(name)                             \
  else if constexpr (backend == Backend::ACCELERATE) {                         \
    return details::name##_accelerate<T, BackendParams...>(n, src1, src2);     \
  }
#else
#define HPC_ENABLE_ACCELERATE_VECTOR_SCALAR_BRANCH(name)
#define HPC_ENABLE_ACCELERATE_BINARY_BRANCH(name)
#define HPC_ENABLE_ACCELERATE_UNARY_BRANCH(name)
#define HPC_ENABLE_ACCELERATE_SCALAR_BRANCH(name)
#define HPC_ENABLE_ACCELERATE_REDUCE_BRANCH(name)
#define HPC_ENABLE_ACCELERATE_REDUCE2_BRANCH(name)
#endif

#ifdef HPC_ENABLE_ACCELERATE
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_accelerate(const size_t &n, T *__restrict__ dst,
                            const T *__restrict__ src, const T &alpha) {
  using traits = accelerate::blasl1_traits<T>;
  traits::axpy(n, dst, src, alpha);
}

// copy
template <typename T>
inline void copy_accelerate(const size_t &n, T *__restrict__ dst,
                            const T *__restrict__ src) {
  using traits = accelerate::blasl1_traits<T>;
  traits::copy(n, dst, src);
}

// scal
template <typename T>
inline void scal_accelerate(const size_t &n, T *__restrict__ dst,
                            const T &alpha) {
  using traits = accelerate::blasl1_traits<T>;
  traits::scal(n, dst, alpha);
}

// dot
template <typename T>
inline T dot_accelerate(const size_t &n, const T *__restrict__ src1,
                        const T *__restrict__ src2) {
  using traits = accelerate::blasl1_traits<T>;
  return traits::dot(n, src1, src2);
}

// swap
template <typename T>
inline void swap_accelerate(const size_t &n, T *__restrict__ src1,
                            T *__restrict__ src2) {
  using traits = accelerate::blasl1_traits<T>;
  traits::swap(n, src1, src2);
}

// asum
template <typename T>
inline void asum_accelerate(const size_t &n, const T *__restrict__ src) {
  using traits = accelerate::blasl1_traits<T>;
  return traits::asum(n, src);
}

// nrm2
template <typename T>
inline void nrm2_accelerate(const size_t &n, const T *__restrict__ src) {
  using traits = accelerate::blasl1_traits<T>;
  return traits::nrm2(n, src);
}

// iamax
template <typename T>
inline void iamax_accelerate(const size_t &n, const T *__restrict__ src) {
  using traits = accelerate::blasl1_traits<T>;
  return traits::iamax(n, src);
}

} // namespace details
} // namespace hpc::l1

#endif
