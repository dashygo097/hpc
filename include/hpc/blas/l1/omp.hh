#pragma once

#ifdef HPC_ENABLE_OPENMP
#include "../../backends/backends.hh"
#include <algorithm>
#include <cmath>
#include <cstring>
#endif

#ifdef HPC_ENABLE_OPENMP
#define HPC_ENABLE_OPENMP_VECTOR_SCALAR_BRANCH(name)                           \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src, scalar);             \
  }
#define HPC_ENABLE_OPENMP_BINARY_BRANCH(name)                                  \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src1, src2);              \
  }
#define HPC_ENABLE_OPENMP_UNARY_BRANCH(name)                                   \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src);                     \
  }
#define HPC_ENABLE_OPENMP_SCALAR_BRANCH(name)                                  \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, scalar);                  \
  }
#define HPC_ENABLE_OPENMP_REDUCE_BRANCH(name)                                  \
  else if constexpr (backend == Backend::OPENMP) {                             \
    return details::name##_omp<T, BackendParams...>(n, src);                   \
  }
#define HPC_ENABLE_OPENMP_REDUCE2_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP) {                             \
    return details::name##_omp<T, BackendParams...>(n, src1, src2);            \
  }
#else
#define HPC_ENABLE_OPENMP_VECTOR_SCALAR_BRANCH(name)
#define HPC_ENABLE_OPENMP_BINARY_BRANCH(name)
#define HPC_ENABLE_OPENMP_UNARY_BRANCH(name)
#define HPC_ENABLE_OPENMP_SCALAR_BRANCH(name)
#define HPC_ENABLE_OPENMP_REDUCE_BRANCH(name)
#define HPC_ENABLE_OPENMP_REDUCE2_BRANCH(name)
#endif

#ifdef HPC_ENABLE_OPENMP
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src, const T &alpha) {
  if (alpha == T{0}) {
    return;
  }

  if (alpha == T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      dst[i] += src[i];
    }
    return;
  }

  if (alpha == T{-1}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      dst[i] -= src[i];
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] += alpha * src[i];
  }
}

template <typename T, const size_t TileSize>
inline void axpy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src, const T &alpha) {
  if (alpha == T{0}) {
    return;
  }

  if (alpha == T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] += src[i];
      }
    }
    return;
  }

  if (alpha == T{-1}) {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] -= src[i];
      }
    }
    return;
  }

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] += alpha * src[i];
    }
  }
}

// copy
template <typename T>
inline void copy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src) {
  memcpy(dst, src, n * sizeof(T));
}

template <typename T, const size_t TileSize>
inline void copy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src) {
#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    memcpy(dst + tile_start, src + tile_start,
           (tile_end - tile_start) * sizeof(T));
  }
}

// scal
template <typename T>
inline void scal_omp(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{1}) {
    return;
  }

  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
    return;
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] *= alpha;
  }
}
template <typename T, const size_t TileSize>
inline void scal_omp(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{1}) {
    return;
  }

  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
  }

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] *= alpha;
    }
  }
}

// dot
template <typename T>
inline T dot_omp(const size_t &n, const T *__restrict__ src1,
                 const T *__restrict__ src2) {
  T result = T{0};

#pragma omp parallel
  {
    T local_sum = T{0};
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      local_sum += src1[i] * src2[i];
    }
#pragma omp critical
    {
      result += local_sum;
    }
  }

  return result;
}

template <typename T, const size_t TileSize>
inline T dot_omp(const size_t &n, const T *__restrict__ src1,
                 const T *__restrict__ src2) {
  T result = T{0};

#pragma omp parallel
  {
    T local_sum = T{0};
#pragma omp for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        local_sum += src1[i] * src2[i];
      }
    }
#pragma omp critical
    {
      result += local_sum;
    }
  }

  return result;
}

template <typename T>
inline void swap_omp(const size_t &n, T *__restrict__ src1,
                     T *__restrict__ src2) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    T tmp = src1[i];
    src1[i] = src2[i];
    src2[i] = tmp;
  }
}

template <typename T, const size_t TileSize>
inline void swap_omp(const size_t &n, T *__restrict__ src1,
                     T *__restrict__ src2) {
#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      T tmp = src1[i];
      src1[i] = src2[i];
      src2[i] = tmp;
    }
  }
}

template <typename T>
inline T asum_omp(const size_t &n, const T *__restrict__ src) {
  T sum = T{0};
#pragma omp parallel for schedule(static) reduction(+ : sum)
  for (size_t i = 0; i < n; ++i) {
    sum += std::abs(src[i]);
  }
  return sum;
}

template <typename T, const size_t TileSize>
inline T asum_omp(const size_t &n, const T *__restrict__ src) {
  T sum = T{0};
#pragma omp parallel for schedule(static) reduction(+ : sum)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      sum += std::abs(src[i]);
    }
  }
  return sum;
}

template <typename T>
inline T nrm2_omp(const size_t &n, const T *__restrict__ src) {
  T sum = T{0};
#pragma omp parallel for schedule(static) reduction(+ : sum)
  for (size_t i = 0; i < n; ++i) {
    sum += src[i] * src[i];
  }
  return std::sqrt(sum);
}

template <typename T, const size_t TileSize>
inline T nrm2_omp(const size_t &n, const T *__restrict__ src) {
  T sum = T{0};
#pragma omp parallel for schedule(static) reduction(+ : sum)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      sum += src[i] * src[i];
    }
  }
  return std::sqrt(sum);
}

template <typename T>
inline size_t iamax_omp(const size_t &n, const T *__restrict__ src) {
  if (n == 0)
    return 0;
  size_t global_idx = 0;
  T global_max = std::abs(src[0]);
#pragma omp parallel
  {
    size_t local_idx = 0;
    T local_max = T{0};
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      T val = std::abs(src[i]);
      if (val > local_max) {
        local_max = val;
        local_idx = i;
      }
    }
#pragma omp critical
    {
      if (local_max > global_max) {
        global_max = local_max;
        global_idx = local_idx;
      }
    }
  }
  return global_idx;
}

template <typename T, const size_t TileSize>
inline size_t iamax_omp(const size_t &n, const T *__restrict__ src) {
  if (n == 0)
    return 0;
  size_t global_idx = 0;
  T global_max = std::abs(src[0]);
#pragma omp parallel
  {
    size_t local_idx = 0;
    T local_max = T{0};
#pragma omp for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        T val = std::abs(src[i]);
        if (val > local_max) {
          local_max = val;
          local_idx = i;
        }
      }
    }
#pragma omp critical
    {
      if (local_max > global_max) {
        global_max = local_max;
        global_idx = local_idx;
      }
    }
  }
  return global_idx;
}

} // namespace details
} // namespace hpc::l1
#endif
