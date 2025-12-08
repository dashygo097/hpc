#pragma once

#ifdef ENABLE_OPENMP
#include "../../backends/backends.hh"
#include <algorithm>
#endif

#ifdef ENABLE_OPENMP
#define ENABLE_OPENMP_VECTOR_SCALAR_BRANCH(name)                               \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src, scalar);             \
  }
#define ENABLE_OPENMP_BINARY_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src1, src2);              \
  }
#define ENABLE_OPENMP_UNARY_BRANCH(name)                                       \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, src);                     \
  }
#define ENABLE_OPENMP_SCALAR_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    details::name##_omp<T, BackendParams...>(n, dst, scalar);                  \
  }
#define ENABLE_OPENMP_REDUCE_BRANCH(name)                                      \
  else if constexpr (backend == Backend::OPENMP) {                             \
    return details::name##_omp<T, BackendParams...>(n, src);                   \
  }
#define ENABLE_OPENMP_REDUCE2_BRANCH(name)                                     \
  else if constexpr (backend == Backend::OPENMP) {                             \
    return details::name##_omp<T, BackendParams...>(n, x, y);                  \
  }
#else
#define ENABLE_OPENMP_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_BINARY_BRANCH(name)
#define ENABLE_OPENMP_UNARY_BRANCH(name)
#define ENABLE_OPENMP_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_REDUCE_BRANCH(name)
#define ENABLE_OPENMP_REDUCE2_BRANCH(name)
#endif

#ifdef ENABLE_OPENMP
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src, const T &alpha) {
  if (alpha == T{0}) {
    return;
  } else if (alpha == T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      dst[i] += src[i];
    }
  } else {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      dst[i] += alpha * src[i];
    }
  }
}

template <typename T, const size_t TileSize>
inline void axpy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src, const T &alpha) {
  if (alpha == T{0}) {
    return;
  } else if (alpha == T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] += src[i];
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] += alpha * src[i];
      }
    }
  }
}

// copy
template <typename T>
inline void copy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] = src[i];
  }
}

template <typename T, const size_t TileSize>
inline void copy_omp(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src) {
#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] = src[i];
    }
  }
}

// scal
template <typename T>
inline void scal_omp(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{0}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      dst[i] = T{0};
    }
  } else if (alpha == T{1}) {
    return;
  } else {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      dst[i] *= alpha;
    }
  }
}
template <typename T, const size_t TileSize>
inline void scal_omp(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{0}) {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] = T{0};
      }
    }
  } else if (alpha == T{1}) {
    return;
  } else {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] *= alpha;
      }
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
inline void dot_omp(const size_t &n, const T *__restrict__ src1,
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

} // namespace details
} // namespace hpc::l1
#endif
