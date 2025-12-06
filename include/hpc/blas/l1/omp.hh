#pragma once

#ifdef ENABLE_OPENMP
#include "../../backends/backends.hh"
#include <algorithm>
#endif

#define DEF_BLAS_L1_OMP_OP(name, op)                                           \
  template <typename T>                                                        \
  inline void name##_omp(T *__restrict__ dst, const T &scalar, size_t n) {     \
    _Pragma("omp parallel for schedule(static)") for (size_t i = 0; i < n;     \
                                                      ++i) {                   \
      dst[i] op scalar;                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline void name##_omp(T *__restrict__ dst, const T *__restrict__ src,       \
                         size_t n) {                                           \
    _Pragma("omp parallel for schedule(static)") for (size_t i = 0; i < n;     \
                                                      ++i) {                   \
      dst[i] op src[i];                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, const size_t TileSize>                                 \
  inline void name##_omp(T *__restrict__ dst, const T &scalar, size_t n) {     \
    _Pragma("omp parallel for schedule(static)") for (size_t tile_start = 0;   \
                                                      tile_start < n;          \
                                                      tile_start +=            \
                                                      TileSize) {              \
      size_t tile_end = std::min(tile_start + TileSize, n);                    \
      for (size_t i = tile_start; i < tile_end; ++i) {                         \
        dst[i] op scalar;                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, const size_t TileSize>                                 \
  inline void name##_omp(T *__restrict__ dst, const T *__restrict__ src,       \
                         size_t n) {                                           \
    _Pragma("omp parallel for schedule(static)") for (size_t tile_start = 0;   \
                                                      tile_start < n;          \
                                                      tile_start +=            \
                                                      TileSize) {              \
      size_t tile_end = std::min(tile_start + TileSize, n);                    \
      for (size_t i = tile_start; i < tile_end; ++i) {                         \
        dst[i] op src[i];                                                      \
      }                                                                        \
    }                                                                          \
  }

#ifdef ENABLE_OPENMP
namespace hpc::l1 {
namespace details {

DEF_BLAS_L1_OMP_OP(vadd, +=)
DEF_BLAS_L1_OMP_OP(vsub, -=)
DEF_BLAS_L1_OMP_OP(vmul, *=)
DEF_BLAS_L1_OMP_OP(vdiv, /=)

// reduce
template <typename T>
inline void vsum_omp(const T *__restrict__ src, T &result, size_t n) {
  T sum = T{};
#pragma omp parallel
  {
    T private_sum = T{};
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      private_sum += src[i];
    }
#pragma omp critical
    {
      sum += private_sum;
    }
  }
  result = sum;
}

template <typename T, const size_t TileSize>
inline void vsum_omp(const T *__restrict__ src, T &result, size_t n) {
  T sum = T{};
#pragma omp parallel
  {
    T private_sum = T{};
#pragma omp for schedule(static)
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        private_sum += src[i];
      }
    }
#pragma omp critical
    {
      sum += private_sum;
    }
  }
  result = sum;
}

// fill
template <typename T>
inline void vfill_omp(T *__restrict__ dst, const T &value, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] = value;
}

template <typename T, const size_t TileSize>
inline void vfill_omp(T *__restrict__ dst, const T &value, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = std::min(tile_start + TileSize, n);
    memset(dst + tile_start, value, (tile_end - tile_start) * sizeof(T));
  }
}

// copy
template <typename T>
inline void vcopy_omp(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] = src[i];
  }
}

template <typename T, const size_t TileSize>
inline void vcopy_omp(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] = src[i];
    }
  }
}

// l1

// axpy
template <typename T>
inline void axpy_omp(T *__restrict__ y, const T *__restrict__ x, const T &a,
                     size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    y[i] += a * x[i];
  }
}

template <typename T, const size_t TileSize>
inline void axpy_omp(T *__restrict__ y, const T *__restrict__ x, const T &a,
                     size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      y[i] += a * x[i];
    }
  }
}

} // namespace details
} // namespace hpc::l1
#endif
