#pragma once

#include <algorithm>
#include <cstddef>

#define DEF_BLAS_L1_SEQUENTIAL_OP(name, op)                                    \
  template <typename T>                                                        \
  inline void name##_seq(T *__restrict__ dst, const T &scalar, size_t n) {     \
    for (size_t i = 0; i < n; ++i) {                                           \
      dst[i] op scalar;                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline void name##_seq(T *__restrict__ dst, const T *__restrict__ src,       \
                         size_t n) {                                           \
    for (size_t i = 0; i < n; ++i) {                                           \
      dst[i] op src[i];                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, const size_t TileSize>                                 \
  inline void name##_seq(T *__restrict__ dst, const T &scalar, size_t n) {     \
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {      \
      const size_t tile_end = std::min(tile_start + TileSize, n);              \
      for (size_t i = tile_start; i < tile_end; ++i) {                         \
        dst[i] op scalar;                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <typename T, const size_t TileSize>                                 \
  inline void name##_seq(T *__restrict__ dst, const T *__restrict__ src,       \
                         size_t n) {                                           \
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {      \
      const size_t tile_end = std::min(tile_start + TileSize, n);              \
      for (size_t i = tile_start; i < tile_end; ++i) {                         \
        dst[i] op src[i];                                                      \
      }                                                                        \
    }                                                                          \
  }

namespace hpc::l1 {
namespace details {

DEF_BLAS_L1_SEQUENTIAL_OP(vadd, +=)
DEF_BLAS_L1_SEQUENTIAL_OP(vsub, -=)
DEF_BLAS_L1_SEQUENTIAL_OP(vmul, *=)
DEF_BLAS_L1_SEQUENTIAL_OP(vdiv, /=)

// reduce
template <typename T>
inline void vsum_seq(T &result, const T *__restrict__ src, size_t n) {
  result = T(0);
  for (size_t i = 0; i < n; ++i) {
    result += src[i];
  }
}

template <typename T, const size_t TileSize>
inline void vsum_seq(T &result, const T *__restrict__ src, size_t n) {
  result = T(0);
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      result += src[i];
    }
  }
}

// fill
template <typename T>
inline void vfill_seq(T *__restrict__ dst, const T &value, size_t n) {
  memset(dst, value, n * sizeof(T));
}

template <typename T, const size_t TileSize>
inline void vfill_seq(T *__restrict__ dst, const T &value, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    memset(dst + tile_start, value, (tile_end - tile_start) * sizeof(T));
  }
}

// copy
template <typename T>
inline void vcopy_seq(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  memcpy(dst, src, n * sizeof(T));
}

template <typename T, const size_t TileSize>
inline void vcopy_seq(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    memcpy(dst + tile_start, src + tile_start,
           (tile_end - tile_start) * sizeof(T));
  }
}

// l1

// axpy
template <typename T>
inline void axpy_seq(T *__restrict__ y, const T *__restrict__ x, const T &a,
                     size_t n) {
  for (size_t i = 0; i < n; ++i) {
    y[i] += a * x[i];
  }
}

template <typename T, const size_t TileSize>
inline void axpy_seq(T *__restrict__ y, const T *__restrict__ x, const T &a,
                     size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      y[i] += a * x[i];
    }
  }
}

} // namespace details

} // namespace hpc::l1
