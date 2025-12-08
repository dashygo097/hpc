#pragma once

#include <algorithm>
#include <cstddef>

namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_seq(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src, const T &alpha) {
  if (alpha == T{0}) {
    return;
  } else if (alpha == T{1}) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] += src[i];
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      dst[i] += alpha * src[i];
    }
  }
}

template <typename T, const size_t TileSize>
inline void axpy_seq(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src, const T &alpha) {
  if (alpha == T{0}) {
    return;
  } else if (alpha == T{1}) {
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] += src[i];
      }
    }
  } else {
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
inline void copy_seq(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src) {
  memcpy(dst, src, n * sizeof(T));
}

template <typename T, const size_t TileSize>
inline void copy_seq(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    memcpy(dst + tile_start, src + tile_start,
           (tile_end - tile_start) * sizeof(T));
  }
}

// scal
template <typename T>
inline void scal_seq(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
  } else if (alpha == T{1}) {
    return;
  } else {
    for (size_t i = 0; i < n; ++i) {
      dst[i] *= alpha;
    }
  }
}

template <typename T, const size_t TileSize>
inline void scal_seq(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{0}) {
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      memset(dst + tile_start, 0, (tile_end - tile_start) * sizeof(T));
    }
  } else if (alpha == T{1}) {
    return;
  } else {
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
inline T dot_seq(const size_t &n, const T *__restrict__ src1,
                 const T *__restrict__ src2) {
  T result = T{0};
  for (size_t i = 0; i < n; ++i) {
    result += src1[i] * src2[i];
  }
  return result;
}

template <typename T, const size_t TileSize>
inline T dot_seq(const size_t &n, const T *__restrict__ src1,
                 const T *__restrict__ src2) {
  T result = T{0};
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      result += src1[i] * src2[i];
    }
  }
  return result;
}

} // namespace details

} // namespace hpc::l1
