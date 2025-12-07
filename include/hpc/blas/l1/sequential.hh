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
  for (size_t i = 0; i < n; ++i) {
    dst[i] = src[i];
  }
}

template <typename T, const size_t TileSize>
inline void copy_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] = src[i];
    }
  }
}

// scal
template <typename T>
inline void scal_seq(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{0}) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = T{0};
    }
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
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] = T{0};
      }
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

} // namespace details

} // namespace hpc::l1
