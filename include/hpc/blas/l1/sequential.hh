#pragma once

#include <algorithm>
#include <cstddef>

namespace hpc::l1 {
namespace details {

// addition
template <typename T>
inline void vadd_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] += scalar;
  }
}

template <typename T>
inline void vadd_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] += src[i];
  }
}

template <typename T, const size_t TileSize>
inline void vadd_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] += scalar;
    }
  }
}

template <typename T, const size_t TileSize>
inline void vadd_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] += src[i];
    }
  }
}

// subtraction
template <typename T>
inline void vsub_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] -= scalar;
  }
}

template <typename T>
inline void vsub_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] -= src[i];
  }
}

template <typename T, const size_t TileSize>
inline void vsub_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] -= scalar;
    }
  }
}

template <typename T, const size_t TileSize>
inline void vsub_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] -= src[i];
    }
  }
}

// multiplication
template <typename T>
inline void vmul_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] *= scalar;
  }
}

template <typename T>
inline void vmul_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] *= src[i];
  }
}

template <typename T, const size_t TileSize>
inline void vmul_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] *= scalar;
    }
  }
}

template <typename T, const size_t TileSize>
inline void vmul_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] *= src[i];
    }
  }
}

// division
template <typename T>
inline void vdiv_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] /= scalar;
  }
}

template <typename T>
inline void vdiv_seq(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] /= src[i];
  }
}

template <typename T, const size_t TileSize>
inline void vdiv_seq(T *__restrict__ dst, const T &scalar, size_t n) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] /= scalar;
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
