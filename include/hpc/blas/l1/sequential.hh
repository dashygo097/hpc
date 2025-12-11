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
  }

  if (alpha == T{1}) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] += src[i];
    }
    return;
  }

  if (alpha == T{-1}) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] -= src[i];
    }
    return;
  }

  for (size_t i = 0; i < n; ++i) {
    dst[i] += alpha * src[i];
  }
}

template <typename T, const size_t TileSize>
inline void axpy_seq(const size_t &n, T *__restrict__ dst,
                     const T *__restrict__ src, const T &alpha) {
  if (alpha == T{0}) {
    return;
  }

  if (alpha == T{1}) {
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] += src[i];
      }
    }
    return;
  }

  if (alpha == T{-1}) {
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      for (size_t i = tile_start; i < tile_end; ++i) {
        dst[i] -= src[i];
      }
    }
    return;
  }

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] += alpha * src[i];
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
  if (alpha == T{1}) {
    return;
  }

  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
    return;
  }

  for (size_t i = 0; i < n; ++i) {
    dst[i] *= alpha;
  }
}

template <typename T, const size_t TileSize>
inline void scal_seq(const size_t &n, T *__restrict__ dst, const T &alpha) {
  if (alpha == T{1}) {
    return;
  }

  if (alpha == T{0}) {
    for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
      const size_t tile_end = std::min(tile_start + TileSize, n);
      memset(dst + tile_start, 0, (tile_end - tile_start) * sizeof(T));
    }
    return;
  }

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] *= alpha;
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

// swap
template <typename T>
inline void swap_seq(const size_t &n, T *__restrict__ src1,
                     T *__restrict__ src2) {
  for (size_t i = 0; i < n; ++i) {
    std::swap(src1[i], src2[i]);
  }
}

template <typename T, const size_t TileSize>
inline void swap_seq(const size_t &n, T *__restrict__ src1,
                     T *__restrict__ src2) {
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      std::swap(src1[i], src2[i]);
    }
  }
}

// asum
template <typename T>
inline T asum_seq(const size_t &n, const T *__restrict__ src) {
  T result = T{0};

  for (size_t i = 0; i < n; ++i) {
    result += std::abs(src[i]);
  }
  return result;
}

template <typename T, const size_t TileSize>
inline T asum_seq(const size_t &n, const T *__restrict__ src) {
  T result = T{0};

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      result += std::abs(src[i]);
    }
  }
  return result;
}

// nrm2
template <typename T>
inline T nrm2_seq(const size_t &n, const T *__restrict__ src) {
  T result = T{0};

  for (size_t i = 0; i < n; ++i) {
    result += src[i] * src[i];
  }
  return std::sqrt(result);
}

template <typename T, const size_t TileSize>
inline void nrm2_seq(const size_t &n, const T *__restrict__ src) {
  T result = T{0};

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      result += src[i] * src[i];
    }
  }
  return std::sqrt(result);
}

// iamax
template <typename T>
inline size_t iamax_seq(const size_t &n, const T *__restrict__ src) {
  size_t max_index = 0;
  T max_value = std::abs(src[0]);

  for (size_t i = 1; i < n; ++i) {
    T abs_value = std::abs(src[i]);
    if (abs_value > max_value) {
      max_value = abs_value;
      max_index = i;
    }
  }
  return max_index;
}

template <typename T, const size_t TileSize>
inline size_t iamax_seq(const size_t &n, const T *__restrict__ src) {
  size_t max_index = 0;
  T max_value = std::abs(src[0]);

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      T abs_value = std::abs(src[i]);
      if (abs_value > max_value) {
        max_value = abs_value;
        max_index = i;
      }
    }
  }
  return max_index;
}

} // namespace details

} // namespace hpc::l1
