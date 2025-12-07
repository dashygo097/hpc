#pragma once

#ifdef ENABLE_OPENMP
#include "../../backends/backends.hh"
#include <algorithm>
#endif

#ifdef ENABLE_OPENMP
namespace hpc::l1 {
namespace details {

// axpy
template <typename T>
inline void axpy_omp(const size_t &n, T *__restrict__ y,
                     const T *__restrict__ x, const T &alpha) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

template <typename T, const size_t TileSize>
inline void axpy_omp(const size_t &n, T *__restrict__ y,
                     const T *__restrict__ x, const T &alpha) {
#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      y[i] += alpha * x[i];
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
    size_t tile_end = std::min(tile_start + TileSize, n);
    for (size_t i = tile_start; i < tile_end; ++i) {
      dst[i] = src[i];
    }
  }
}

} // namespace details
} // namespace hpc::l1
#endif
