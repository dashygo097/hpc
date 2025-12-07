#pragma once

#ifdef ENABLE_SIMD
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_SIMD
namespace hpc::l1 {
namespace details {

// axpy
template <typename T, const size_t SimdWidth>
inline void axpy_simd(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, alpha);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vy = SIMD_LOAD(traits, dst + i);
    simd_t vx = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, SIMD_FMA(traits, v_a, vx, vy));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] += alpha * src[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void axpy_simd(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, alpha);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t vy = SIMD_LOAD(traits, dst + i);
      simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, SIMD_FMA(traits, v_a, vx, vy));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] += alpha * src[i];
}

// copy
template <typename T, const size_t SimdWidth>
inline void copy_simd(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vs = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, vs);
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] = src[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void copy_simd(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, v_src);
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] = src[i];
}

} // namespace details
} // namespace hpc::l1
#endif
