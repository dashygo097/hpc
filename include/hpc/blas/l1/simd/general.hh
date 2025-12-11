#pragma once

#ifdef ENABLE_SIMD
#include "../../../backends/backends.hh"
#include <cstring>
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

  if (alpha == T{0}) {
    return;
  }
  if (alpha == T{1}) {
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      simd_t vy = SIMD_LOAD(traits, dst + i);
      simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, SIMD_ADD(traits, vx, vy));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += src[i];
    return;
  }

  if (alpha == T{-1}) {
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      simd_t vy = SIMD_LOAD(traits, dst + i);
      simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, SIMD_SUB(traits, vy, vx));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] -= src[i];
    return;
  }

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

  if (alpha == T{0}) {
    return;
  }

  if (alpha == T{1}) {
    for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
      size_t tile_end = tile_start + TileSize;
      for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
        simd_t vy = SIMD_LOAD(traits, dst + i);
        simd_t vx = SIMD_LOAD(traits, src + i);
        SIMD_STORE(traits, dst + i, SIMD_ADD(traits, vx, vy));
      }
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += src[i];
    return;
  }

  if (alpha == T{-1}) {
    for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
      size_t tile_end = tile_start + TileSize;
      for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
        simd_t vy = SIMD_LOAD(traits, dst + i);
        simd_t vx = SIMD_LOAD(traits, src + i);
        SIMD_STORE(traits, dst + i, SIMD_SUB(traits, vy, vx));
      }
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] -= src[i];
    return;
  }

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

  memcpy(dst, src, n * sizeof(T));
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void copy_simd(const size_t &n, T *__restrict__ dst,
                      const T *__restrict__ src) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = tile_start + TileSize;
    memcpy(dst + tile_start, src + tile_start,
           (tile_end - tile_start) * sizeof(T));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] = src[i];
}

// scal
template <typename T, const size_t SimdWidth>
inline void scal_simd(const size_t &n, T *__restrict__ dst, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_zero = SIMD_DUP(traits, T{0});
  const simd_t v_a = SIMD_DUP(traits, alpha);

  if (alpha == T{1}) {
    return;
  }

  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
    return;
  }

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vx = SIMD_LOAD(traits, dst + i);
    SIMD_STORE(traits, dst + i, SIMD_MUL(traits, v_a, vx));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= alpha;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void scal_simd(const size_t &n, T *__restrict__ dst, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, alpha);
  const simd_t v_zero = SIMD_DUP(traits, T{0});

  if (alpha == T{1}) {
    return;
  }

  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
    return;
  }

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t vx = SIMD_LOAD(traits, dst + i);
      SIMD_STORE(traits, dst + i, SIMD_MUL(traits, v_a, vx));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= alpha;
}

// dot
template <typename T, const size_t SimdWidth>
inline T dot_simd(const size_t &n, const T *__restrict__ src1,
                  const T *__restrict__ src2) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

  const size_t simd_end = n - (n % SimdWidth);

  simd_t v_sum0 = SIMD_DUP(traits, T{0});
  simd_t v_sum1 = SIMD_DUP(traits, T{0});
  simd_t v_sum2 = SIMD_DUP(traits, T{0});
  simd_t v_sum3 = SIMD_DUP(traits, T{0});

  size_t i = 0;
  const size_t simd_end4 = simd_end - (simd_end % (SimdWidth * 4));

  for (; i < simd_end4; i += SimdWidth * 4) {
    simd_t vx0 = SIMD_LOAD(traits, src1 + i + SimdWidth * 0);
    simd_t vy0 = SIMD_LOAD(traits, src2 + i + SimdWidth * 0);
    v_sum0 = SIMD_FMA(traits, vx0, vy0, v_sum0);

    simd_t vx1 = SIMD_LOAD(traits, src1 + i + SimdWidth * 1);
    simd_t vy1 = SIMD_LOAD(traits, src2 + i + SimdWidth * 1);
    v_sum1 = SIMD_FMA(traits, vx1, vy1, v_sum1);

    simd_t vx2 = SIMD_LOAD(traits, src1 + i + SimdWidth * 2);
    simd_t vy2 = SIMD_LOAD(traits, src2 + i + SimdWidth * 2);
    v_sum2 = SIMD_FMA(traits, vx2, vy2, v_sum2);

    simd_t vx3 = SIMD_LOAD(traits, src1 + i + SimdWidth * 3);
    simd_t vy3 = SIMD_LOAD(traits, src2 + i + SimdWidth * 3);
    v_sum3 = SIMD_FMA(traits, vx3, vy3, v_sum3);
  }

  for (; i < simd_end; i += SimdWidth) {
    simd_t vx = SIMD_LOAD(traits, src1 + i);
    simd_t vy = SIMD_LOAD(traits, src2 + i);
    v_sum0 = SIMD_FMA(traits, vx, vy, v_sum0);
  }

  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum1);
  v_sum2 = SIMD_ADD(traits, v_sum2, v_sum3);
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum2);

  T sum = SIMD_SUM(traits, v_sum0);

  for (; i < n; ++i) {
    sum += src1[i] * src2[i];
  }

  return sum;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline T dot_simd(const size_t &n, const T *__restrict__ src1,
                  const T *__restrict__ src2) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

  const size_t simd_end = n - (n % SimdWidth);

  simd_t v_sum0 = SIMD_DUP(traits, T{0});
  simd_t v_sum1 = SIMD_DUP(traits, T{0});
  simd_t v_sum2 = SIMD_DUP(traits, T{0});
  simd_t v_sum3 = SIMD_DUP(traits, T{0});

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    const size_t tile_len = tile_end - tile_start;
    const size_t tile_end4 =
        tile_start + (tile_len - (tile_len % (SimdWidth * 4)));

    size_t i = tile_start;

    for (; i < tile_end4; i += SimdWidth * 4) {
      simd_t vx0 = SIMD_LOAD(traits, src1 + i + SimdWidth * 0);
      simd_t vy0 = SIMD_LOAD(traits, src2 + i + SimdWidth * 0);
      v_sum0 = SIMD_FMA(traits, vx0, vy0, v_sum0);

      simd_t vx1 = SIMD_LOAD(traits, src1 + i + SimdWidth * 1);
      simd_t vy1 = SIMD_LOAD(traits, src2 + i + SimdWidth * 1);
      v_sum1 = SIMD_FMA(traits, vx1, vy1, v_sum1);

      simd_t vx2 = SIMD_LOAD(traits, src1 + i + SimdWidth * 2);
      simd_t vy2 = SIMD_LOAD(traits, src2 + i + SimdWidth * 2);
      v_sum2 = SIMD_FMA(traits, vx2, vy2, v_sum2);

      simd_t vx3 = SIMD_LOAD(traits, src1 + i + SimdWidth * 3);
      simd_t vy3 = SIMD_LOAD(traits, src2 + i + SimdWidth * 3);
      v_sum3 = SIMD_FMA(traits, vx3, vy3, v_sum3);
    }

    for (; i < tile_end; i += SimdWidth) {
      simd_t vx = SIMD_LOAD(traits, src1 + i);
      simd_t vy = SIMD_LOAD(traits, src2 + i);
      v_sum0 = SIMD_FMA(traits, vx, vy, v_sum0);
    }
  }

  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum1);
  v_sum2 = SIMD_ADD(traits, v_sum2, v_sum3);
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum2);

  T sum = SIMD_SUM(traits, v_sum0);

  for (size_t i = simd_end; i < n; ++i) {
    sum += src1[i] * src2[i];
  }

  return sum;
}

// swap
template <typename T, const size_t SimdWidth>
inline void swap_simd(const size_t &n, T *__restrict__ src1,
                      T *__restrict__ src2) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  size_t i = 0;
  for (; i < simd_end; i += SimdWidth) {
    simd_t vx = SIMD_LOAD(traits, src1 + i);
    simd_t vy = SIMD_LOAD(traits, src2 + i);
    SIMD_STORE(traits, src1 + i, vy);
    SIMD_STORE(traits, src2 + i, vx);
  }
  for (; i < n; ++i) {
    T tmp = src1[i];
    src1[i] = src2[i];
    src2[i] = tmp;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void swap_simd(const size_t &n, T *__restrict__ src1,
                      T *__restrict__ src2) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t vx = SIMD_LOAD(traits, src1 + i);
      simd_t vy = SIMD_LOAD(traits, src2 + i);
      SIMD_STORE(traits, src1 + i, vy);
      SIMD_STORE(traits, src2 + i, vx);
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    T tmp = src1[i];
    src1[i] = src2[i];
    src2[i] = tmp;
  }
}

// asum
template <typename T, const size_t SimdWidth>
inline T asum_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  simd_t v_sum0 = SIMD_DUP(traits, T{0});
  simd_t v_sum1 = SIMD_DUP(traits, T{0});
  simd_t v_sum2 = SIMD_DUP(traits, T{0});
  simd_t v_sum3 = SIMD_DUP(traits, T{0});
  size_t i = 0;
  const size_t step4 = SimdWidth * 4;
  const size_t simd_end4 = simd_end - (simd_end % step4);

  for (; i < simd_end4; i += step4) {
    simd_t vx0 = SIMD_LOAD(traits, src + i);
    v_sum0 = SIMD_ADD(traits, v_sum0, SIMD_ABS(traits, vx0));
    simd_t vx1 = SIMD_LOAD(traits, src + i + SimdWidth);
    v_sum1 = SIMD_ADD(traits, v_sum1, SIMD_ABS(traits, vx1));
    simd_t vx2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
    v_sum2 = SIMD_ADD(traits, v_sum2, SIMD_ABS(traits, vx2));
    simd_t vx3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
    v_sum3 = SIMD_ADD(traits, v_sum3, SIMD_ABS(traits, vx3));
  }
  for (; i < simd_end; i += SimdWidth) {
    simd_t vx = SIMD_LOAD(traits, src + i);
    v_sum0 = SIMD_ADD(traits, v_sum0, SIMD_ABS(traits, vx));
  }
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum1);
  v_sum2 = SIMD_ADD(traits, v_sum2, v_sum3);
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum2);
  T sum = SIMD_SUM(traits, v_sum0);
  for (size_t j = simd_end; j < n; ++j)
    sum += std::abs(src[j]);
  return sum;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline T asum_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  simd_t v_sum0 = SIMD_DUP(traits, T{0});
  simd_t v_sum1 = SIMD_DUP(traits, T{0});
  simd_t v_sum2 = SIMD_DUP(traits, T{0});
  simd_t v_sum3 = SIMD_DUP(traits, T{0});
  const size_t step4 = SimdWidth * 4;

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    const size_t tile_len = tile_end - tile_start;
    const size_t tile_end4 = tile_start + (tile_len - (tile_len % step4));
    size_t i = tile_start;
    for (; i < tile_end4; i += step4) {
      simd_t vx0 = SIMD_LOAD(traits, src + i);
      v_sum0 = SIMD_ADD(traits, v_sum0, SIMD_ABS(traits, vx0));
      simd_t vx1 = SIMD_LOAD(traits, src + i + SimdWidth);
      v_sum1 = SIMD_ADD(traits, v_sum1, SIMD_ABS(traits, vx1));
      simd_t vx2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
      v_sum2 = SIMD_ADD(traits, v_sum2, SIMD_ABS(traits, vx2));
      simd_t vx3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
      v_sum3 = SIMD_ADD(traits, v_sum3, SIMD_ABS(traits, vx3));
    }
    for (; i < tile_end; i += SimdWidth) {
      simd_t vx = SIMD_LOAD(traits, src + i);
      v_sum0 = SIMD_ADD(traits, v_sum0, SIMD_ABS(traits, vx));
    }
  }
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum1);
  v_sum2 = SIMD_ADD(traits, v_sum2, v_sum3);
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum2);
  T sum = SIMD_SUM(traits, v_sum0);
  for (size_t i = simd_end; i < n; ++i)
    sum += std::abs(src[i]);
  return sum;
}

// nrm2
template <typename T, const size_t SimdWidth>
inline T nrm2_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  simd_t v_sum0 = SIMD_DUP(traits, T{0});
  simd_t v_sum1 = SIMD_DUP(traits, T{0});
  simd_t v_sum2 = SIMD_DUP(traits, T{0});
  simd_t v_sum3 = SIMD_DUP(traits, T{0});
  size_t i = 0;
  const size_t step4 = SimdWidth * 4;
  const size_t simd_end4 = simd_end - (simd_end % step4);
  for (; i < simd_end4; i += step4) {
    simd_t vx0 = SIMD_LOAD(traits, src + i);
    v_sum0 = SIMD_FMA(traits, vx0, vx0, v_sum0);
    simd_t vx1 = SIMD_LOAD(traits, src + i + SimdWidth);
    v_sum1 = SIMD_FMA(traits, vx1, vx1, v_sum1);
    simd_t vx2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
    v_sum2 = SIMD_FMA(traits, vx2, vx2, v_sum2);
    simd_t vx3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
    v_sum3 = SIMD_FMA(traits, vx3, vx3, v_sum3);
  }
  for (; i < simd_end; i += SimdWidth) {
    simd_t vx = SIMD_LOAD(traits, src + i);
    v_sum0 = SIMD_FMA(traits, vx, vx, v_sum0);
  }
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum1);
  v_sum2 = SIMD_ADD(traits, v_sum2, v_sum3);
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum2);
  T sum = SIMD_SUM(traits, v_sum0);
  for (size_t j = simd_end; j < n; ++j)
    sum += src[j] * src[j];
  return std::sqrt(sum);
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline T nrm2_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  simd_t v_sum0 = SIMD_DUP(traits, T{0});
  simd_t v_sum1 = SIMD_DUP(traits, T{0});
  simd_t v_sum2 = SIMD_DUP(traits, T{0});
  simd_t v_sum3 = SIMD_DUP(traits, T{0});
  const size_t step4 = SimdWidth * 4;

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    const size_t tile_len = tile_end - tile_start;
    const size_t tile_end4 = tile_start + (tile_len - (tile_len % step4));
    size_t i = tile_start;
    for (; i < tile_end4; i += step4) {
      simd_t vx0 = SIMD_LOAD(traits, src + i);
      v_sum0 = SIMD_FMA(traits, vx0, vx0, v_sum0);
      simd_t vx1 = SIMD_LOAD(traits, src + i + SimdWidth);
      v_sum1 = SIMD_FMA(traits, vx1, vx1, v_sum1);
      simd_t vx2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
      v_sum2 = SIMD_FMA(traits, vx2, vx2, v_sum2);
      simd_t vx3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
      v_sum3 = SIMD_FMA(traits, vx3, vx3, v_sum3);
    }
    for (; i < tile_end; i += SimdWidth) {
      simd_t vx = SIMD_LOAD(traits, src + i);
      v_sum0 = SIMD_FMA(traits, vx, vx, v_sum0);
    }
  }
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum1);
  v_sum2 = SIMD_ADD(traits, v_sum2, v_sum3);
  v_sum0 = SIMD_ADD(traits, v_sum0, v_sum2);
  T sum = SIMD_SUM(traits, v_sum0);
  for (size_t i = simd_end; i < n; ++i)
    sum += src[i] * src[i];
  return std::sqrt(sum);
}

template <typename T, const size_t SimdWidth>
inline size_t iamax_simd(const size_t &n, const T *__restrict__ src) {
  if (n == 0)
    return 0;
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  size_t max_index = 0;
  T max_value = std::abs(src[0]);

  size_t i = 0;
  const size_t step4 = SimdWidth * 4;
  alignas(64) T tmp[SimdWidth * 4 > 1 ? SimdWidth * 4 : 1];
  const size_t simd_end4 = simd_end - (simd_end % step4);
  for (; i < simd_end4; i += step4) {
    simd_t v0 = SIMD_LOAD(traits, src + i);
    simd_t a0 = SIMD_ABS(traits, v0);
    SIMD_STORE(traits, tmp + 0, a0);
    simd_t v1 = SIMD_LOAD(traits, src + i + SimdWidth);
    simd_t a1 = SIMD_ABS(traits, v1);
    SIMD_STORE(traits, tmp + SimdWidth, a1);
    simd_t v2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
    simd_t a2 = SIMD_ABS(traits, v2);
    SIMD_STORE(traits, tmp + SimdWidth * 2, a2);
    simd_t v3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
    simd_t a3 = SIMD_ABS(traits, v3);
    SIMD_STORE(traits, tmp + SimdWidth * 3, a3);
    for (size_t j = 0; j < step4; ++j) {
      T val = tmp[j];
      if (val > max_value) {
        max_value = val;
        max_index = i + j;
      }
    }
  }
  for (; i < simd_end; i += SimdWidth) {
    simd_t v = SIMD_LOAD(traits, src + i);
    simd_t a = SIMD_ABS(traits, v);
    SIMD_STORE(traits, tmp, a);
    for (size_t j = 0; j < SimdWidth; ++j) {
      T val = tmp[j];
      if (val > max_value) {
        max_value = val;
        max_index = i + j;
      }
    }
  }
  for (; i < n; ++i) {
    T val = std::abs(src[i]);
    if (val > max_value) {
      max_value = val;
      max_index = i;
    }
  }
  return max_index;
}

// iamax
template <typename T, const size_t TileSize, const size_t SimdWidth>
inline size_t iamax_simd(const size_t &n, const T *__restrict__ src) {
  if (n == 0)
    return 0;
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  size_t max_index = 0;
  T max_value = std::abs(src[0]);
  alignas(64) T tmp[SimdWidth * 4 > 1 ? SimdWidth * 4 : 1];

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    size_t i = tile_start;
    const size_t step4 = SimdWidth * 4;
    const size_t tile_end4 =
        tile_start + ((tile_end - tile_start) / step4) * step4;
    for (; i < tile_end4; i += step4) {
      simd_t v0 = SIMD_LOAD(traits, src + i);
      simd_t a0 = SIMD_ABS(traits, v0);
      SIMD_STORE(traits, tmp + 0, a0);
      simd_t v1 = SIMD_LOAD(traits, src + i + SimdWidth);
      simd_t a1 = SIMD_ABS(traits, v1);
      SIMD_STORE(traits, tmp + SimdWidth, a1);
      simd_t v2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
      simd_t a2 = SIMD_ABS(traits, v2);
      SIMD_STORE(traits, tmp + SimdWidth * 2, a2);
      simd_t v3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
      simd_t a3 = SIMD_ABS(traits, v3);
      SIMD_STORE(traits, tmp + SimdWidth * 3, a3);
      for (size_t j = 0; j < step4; ++j) {
        T val = tmp[j];
        if (val > max_value) {
          max_value = val;
          max_index = i + j;
        }
      }
    }
    for (; i < tile_end; i += SimdWidth) {
      simd_t v = SIMD_LOAD(traits, src + i);
      simd_t a = SIMD_ABS(traits, v);
      SIMD_STORE(traits, tmp, a);
      for (size_t j = 0; j < SimdWidth; ++j) {
        T val = tmp[j];
        if (val > max_value) {
          max_value = val;
          max_index = i + j;
        }
      }
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    T val = std::abs(src[i]);
    if (val > max_value) {
      max_value = val;
      max_index = i;
    }
  }
  return max_index;
}

} // namespace details
} // namespace hpc::l1

#endif
