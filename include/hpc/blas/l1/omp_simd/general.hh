#pragma once

#if defined(ENABLE_SIMD) && defined(ENABLE_OPENMP)
#include "../../../backends/backends.hh"
#include <algorithm>
#include <cstring>
#endif

#if defined(ENABLE_SIMD) && defined(ENABLE_OPENMP)
namespace hpc::l1 {
namespace details {

// axpy
template <typename T, const size_t SimdWidth>
inline void axpy_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, alpha);

  if (alpha == T{0}) {
    return;
  }

  if (alpha == T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      const simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, vx + SIMD_LOAD(traits, dst + i));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += src[i];
    return;
  }

  if (alpha == T{-1}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      const simd_t vy = SIMD_LOAD(traits, dst + i);
      const simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, vy - vx);
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] -= src[i];
    return;
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    const simd_t vy = SIMD_LOAD(traits, dst + i);
    const simd_t vx = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, SIMD_FMA(traits, v_a, vx, vy));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] += alpha * src[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void axpy_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, alpha);

  if (alpha == T{0}) {
    return;
  }

  if (alpha == T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
      const size_t tile_end = tile_start + TileSize;
      for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
        const simd_t vx = SIMD_LOAD(traits, src + i);
        SIMD_STORE(traits, dst + i, vx + SIMD_LOAD(traits, dst + i));
      }
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += src[i];
    return;
  }

  if (alpha == T{-1}) {
#pragma omp parallel for schedule(static)
    for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
      const size_t tile_end = tile_start + TileSize;
      for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
        const simd_t vy = SIMD_LOAD(traits, dst + i);
        const simd_t vx = SIMD_LOAD(traits, src + i);
        SIMD_STORE(traits, dst + i, vy - vx);
      }
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] -= src[i];
    return;
  }

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      const simd_t vy = SIMD_LOAD(traits, dst + i);
      const simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, SIMD_FMA(traits, v_a, vx, vy));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] += alpha * src[i];
}

// copy
template <typename T, const size_t SimdWidth>
inline void copy_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  memcpy(dst, src, n * sizeof(T));
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void copy_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
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
inline void scal_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T &alpha) {
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

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    const simd_t vx = SIMD_LOAD(traits, dst + i);
    SIMD_STORE(traits, dst + i, SIMD_MUL(traits, v_a, vx));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= alpha;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void scal_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T &alpha) {
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

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      const simd_t vx = SIMD_LOAD(traits, dst + i);
      SIMD_STORE(traits, dst + i, SIMD_MUL(traits, v_a, vx));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= alpha;
}

// dot
template <typename T, const size_t SimdWidth>
inline T dot_omp_simd(const size_t &n, const T *__restrict__ src1,
                      const T *__restrict__ src2) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

  const size_t simd_end = n - (n % SimdWidth);
  T result = T{0};

#pragma omp parallel reduction(+ : result)
  {
    simd_t v_sum0 = SIMD_DUP(traits, T{0});
    simd_t v_sum1 = SIMD_DUP(traits, T{0});
    simd_t v_sum2 = SIMD_DUP(traits, T{0});
    simd_t v_sum3 = SIMD_DUP(traits, T{0});

#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < simd_end; i += SimdWidth * 4) {
      if (i + SimdWidth * 4 <= simd_end) {
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
      } else {
        for (size_t j = i; j < simd_end; j += SimdWidth) {
          simd_t vx = SIMD_LOAD(traits, src1 + j);
          simd_t vy = SIMD_LOAD(traits, src2 + j);
          v_sum0 = SIMD_FMA(traits, vx, vy, v_sum0);
        }
      }
    }

    v_sum0 = SIMD_ADD(traits, v_sum0, v_sum1);
    v_sum2 = SIMD_ADD(traits, v_sum2, v_sum3);
    v_sum0 = SIMD_ADD(traits, v_sum0, v_sum2);

    result += SIMD_SUM(traits, v_sum0);
  }

  for (size_t i = simd_end; i < n; ++i) {
    result += src1[i] * src2[i];
  }

  return result;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline T dot_omp_simd(const size_t &n, const T *__restrict__ src1,
                      const T *__restrict__ src2) {
  static_assert(TileSize % (SimdWidth * 4) == 0 &&
                "TileSize must be multiple of SimdWidth * 4");
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

  const size_t simd_end = n - (n % SimdWidth);
  T result = T{0};

#pragma omp parallel reduction(+ : result)
  {
    simd_t v_sum0 = SIMD_DUP(traits, T{0});
    simd_t v_sum1 = SIMD_DUP(traits, T{0});
    simd_t v_sum2 = SIMD_DUP(traits, T{0});
    simd_t v_sum3 = SIMD_DUP(traits, T{0});

#pragma omp for schedule(static) nowait
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

    result += SIMD_SUM(traits, v_sum0);
  }

  for (size_t i = simd_end; i < n; ++i) {
    result += src1[i] * src2[i];
  }

  return result;
}

// swap
template <typename T, const size_t SimdWidth>
inline void swap_omp_simd(const size_t &n, T *__restrict__ src1,
                          T *__restrict__ src2) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vx = SIMD_LOAD(traits, src1 + i);
    simd_t vy = SIMD_LOAD(traits, src2 + i);
    SIMD_STORE(traits, src1 + i, vy);
    SIMD_STORE(traits, src2 + i, vx);
  }
  for (size_t i = simd_end; i < n; ++i) {
    T tmp = src1[i];
    src1[i] = src2[i];
    src2[i] = tmp;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void swap_simd_omp(const size_t &n, T *__restrict__ src1,
                          T *__restrict__ src2) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
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
inline T asum_omp_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const size_t step4 = SimdWidth * 4;
  const size_t simd_end4 = simd_end - (simd_end % step4);
  T global_sum = T{0};

#pragma omp parallel
  {
    simd_t v0 = SIMD_DUP(traits, T{0});
    simd_t v1 = SIMD_DUP(traits, T{0});
    simd_t v2 = SIMD_DUP(traits, T{0});
    simd_t v3 = SIMD_DUP(traits, T{0});
#pragma omp for schedule(static)
    for (size_t i = 0; i < simd_end4; i += step4) {
      simd_t x0 = SIMD_LOAD(traits, src + i);
      v0 = SIMD_ADD(traits, v0, SIMD_ABS(traits, x0));
      simd_t x1 = SIMD_LOAD(traits, src + i + SimdWidth);
      v1 = SIMD_ADD(traits, v1, SIMD_ABS(traits, x1));
      simd_t x2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
      v2 = SIMD_ADD(traits, v2, SIMD_ABS(traits, x2));
      simd_t x3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
      v3 = SIMD_ADD(traits, v3, SIMD_ABS(traits, x3));
    }
    for (size_t i = simd_end4; i < simd_end; i += SimdWidth) {
      simd_t x = SIMD_LOAD(traits, src + i);
      v0 = SIMD_ADD(traits, v0, SIMD_ABS(traits, x));
    }
    T local_sum = SIMD_SUM(traits, SIMD_ADD(traits, SIMD_ADD(traits, v0, v1),
                                            SIMD_ADD(traits, v2, v3)));
    for (size_t i = simd_end; i < n; ++i)
      local_sum += std::abs(src[i]);
#pragma omp atomic
    global_sum += local_sum;
  }
  return global_sum;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline T asum_omp_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const size_t step4 = SimdWidth * 4;
  T global_sum = T{0};

#pragma omp parallel for schedule(static) reduction(+ : global_sum)
  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    const size_t tile_len = tile_end - tile_start;
    const size_t tile_end4 = tile_start + (tile_len - (tile_len % step4));
    simd_t v0 = SIMD_DUP(traits, T{0});
    simd_t v1 = SIMD_DUP(traits, T{0});
    simd_t v2 = SIMD_DUP(traits, T{0});
    simd_t v3 = SIMD_DUP(traits, T{0});
    size_t i = tile_start;
    for (; i < tile_end4; i += step4) {
      simd_t x0 = SIMD_LOAD(traits, src + i);
      v0 = SIMD_ADD(traits, v0, SIMD_ABS(traits, x0));
      simd_t x1 = SIMD_LOAD(traits, src + i + SimdWidth);
      v1 = SIMD_ADD(traits, v1, SIMD_ABS(traits, x1));
      simd_t x2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
      v2 = SIMD_ADD(traits, v2, SIMD_ABS(traits, x2));
      simd_t x3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
      v3 = SIMD_ADD(traits, v3, SIMD_ABS(traits, x3));
    }
    for (; i < tile_end; i += SimdWidth) {
      simd_t x = SIMD_LOAD(traits, src + i);
      v0 = SIMD_ADD(traits, v0, SIMD_ABS(traits, x));
    }
    T local_sum = SIMD_SUM(traits, SIMD_ADD(traits, SIMD_ADD(traits, v0, v1),
                                            SIMD_ADD(traits, v2, v3)));
    global_sum += local_sum;
  }
  for (size_t i = simd_end; i < n; ++i)
    global_sum += std::abs(src[i]);
  return global_sum;
}

template <typename T, const size_t SimdWidth>
inline T nrm2_omp_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const size_t step4 = SimdWidth * 4;
  const size_t simd_end4 = simd_end - (simd_end % step4);
  T global_sum = T{0};

#pragma omp parallel
  {
    simd_t v0 = SIMD_DUP(traits, T{0});
    simd_t v1 = SIMD_DUP(traits, T{0});
    simd_t v2 = SIMD_DUP(traits, T{0});
    simd_t v3 = SIMD_DUP(traits, T{0});
#pragma omp for schedule(static)
    for (size_t i = 0; i < simd_end4; i += step4) {
      simd_t x0 = SIMD_LOAD(traits, src + i);
      v0 = SIMD_FMA(traits, x0, x0, v0);
      simd_t x1 = SIMD_LOAD(traits, src + i + SimdWidth);
      v1 = SIMD_FMA(traits, x1, x1, v1);
      simd_t x2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
      v2 = SIMD_FMA(traits, x2, x2, v2);
      simd_t x3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
      v3 = SIMD_FMA(traits, x3, x3, v3);
    }
    for (size_t i = simd_end4; i < simd_end; i += SimdWidth) {
      simd_t x = SIMD_LOAD(traits, src + i);
      v0 = SIMD_FMA(traits, x, x, v0);
    }
    T local_sum = SIMD_SUM(traits, SIMD_ADD(traits, SIMD_ADD(traits, v0, v1),
                                            SIMD_ADD(traits, v2, v3)));
    for (size_t i = simd_end; i < n; ++i)
      local_sum += src[i] * src[i];
#pragma omp atomic
    global_sum += local_sum;
  }
  return std::sqrt(global_sum);
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline T nrm2_omp_simd(const size_t &n, const T *__restrict__ src) {
  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const size_t step4 = SimdWidth * 4;
  T global_sum = T{0};

#pragma omp parallel for schedule(static) reduction(+ : global_sum)
  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    const size_t tile_len = tile_end - tile_start;
    const size_t tile_end4 = tile_start + (tile_len - (tile_len % step4));
    simd_t v0 = SIMD_DUP(traits, T{0});
    simd_t v1 = SIMD_DUP(traits, T{0});
    simd_t v2 = SIMD_DUP(traits, T{0});
    simd_t v3 = SIMD_DUP(traits, T{0});
    size_t i = tile_start;
    for (; i < tile_end4; i += step4) {
      simd_t x0 = SIMD_LOAD(traits, src + i);
      v0 = SIMD_FMA(traits, x0, x0, v0);
      simd_t x1 = SIMD_LOAD(traits, src + i + SimdWidth);
      v1 = SIMD_FMA(traits, x1, x1, v1);
      simd_t x2 = SIMD_LOAD(traits, src + i + SimdWidth * 2);
      v2 = SIMD_FMA(traits, x2, x2, v2);
      simd_t x3 = SIMD_LOAD(traits, src + i + SimdWidth * 3);
      v3 = SIMD_FMA(traits, x3, x3, v3);
    }
    for (; i < tile_end; i += SimdWidth) {
      simd_t x = SIMD_LOAD(traits, src + i);
      v0 = SIMD_FMA(traits, x, x, v0);
    }
    T local_sum = SIMD_SUM(traits, SIMD_ADD(traits, SIMD_ADD(traits, v0, v1),
                                            SIMD_ADD(traits, v2, v3)));
    global_sum += local_sum;
  }
  for (size_t i = simd_end; i < n; ++i)
    global_sum += src[i] * src[i];
  return std::sqrt(global_sum);
}

template <typename T, const size_t SimdWidth>
inline size_t iamax_omp_simd(const size_t &n, const T *__restrict__ src) {
  if (n == 0)
    return 0;

  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const size_t step4 = SimdWidth * 4;
  const size_t simd_end4 = simd_end - (simd_end % step4);
  size_t global_idx = 0;
  T global_max = std::abs(src[0]);

#pragma omp parallel
  {
    size_t local_idx = 0;
    T local_max = T{0};
    alignas(64) T tmp[SimdWidth * 4 > 1 ? SimdWidth * 4 : 1];
#pragma omp for schedule(static)
    for (size_t i = 0; i < simd_end4; i += step4) {
      simd_t v0 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i));
      SIMD_STORE(traits, tmp + 0, v0);
      simd_t v1 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i + SimdWidth));
      SIMD_STORE(traits, tmp + SimdWidth, v1);
      simd_t v2 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i + SimdWidth * 2));
      SIMD_STORE(traits, tmp + SimdWidth * 2, v2);
      simd_t v3 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i + SimdWidth * 3));
      SIMD_STORE(traits, tmp + SimdWidth * 3, v3);
      for (size_t j = 0; j < step4; ++j) {
        T val = tmp[j];
        if (val > local_max) {
          local_max = val;
          local_idx = i + j;
        }
      }
    }
    for (size_t i = simd_end4; i < simd_end; i += SimdWidth) {
      simd_t v = SIMD_ABS(traits, SIMD_LOAD(traits, src + i));
      SIMD_STORE(traits, tmp, v);
      for (size_t j = 0; j < SimdWidth; ++j) {
        T val = tmp[j];
        if (val > local_max) {
          local_max = val;
          local_idx = i + j;
        }
      }
    }
    for (size_t i = simd_end; i < n; ++i) {
      T val = std::abs(src[i]);
      if (val > local_max) {
        local_max = val;
        local_idx = i;
      }
    }
#pragma omp critical
    {
      if (local_max > global_max) {
        global_max = local_max;
        global_idx = local_idx;
      }
    }
  }
  return global_idx;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline size_t iamax_omp_simd(const size_t &n, const T *__restrict__ src) {
  if (n == 0)
    return 0;

  using traits = hpc::simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  size_t global_idx = 0;
  T global_max = std::abs(src[0]);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    const size_t tile_end = std::min(tile_start + TileSize, simd_end);
    size_t local_idx = 0;
    T local_max = T{0};
    const size_t step4 = SimdWidth * 4;
    const size_t tile_end4 =
        tile_start + ((tile_end - tile_start) / step4) * step4;
    alignas(64) T tmp[SimdWidth * 4 > 1 ? SimdWidth * 4 : 1];
    size_t i = tile_start;
    for (; i < tile_end4; i += step4) {
      simd_t v0 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i));
      SIMD_STORE(traits, tmp + 0, v0);
      simd_t v1 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i + SimdWidth));
      SIMD_STORE(traits, tmp + SimdWidth, v1);
      simd_t v2 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i + SimdWidth * 2));
      SIMD_STORE(traits, tmp + SimdWidth * 2, v2);
      simd_t v3 = SIMD_ABS(traits, SIMD_LOAD(traits, src + i + SimdWidth * 3));
      SIMD_STORE(traits, tmp + SimdWidth * 3, v3);
      for (size_t j = 0; j < step4; ++j) {
        T val = tmp[j];
        if (val > local_max) {
          local_max = val;
          local_idx = i + j;
        }
      }
    }
    for (; i < tile_end; i += SimdWidth) {
      simd_t v = SIMD_ABS(traits, SIMD_LOAD(traits, src + i));
      SIMD_STORE(traits, tmp, v);
      for (size_t j = 0; j < SimdWidth; ++j) {
        T val = tmp[j];
        if (val > local_max) {
          local_max = val;
          local_idx = i + j;
        }
      }
    }
#pragma omp critical
    {
      if (local_max > global_max) {
        global_max = local_max;
        global_idx = local_idx;
      }
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    T val = std::abs(src[i]);
    if (val > global_max) {
      global_max = val;
      global_idx = i;
    }
  }
  return global_idx;
}

} // namespace details
} // namespace hpc::l1
#endif
