#pragma once

#if defined(ENABLE_SIMD) && defined(ENABLE_OPENMP)
#include "../../backends/backends.hh"
#endif

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
#define ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)                          \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src, scalar);        \
  }
#define ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src1, src2);         \
  }
#define ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)                                  \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, src);                \
  }
#define ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    details::name##_omp_simd<T, BackendParams...>(n, dst, scalar);             \
  }
#define ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)                                 \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    return details::name##_omp_simd<T, BackendParams...>(n, src);              \
  }
#define ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)                                \
  else if constexpr (backend == Backend::OPENMP_SIMD) {                        \
    return details::name##_omp_simd<T, BackendParams...>(n, src1, src2);       \
  }
#else
#define ENABLE_OPENMP_SIMD_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_SIMD_BINARY_BRANCH(name)
#define ENABLE_OPENMP_SIMD_UNARY_BRANCH(name)
#define ENABLE_OPENMP_SIMD_SCALAR_BRANCH(name)
#define ENABLE_OPENMP_SIMD_REDUCE_BRANCH(name)
#define ENABLE_OPENMP_SIMD_REDUCE2_BRANCH(name)
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
  } else if (alpha == T{1}) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      const simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, vx + SIMD_LOAD(traits, dst + i));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += src[i];
    return;
  } else {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      const simd_t vy = SIMD_LOAD(traits, dst + i);
      const simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, SIMD_FMA(traits, v_a, vx, vy));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += alpha * src[i];
  }
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
  } else if (alpha == T{1}) {
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
  } else {
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

  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
  } else if (alpha == T{1}) {
    return;
  } else {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      const simd_t vx = SIMD_LOAD(traits, dst + i);
      SIMD_STORE(traits, dst + i, SIMD_MUL(traits, v_a, vx));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] *= alpha;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void scal_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, alpha);
  const simd_t v_zero = SIMD_DUP(traits, T{0});

  if (alpha == T{0}) {
    memset(dst, 0, n * sizeof(T));
  } else if (alpha == T{1}) {
    return;
  } else {
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
  assert(TileSize % (SimdWidth * 4) == 0 &&
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

} // namespace details
} // namespace hpc::l1
#endif
