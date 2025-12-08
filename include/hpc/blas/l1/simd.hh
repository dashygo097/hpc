#pragma once

#ifdef ENABLE_SIMD
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_SIMD
#define ENABLE_SIMD_VECTOR_SCALAR_BRANCH(name)                                 \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, src, scalar);            \
  }
#define ENABLE_SIMD_BINARY_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, src1, src2);             \
  }
#define ENABLE_SIMD_UNARY_BRANCH(name)                                         \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, src);                    \
  }
#define ENABLE_SIMD_SCALAR_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    details::name##_simd<T, BackendParams...>(n, dst, scalar);                 \
  }
#define ENABLE_SIMD_REDUCE_BRANCH(name)                                        \
  else if constexpr (backend == Backend::SIMD) {                               \
    return details::name##_simd<T, BackendParams...>(n, src);                  \
  }
#define ENABLE_SIMD_REDUCE2_BRANCH(name)                                       \
  else if constexpr (backend == Backend::SIMD) {                               \
    return details::name##_simd<T, BackendParams...>(n, src1, src2);           \
  }
#else
#define ENABLE_SIMD_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_SIMD_BINARY_BRANCH(name)
#define ENABLE_SIMD_UNARY_BRANCH(name)
#define ENABLE_SIMD_SCALAR_BRANCH(name)
#define ENABLE_SIMD_REDUCE_BRANCH(name)
#define ENABLE_SIMD_REDUCE2_BRANCH(name)
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
  } else if (alpha == T{1}) {
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      simd_t vy = SIMD_LOAD(traits, dst + i);
      simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, SIMD_ADD(traits, vx, vy));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += src[i];
    return;
  } else {
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      simd_t vy = SIMD_LOAD(traits, dst + i);
      simd_t vx = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i, SIMD_FMA(traits, v_a, vx, vy));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] += alpha * src[i];
  }
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
  } else if (alpha == T{1}) {
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
  } else {
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

// scal
template <typename T, const size_t SimdWidth>
inline void scal_simd(const size_t &n, T *__restrict__ dst, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_zero = SIMD_DUP(traits, T{0});
  const simd_t v_a = SIMD_DUP(traits, alpha);

  if (alpha == T{0}) {
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i, v_zero);
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] = T{0};
  } else if (alpha == T{1}) {
    return;
  } else {
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      simd_t vx = SIMD_LOAD(traits, dst + i);
      SIMD_STORE(traits, dst + i, SIMD_MUL(traits, v_a, vx));
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] *= alpha;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void scal_simd(const size_t &n, T *__restrict__ dst, const T &alpha) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, alpha);
  const simd_t v_zero = SIMD_DUP(traits, T{0});

  if (alpha == T{0}) {
    for (size_t i = 0; i < simd_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i, v_zero);
    }
    for (size_t i = simd_end; i < n; ++i)
      dst[i] = T{0};
  } else if (alpha == T{1}) {
    return;
  } else {
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
}

// dot
template <typename T, const size_t SimdWidth>
inline T dot_simd(const size_t &n, const T *__restrict__ src1,
                  const T *__restrict__ src2) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  simd_t v_sum = SIMD_DUP(traits, T{0});

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vx = SIMD_LOAD(traits, src1 + i);
    simd_t vy = SIMD_LOAD(traits, src2 + i);
    v_sum = SIMD_FMA(traits, vx, vy, v_sum);
  }

  T sum = SIMD_SUM(traits, v_sum);
  for (size_t i = simd_end; i < n; ++i)
    sum += src1[i] * src2[i];
  return sum;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline T dot_simd(const size_t &n, const T *__restrict__ src1,
                  const T *__restrict__ src2) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  simd_t v_sum = SIMD_DUP(traits, T{0});
  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t vx = SIMD_LOAD(traits, src1 + i);
      simd_t vy = SIMD_LOAD(traits, src2 + i);
      v_sum = SIMD_FMA(traits, vx, vy, v_sum);
    }
  }

  T sum = SIMD_SUM(traits, v_sum);
  for (size_t i = simd_end; i < n; ++i)
    sum += src1[i] * src2[i];

  return sum;
}

} // namespace details
} // namespace hpc::l1
#endif
