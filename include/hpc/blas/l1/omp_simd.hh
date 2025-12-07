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
    return details::name##_omp_simd<T, BackendParams...>(n, x, y);             \
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

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vy = SIMD_LOAD(traits, dst + i);
    simd_t vx = SIMD_LOAD(traits, src + i);
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

#pragma omp parallel for schedule(static)
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
inline void copy_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vs = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, vs);
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] = src[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void copy_omp_simd(const size_t &n, T *__restrict__ dst,
                          const T *__restrict__ src) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
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
