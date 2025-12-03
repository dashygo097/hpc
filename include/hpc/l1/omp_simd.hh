#pragma once

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
#include "../backends/backends.hh"
#include <algorithm>
#endif

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
namespace hpc::l1 {
namespace details {

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vadd_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i,
                 SIMD_ADD(traits, SIMD_LOAD(traits, dst + i), scalar_vec));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] += scalar;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vadd_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i,
                 SIMD_ADD(traits, SIMD_LOAD(traits, dst + i), v_src));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] += src[i];
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vsub_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i,
                 SIMD_SUB(traits, SIMD_LOAD(traits, dst + i), scalar_vec));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] -= scalar;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vsub_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i,
                 SIMD_SUB(traits, SIMD_LOAD(traits, dst + i), v_src));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] -= src[i];
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vmul_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);
  simd_t scalar_vec = SIMD_DUP(traits, scalar);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i,
                 SIMD_MUL(traits, SIMD_LOAD(traits, dst + i), scalar_vec));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] *= scalar;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vmul_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i,
                 SIMD_MUL(traits, SIMD_LOAD(traits, dst + i), v_src));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] *= src[i];
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vdiv_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);
  const size_t scalar_vec = SIMD_DUP(traits, scalar);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i,
                 SIMD_DIV(traits, SIMD_LOAD(traits, dst + i), scalar_vec));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] /= scalar;
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vdiv_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i,
                 SIMD_DIV(traits, SIMD_LOAD(traits, dst + i), v_src));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] /= src[i];
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vfill_omp_simd(T *__restrict__ dst, const T &value, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);
  simd_t v_value = SIMD_DUP(traits, value);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i, v_value);
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] = value;
  }
}

// l1
template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void axpy_omp_simd(T *__restrict__ y, const T *__restrict__ x, const T a,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);
  simd_t v_a = SIMD_DUP(traits, a);

#pragma omp parallel for schedule(static)
  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_x = SIMD_LOAD(traits, x + i);
      simd_t v_y = SIMD_LOAD(traits, y + i);
      SIMD_STORE(traits, y + i, SIMD_FMA(traits, v_a, v_x, v_y));
    }
  }
  for (size_t i = simd_end; i < n; ++i) {
    y[i] += a * x[i];
  }
}

} // namespace details
} // namespace hpc::l1
#endif
