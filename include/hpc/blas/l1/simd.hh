#pragma once

#ifdef ENABLE_SIMD
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_SIMD
namespace hpc::l1 {
namespace details {

// addition
template <typename T, const size_t SimdWidth>
inline void vadd_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t v = SIMD_LOAD(traits, dst + i);
    SIMD_STORE(traits, dst + i, SIMD_ADD(traits, v, scalar_vec));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] += scalar;
}

template <typename T, const size_t SimdWidth>
inline void vadd_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vd = SIMD_LOAD(traits, dst + i);
    simd_t vs = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, SIMD_ADD(traits, vd, vs));
  }

  for (size_t i = simd_end; i < n; ++i) {
    dst[i] += src[i];
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vadd_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

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
inline void vadd_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  size_t simd_end = n - (n % SimdWidth);

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

// subtraction
template <typename T, const size_t SimdWidth>
inline void vsub_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t v = SIMD_LOAD(traits, dst + i);
    SIMD_STORE(traits, dst + i, SIMD_SUB(traits, v, scalar_vec));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] -= scalar;
}

template <typename T, const size_t SimdWidth>
inline void vsub_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vd = SIMD_LOAD(traits, dst + i);
    simd_t vs = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, SIMD_SUB(traits, vd, vs));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] -= src[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vsub_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i,
                 SIMD_SUB(traits, SIMD_LOAD(traits, dst + i), scalar_vec));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] -= scalar;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vsub_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i,
                 SIMD_SUB(traits, SIMD_LOAD(traits, dst + i), v_src));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] -= src[i];
}

// multiplication
template <typename T, const size_t SimdWidth>
inline void vmul_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t v = traits::load(dst + i);
    traits::store(dst + i, traits::mul(v, scalar_vec));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= scalar;
}

template <typename T, const size_t SimdWidth>
inline void vmul_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vd = SIMD_LOAD(traits, dst + i);
    simd_t vs = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, SIMD_MUL(traits, vd, vs));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= src[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vmul_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i,
                 SIMD_MUL(traits, SIMD_LOAD(traits, dst + i), scalar_vec));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= scalar;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vmul_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i,
                 SIMD_MUL(traits, SIMD_LOAD(traits, dst + i), v_src));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= src[i];
}

// division
template <typename T, const size_t SimdWidth>
inline void vdiv_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t v = SIMD_LOAD(traits, dst + i);
    SIMD_STORE(traits, dst + i, SIMD_DIV(traits, v, scalar_vec));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] /= scalar;
}

template <typename T, const size_t SimdWidth>
inline void vdiv_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vd = SIMD_LOAD(traits, dst + i);
    simd_t vs = SIMD_LOAD(traits, src + i);
    SIMD_STORE(traits, dst + i, SIMD_DIV(traits, vd, vs));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] /= src[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vdiv_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t scalar_vec = SIMD_DUP(traits, scalar);

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i,
                 SIMD_DIV(traits, SIMD_LOAD(traits, dst + i), scalar_vec));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] /= scalar;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vdiv_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t v_src = SIMD_LOAD(traits, src + i);
      SIMD_STORE(traits, dst + i,
                 SIMD_DIV(traits, SIMD_LOAD(traits, dst + i), v_src));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] /= src[i];
}

// fill
template <typename T, const size_t SimdWidth>
inline void vfill_simd(T *__restrict__ dst, const T &value, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t value_vec = SIMD_DUP(traits, value);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    SIMD_STORE(traits, dst + i, value_vec);
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] = value;
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vfill_simd(T *__restrict__ dst, const T &value, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t value_vec = SIMD_DUP(traits, value);

  for (size_t tile_start = 0; tile_start < n; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      SIMD_STORE(traits, dst + i, value_vec);
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] = value;
}

// copy
template <typename T, const size_t SimdWidth>
inline void vcopy_simd(T *__restrict__ dst, const T *__restrict__ src,
                       size_t n) {
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
inline void vcopy_simd(T *__restrict__ dst, const T *__restrict__ src,
                       size_t n) {
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

// l1

// axpy
template <typename T, const size_t SimdWidth>
inline void axpy_simd(T *__restrict__ y, const T *__restrict__ x, const T &a,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, a);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vy = SIMD_LOAD(traits, y + i);
    simd_t vx = SIMD_LOAD(traits, x + i);
    SIMD_STORE(traits, y + i, SIMD_FMA(traits, v_a, vx, vy));
  }
  for (size_t i = simd_end; i < n; ++i)
    y[i] += a * x[i];
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void axpy_simd(T *__restrict__ y, const T *__restrict__ x, const T &a,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  const size_t simd_end = n - (n % SimdWidth);
  const simd_t v_a = SIMD_DUP(traits, a);

  for (size_t tile_start = 0; tile_start < simd_end; tile_start += TileSize) {
    size_t tile_end = tile_start + TileSize;
    for (size_t i = tile_start; i < tile_end; i += SimdWidth) {
      simd_t vy = SIMD_LOAD(traits, y + i);
      simd_t vx = SIMD_LOAD(traits, x + i);
      SIMD_STORE(traits, y + i, SIMD_FMA(traits, v_a, vx, vy));
    }
  }
  for (size_t i = simd_end; i < n; ++i)
    y[i] += a * x[i];
}

} // namespace details
} // namespace hpc::l1
#endif
