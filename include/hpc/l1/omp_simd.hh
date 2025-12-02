#pragma once

#include "../backends/backends.hh"
#include <algorithm>

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
namespace hpc::l1 {
namespace details {

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vadd_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) += traits::duplicate(scalar);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::add(v, traits::duplicate(scalar));
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vadd_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) += *((simd_t *)(src + i));
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v_dst = *((simd_t *)(dst + SimdWidth));
    simd_t v_src = *((simd_t *)(src + SimdWidth));
    simd_t result = traits::add(v_dst, v_src);
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vsub_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) -= traits::duplicate(scalar);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::sub(v, traits::duplicate(scalar));
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vsub_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) -= *((simd_t *)(src + i));
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v_dst = *((simd_t *)(dst + SimdWidth));
    simd_t v_src = *((simd_t *)(src + SimdWidth));
    simd_t result = traits::sub(v_dst, v_src);
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vmul_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) *= traits::duplicate(scalar);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::mul(v, traits::duplicate(scalar));
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vmul_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) *= *((simd_t *)(src + i));
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v_dst = *((simd_t *)(dst + SimdWidth));
    simd_t v_src = *((simd_t *)(src + SimdWidth));
    simd_t result = traits::mul(v_dst, v_src);
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vdiv_omp_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) /= traits::duplicate(scalar);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::div(v, traits::duplicate(scalar));
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vdiv_omp_simd(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) /= *((simd_t *)(src + i));
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v_dst = *((simd_t *)(dst + SimdWidth));
    simd_t v_src = *((simd_t *)(src + SimdWidth));
    simd_t result = traits::div(v_dst, v_src);
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = result[i];
    }
  }
}

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vfill_omp_simd(T *__restrict__ dst, const T &value, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  simd_t v_value = traits::duplicate(value);
#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) = v_value;
    }
  }
  if (n % SimdWidth != 0) {
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((dst + SimdWidth + i)) = value;
    }
  }
}

// l1
template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void axpy_omp_simd(T *__restrict__ y, const T *__restrict__ x, const T a,
                          size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;
  simd_t v_a = traits::duplicate(a);
#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      simd_t v_x = *((simd_t *)(x + i));
      simd_t v_y = *((simd_t *)(y + i));
      *((simd_t *)(y + i)) = traits::add(v_y, traits::mul(v_a, v_x));
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v_a = traits::duplicate(a);
    simd_t v_x = *((simd_t *)(x + SimdWidth));
    simd_t v_y = *((simd_t *)(y + SimdWidth));
    simd_t result = traits::add(v_y, traits::mul(v_a, v_x));
    for (size_t i = 0; i < n % SimdWidth; ++i) {
      *((y + SimdWidth + i)) = result[i];
    }
  }
}

} // namespace details
} // namespace hpc::l1
#endif
