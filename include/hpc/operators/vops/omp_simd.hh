#pragma once

#include "../../backends/backends.hh"
#include <algorithm>

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
namespace hpc::op {
namespace details {

template <typename T, const size_t TileSize, const size_t SimdWidth>
inline void vadd_omp_simd(T *__restrict__ dst, const T &scaler, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) += traits::duplicate(scaler);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::add(v, traits::duplicate(scaler));
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
inline void vsub_omp_simd(T *__restrict__ dst, const T &scaler, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) -= traits::duplicate(scaler);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::sub(v, traits::duplicate(scaler));
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
inline void vmul_omp_simd(T *__restrict__ dst, const T &scaler, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) *= traits::duplicate(scaler);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::mul(v, traits::duplicate(scaler));
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
inline void vdiv_omp_simd(T *__restrict__ dst, const T &scaler, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

#pragma omp parallel for schedule(static)
  for (size_t block_idx = 0; block_idx < n; block_idx += TileSize) {
    size_t i_end = std::min(block_idx + TileSize, SimdWidth);
    for (size_t i = block_idx; i < i_end; i += SimdWidth) {
      *((simd_t *)(dst + i)) /= traits::duplicate(scaler);
    }
  }
  if (n % SimdWidth != 0) {
    simd_t v = *((simd_t *)(dst + SimdWidth));
    simd_t result = traits::div(v, traits::duplicate(scaler));
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

} // namespace details
} // namespace hpc::op
#endif
