#pragma once

#ifdef ENABLE_OPENMP
#include "../backends/backends.hh"
#include <algorithm>
#endif

#ifdef ENABLE_OPENMP
namespace hpc::l1 {
namespace details {

template <typename T>
inline void vadd_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] += scalar;
  }
}

template <typename T>
inline void vadd_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    dst[i] += src[i];
  }
}

template <typename T>
inline void vsub_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] -= scalar;
}

template <typename T>
inline void vsub_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] -= src[i];
}

template <typename T>
inline void vmul_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] *= scalar;
}

template <typename T>
inline void vmul_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] *= src[i];
}

template <typename T>
inline void vdiv_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] /= scalar;
}

template <typename T>
inline void vdiv_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] /= src[i];
}

template <typename T>
inline void vfill_omp(T *__restrict__ dst, const T &value, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    dst[i] = value;
}

template <typename T, const size_t TileSize>
inline void vadd_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] += scalar;
    }
  }
}

template <typename T, const size_t TileSize>
inline void vadd_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] += src[i];
    }
  }
}

template <typename T, const size_t TileSize>
inline void vsub_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] -= scalar;
    }
  }
}

template <typename T, const size_t TileSize>
inline void vsub_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] -= src[i];
    }
  }
}

template <typename T, const size_t TileSize>
inline void vmul_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] *= scalar;
    }
  }
}

template <typename T, const size_t TileSize>
inline void vmul_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] *= src[i];
    }
  }
}

template <typename T, const size_t TileSize>
inline void vdiv_omp(T *__restrict__ dst, const T &scalar, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] /= scalar;
    }
  }
}

template <typename T, const size_t TileSize>
inline void vdiv_omp(T *__restrict__ dst, const T *__restrict__ src, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] /= src[i];
    }
  }
}

template <typename T, const size_t TileSize>
inline void vfill_omp(T *__restrict__ dst, const T &value, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      dst[i] = value;
    }
  }
}

template <typename T, const size_t TileSize>
inline void axpy_omp(T *__restrict__ y, const T *__restrict__ x, const T &a,
                     size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t block = 0; block < n; block += TileSize) {
    size_t end = std::min(block + TileSize, n);
    for (size_t i = block; i < end; ++i) {
      y[i] += a * x[i];
    }
  }
}

} // namespace details
} // namespace hpc::l1
#endif
