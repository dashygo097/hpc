#pragma once

#include "../constants.hh"
#include "./simd_impl.hh"

namespace hpc::openmp {
template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vadd_simd_impl(T *__restrict__ A, const T &value, size_t N) {

#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) += simd_t(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v + simd_t(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) += simd_t(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v + simd_t(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) += traits::duplicate(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v + traits::duplicate(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) += traits::duplicate(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v + traits::duplicate(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `+=` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vadd_simd_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) += *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA + vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) += *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));
      simd_t result = vA + vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) += *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA + vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) += *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));
      simd_t result = vA + vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `+=` function for "
               "this platform."
            << std::endl;

#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vsub_simd_impl(T *__restrict__ A, const T &value, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) -= simd_t(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v - simd_t(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) -= simd_t(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v - simd_t(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) -= traits::duplicate(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v - traits::duplicate(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) -= traits::duplicate(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v - traits::duplicate(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `-=` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vsub_simd_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) -= *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA - vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) -= *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));

      simd_t result = vA - vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) -= *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA - vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) -= *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));
      simd_t result = vA - vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `-=` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vmul_simd_impl(T *__restrict__ A, const T &value, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) *= simd_t(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v * simd_t(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) *= simd_t(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v * simd_t(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) *= traits::duplicate(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v * traits::duplicate(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) *= traits::duplicate(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v * traits::duplicate(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `*=` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vmul_simd_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) *= *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA * vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) *= *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));
      simd_t result = vA * vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) *= *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA * vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) *= *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));
      simd_t result = vA * vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }
#else
  std::cerr << "Not Implement SIMD for `*=` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vdiv_simd_impl(T *__restrict__ A, const T &value, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) /= simd_t(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v / simd_t(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) /= simd_t(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v / simd_t(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) /= traits::duplicate(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(A + kSimdWidth));
        simd_t result = v / traits::duplicate(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) /= traits::duplicate(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t v = *((simd_t *)(A + kSimdWidth));
      simd_t result = v / traits::duplicate(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `/=` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vdiv_simd_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) /= *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA / vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) /= *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));
      simd_t result = vA / vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) /= *((simd_t *)(B + i));
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t vA = *((simd_t *)(A + kSimdWidth));
        simd_t vB = *((simd_t *)(B + kSimdWidth));
        simd_t result = vA / vB;
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) /= *((simd_t *)(B + i));
    }
    if (N % kSimdWidth != 0) {
      simd_t vA = *((simd_t *)(A + kSimdWidth));
      simd_t vB = *((simd_t *)(B + kSimdWidth));
      simd_t result = vA / vB;
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `/=` function for "
               "this platform."
            << std::endl;
#endif
}

template <typename T, const size_t kBlockSize, const size_t kSimdWidth>
void vfill_simd_impl(T *__restrict__ A, const T &value, size_t N) {
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) = simd_t(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t result = simd_t(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) = simd_t(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t result = simd_t(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block_idx = 0; block_idx < N; block_idx += kBlockSize) {
        size_t i_end = std::min(block_idx + kBlockSize, kSimdWidth);
        for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
          *((simd_t *)(A + i)) = traits::duplicate(value);
        }
      }
      if (N % kSimdWidth != 0) {
        simd_t result = traits::duplicate(value);
        for (size_t i = 0; i < N % kSimdWidth; ++i) {
          *((A + kSimdWidth + i)) = result[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; i += kSimdWidth) {
      *((simd_t *)(A + i)) = traits::duplicate(value);
    }
    if (N % kSimdWidth != 0) {
      simd_t result = traits::duplicate(value);
      for (size_t i = 0; i < N % kSimdWidth; ++i) {
        *((A + kSimdWidth + i)) = result[i];
      }
    }
  }

#else
  std::cerr << "Not Implement SIMD for `fill` function for "
               "this platform."
            << std::endl;
#endif
}

} // namespace hpc::openmp
