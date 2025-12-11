#pragma once

#ifdef SIMD_SSE
#include <cstddef>
#include <immintrin.h>
#endif

#ifdef SIMD_SSE
namespace hpc::l1 {
namespace details {

// axpy
template <>
inline void axpy_simd<float, 4>(const size_t &n, float *__restrict__ dst,
                                const float *__restrict__ src,
                                const float &alpha) {
  // Handle special cases early
  if (alpha == 0.0f) {
    return;
  }

  const size_t simd_end = (n / 4) * 4;

  if (alpha == 1.0f) {
    // Optimized path for alpha = 1: just addition
    for (size_t i = 0; i < simd_end; i += 4) {
      __m128 vx = _mm_loadu_ps(src + i);
      __m128 vy = _mm_loadu_ps(dst + i);
      __m128 result = _mm_add_ps(vx, vy);
      _mm_storeu_ps(dst + i, result);
    }
  } else if (alpha == -1.0f) {
    // Optimized path for alpha = -1: subtraction
    for (size_t i = 0; i < simd_end; i += 4) {
      __m128 vx = _mm_loadu_ps(src + i);
      __m128 vy = _mm_loadu_ps(dst + i);
      __m128 result = _mm_sub_ps(vy, vx);
      _mm_storeu_ps(dst + i, result);
    }
  } else {
    // General case: FMA if available, otherwise multiply-add
    __m128 v_alpha = _mm_set1_ps(alpha);

#ifdef __FMA__
    // Use FMA for better performance and accuracy
    for (size_t i = 0; i < simd_end; i += 4) {
      __m128 vx = _mm_loadu_ps(src + i);
      __m128 vy = _mm_loadu_ps(dst + i);
      __m128 result = _mm_fmadd_ps(v_alpha, vx, vy);
      _mm_storeu_ps(dst + i, result);
    }
#else
    // Fallback to separate multiply and add
    for (size_t i = 0; i < simd_end; i += 4) {
      __m128 vx = _mm_loadu_ps(src + i);
      __m128 vy = _mm_loadu_ps(dst + i);
      __m128 scaled = _mm_mul_ps(v_alpha, vx);
      __m128 result = _mm_add_ps(scaled, vy);
      _mm_storeu_ps(dst + i, result);
    }
#endif
  }

  // Scalar tail for remaining elements
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] += alpha * src[i];
  }
}

// SSE double AXPY: y = alpha * x + y
template <>
inline void axpy_simd<double, 2>(const size_t &n, double *__restrict__ dst,
                                 const double *__restrict__ src,
                                 const double &alpha) {
  if (alpha == 0.0) {
    return;
  }

  const size_t simd_end = (n / 2) * 2;

  if (alpha == 1.0) {
    for (size_t i = 0; i < simd_end; i += 2) {
      __m128d vx = _mm_loadu_pd(src + i);
      __m128d vy = _mm_loadu_pd(dst + i);
      __m128d result = _mm_add_pd(vx, vy);
      _mm_storeu_pd(dst + i, result);
    }
  } else if (alpha == -1.0) {
    for (size_t i = 0; i < simd_end; i += 2) {
      __m128d vx = _mm_loadu_pd(src + i);
      __m128d vy = _mm_loadu_pd(dst + i);
      __m128d result = _mm_sub_pd(vy, vx);
      _mm_storeu_pd(dst + i, result);
    }
  } else {
    __m128d v_alpha = _mm_set1_pd(alpha);

#ifdef __FMA__
    for (size_t i = 0; i < simd_end; i += 2) {
      __m128d vx = _mm_loadu_pd(src + i);
      __m128d vy = _mm_loadu_pd(dst + i);
      __m128d result = _mm_fmadd_pd(v_alpha, vx, vy);
      _mm_storeu_pd(dst + i, result);
    }
#else
    for (size_t i = 0; i < simd_end; i += 2) {
      __m128d vx = _mm_loadu_pd(src + i);
      __m128d vy = _mm_loadu_pd(dst + i);
      __m128d scaled = _mm_mul_pd(v_alpha, vx);
      __m128d result = _mm_add_pd(scaled, vy);
      _mm_storeu_pd(dst + i, result);
    }
#endif
  }

  // Scalar tail
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] += alpha * src[i];
  }
}

// Unrolled version for better throughput on large vectors
template <>
inline void axpy_simd<float, 16>(const size_t &n, float *__restrict__ dst,
                                 const float *__restrict__ src,
                                 const float &alpha) {
  if (alpha == 0.0f) {
    return;
  }

  const size_t simd_end = (n / 16) * 16;

  if (alpha == 1.0f) {
    for (size_t i = 0; i < simd_end; i += 16) {
      __m128 vx0 = _mm_loadu_ps(src + i);
      __m128 vx1 = _mm_loadu_ps(src + i + 4);
      __m128 vx2 = _mm_loadu_ps(src + i + 8);
      __m128 vx3 = _mm_loadu_ps(src + i + 12);

      __m128 vy0 = _mm_loadu_ps(dst + i);
      __m128 vy1 = _mm_loadu_ps(dst + i + 4);
      __m128 vy2 = _mm_loadu_ps(dst + i + 8);
      __m128 vy3 = _mm_loadu_ps(dst + i + 12);

      _mm_storeu_ps(dst + i, _mm_add_ps(vx0, vy0));
      _mm_storeu_ps(dst + i + 4, _mm_add_ps(vx1, vy1));
      _mm_storeu_ps(dst + i + 8, _mm_add_ps(vx2, vy2));
      _mm_storeu_ps(dst + i + 12, _mm_add_ps(vx3, vy3));
    }
  } else {
    __m128 v_alpha = _mm_set1_ps(alpha);

#ifdef __FMA__
    for (size_t i = 0; i < simd_end; i += 16) {
      __m128 vx0 = _mm_loadu_ps(src + i);
      __m128 vx1 = _mm_loadu_ps(src + i + 4);
      __m128 vx2 = _mm_loadu_ps(src + i + 8);
      __m128 vx3 = _mm_loadu_ps(src + i + 12);

      __m128 vy0 = _mm_loadu_ps(dst + i);
      __m128 vy1 = _mm_loadu_ps(dst + i + 4);
      __m128 vy2 = _mm_loadu_ps(dst + i + 8);
      __m128 vy3 = _mm_loadu_ps(dst + i + 12);

      _mm_storeu_ps(dst + i, _mm_fmadd_ps(v_alpha, vx0, vy0));
      _mm_storeu_ps(dst + i + 4, _mm_fmadd_ps(v_alpha, vx1, vy1));
      _mm_storeu_ps(dst + i + 8, _mm_fmadd_ps(v_alpha, vx2, vy2));
      _mm_storeu_ps(dst + i + 12, _mm_fmadd_ps(v_alpha, vx3, vy3));
    }
#else
    for (size_t i = 0; i < simd_end; i += 16) {
      __m128 vx0 = _mm_loadu_ps(src + i);
      __m128 vx1 = _mm_loadu_ps(src + i + 4);
      __m128 vx2 = _mm_loadu_ps(src + i + 8);
      __m128 vx3 = _mm_loadu_ps(src + i + 12);

      __m128 vy0 = _mm_loadu_ps(dst + i);
      __m128 vy1 = _mm_loadu_ps(dst + i + 4);
      __m128 vy2 = _mm_loadu_ps(dst + i + 8);
      __m128 vy3 = _mm_loadu_ps(dst + i + 12);

      _mm_storeu_ps(dst + i, _mm_add_ps(_mm_mul_ps(v_alpha, vx0), vy0));
      _mm_storeu_ps(dst + i + 4, _mm_add_ps(_mm_mul_ps(v_alpha, vx1), vy1));
      _mm_storeu_ps(dst + i + 8, _mm_add_ps(_mm_mul_ps(v_alpha, vx2), vy2));
      _mm_storeu_ps(dst + i + 12, _mm_add_ps(_mm_mul_ps(v_alpha, vx3), vy3));
    }
#endif
  }

  // Scalar tail
  for (size_t i = simd_end; i < n; ++i) {
    dst[i] += alpha * src[i];
  }
}

} // namespace details
} // namespace hpc::l1
#endif
