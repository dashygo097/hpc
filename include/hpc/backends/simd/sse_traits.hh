#pragma once

#ifdef SIMD_SSE
#include <xmmintrin.h>
#endif

// FIXME: overhead is detected even after inlining these traits' methods.
#ifdef SIMD_SSE
#define SIMD_LOAD(traits, ptr) traits::load(ptr)
#define SIMD_STORE(traits, ptr, value) traits::store(ptr, value)
#define SIMD_DUP(traits, value) traits::duplicate(value)
#define SIMD_ADD(traits, a, b) traits::add(a, b)
#define SIMD_SUB(traits, a, b) traits::sub(a, b)
#define SIMD_MUL(traits, a, b) traits::mul(a, b)
#define SIMD_FMA(traits, a, b, c) traits::add(traits::mul(a, b), c)
#define SIMD_DIV(traits, a, b) traits::div(a, b)
#endif

#ifdef SIMD_SSE
namespace hpc::simd {
template <typename T, size_t Width> struct simd_traits;

template <> struct simd_traits<float, 1> {
  using type = float;
  static constexpr size_t width = 1;

  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return ptr[0];
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return v;
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    ptr[0] = v;
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return a + b;
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return a - b;
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return a * b;
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return a / b;
  }
};

template <> struct simd_traits<float, 4> {
  using type = __m128;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return _mm_loadu_ps(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return _mm_set1_ps(v);
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    _mm_storeu_ps(ptr, v);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return _mm_add_ps(a, b);
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return _mm_sub_ps(a, b);
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return _mm_mul_ps(a, b);
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return _mm_div_ps(a, b);
  }
};

template <> struct simd_traits<double, 1> {
  using type = double;
  static constexpr size_t width = 1;

  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return ptr[0];
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return v;
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    ptr[0] = v;
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return a + b;
  }
};

template <> struct simd_traits<double, 2> {
  using type = __m128d;
  static constexpr size_t width = 2;

  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return _mm_loadu_pd(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return _mm_set1_pd(v);
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    _mm_storeu_pd(ptr, v);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return _mm_add_pd(a, b);
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return _mm_sub_pd(a, b);
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return _mm_mul_pd(a, b);
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return _mm_div_pd(a, b);
  }
};

template <> struct simd_traits<int, 1> {
  using type = int;
  static constexpr size_t width = 1;

  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return ptr[0];
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return v;
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    ptr[0] = v;
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return a + b;
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return a - b;
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return a * b;
  }
};

template <> struct simd_traits<int, 4> {
  using type = __m128i;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return _mm_loadu_si128((__m128i const *)ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return _mm_set1_epi32(v);
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    _mm_storeu_si128((__m128i *)ptr, v);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return _mm_add_epi32(a, b);
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return _mm_sub_epi32(a, b);
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    alignas(16) int a_arr[4], b_arr[4], res_arr[4];
    _mm_storeu_si128((__m128i *)a_arr, a);
    _mm_storeu_si128((__m128i *)b_arr, b);
    for (size_t i = 0; i < 4; ++i) {
      res_arr[i] = a_arr[i] * b_arr[i];
    }
    return _mm_loadu_si128((__m128i *)res_arr);
  }
};

} // namespace hpc::simd
#endif
