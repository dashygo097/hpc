#pragma once

#ifdef SIMD_NEON
#include <arm_neon.h>
#endif

// FIXME: overhead is detected even after inlining these traits' methods.
#ifdef SIMD_NEON
#define SIMD_LOAD(traits, ptr) traits::load(ptr)
#define SIMD_STORE(traits, ptr, value) traits::store(ptr, value)
#define SIMD_DUP(traits, value) traits::duplicate(value)
#define SIMD_ADD(traits, a, b) traits::add(a, b)
#define SIMD_SUB(traits, a, b) traits::sub(a, b)
#define SIMD_MUL(traits, a, b) traits::mul(a, b)
#define SIMD_FMA(traits, a, b, c) traits::add(traits::mul(a, b), c)
#define SIMD_DIV(traits, a, b) traits::div(a, b)
#endif

#ifdef SIMD_NEON
namespace hpc::simd {
template <typename T, size_t Width> struct simd_traits;

template <> struct simd_traits<float, 1> {
  using type = float;
  static constexpr size_t width = 1;

  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return *ptr;
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return v;
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    *ptr = v;
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
}

template <>
struct simd_traits<float, 4> {
  using type = float32x4_t;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return vld1q_f32(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return vdupq_n_f32(v);
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    vst1q_f32(ptr, v);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return vaddq_f32(a, b);
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return vsubq_f32(a, b);
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return vmulq_f32(a, b);
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return vmulq_f32(a, vrecpeq_f32(b));
  }
};

template <> struct simd_traits<float, 8> {
  using type = float32x4x2_t;
  static constexpr size_t width = 8;

  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return {vld1q_f32(ptr), vld1q_f32(ptr + 4)};
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return {vdupq_n_f32(v), vdupq_n_f32(v)};
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    vst1q_f32(ptr, v.val[0]);
    vst1q_f32(ptr + 4, v.val[1]);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return {vaddq_f32(a.val[0], b.val[0]), vaddq_f32(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return {vsubq_f32(a.val[0], b.val[0]), vsubq_f32(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return {vmulq_f32(a.val[0], b.val[0]), vmulq_f32(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return {vmulq_f32(a.val[0], vrecpeq_f32(b.val[0])),
            vmulq_f32(a.val[1], vrecpeq_f32(b.val[1]))};
  }
}

template <>
struct simd_traits<double, 1> {
  using type = double;
  static constexpr size_t width = 1;

  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return *ptr;
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return v;
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    *ptr = v;
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
}

template <>
struct simd_traits<double, 2> {
  using type = float64x2_t;
  static constexpr size_t width = 2;

  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return vld1q_f64(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return vdupq_n_f64(v);
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    vst1q_f64(ptr, v);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return vaddq_f64(a, b);
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return vsubq_f64(a, b);
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return vmulq_f64(a, b);
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return vmulq_f64(a, vrecpeq_f64(b));
  }
};

template <> struct simd_traits<double, 4> {
  using type = float64x2x2_t;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return {vld1q_f64(ptr), vld1q_f64(ptr + 2)};
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return {vdupq_n_f64(v), vdupq_n_f64(v)};
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    vst1q_f64(ptr, v.val[0]);
    vst1q_f64(ptr + 2, v.val[1]);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return {vaddq_f64(a.val[0], b.val[0]), vaddq_f64(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return {vsubq_f64(a.val[0], b.val[0]), vsubq_f64(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return {vmulq_f64(a.val[0], b.val[0]), vmulq_f64(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return {vmulq_f64(a.val[0], vrecpeq_f64(b.val[0])),
            vmulq_f64(a.val[1], vrecpeq_f64(b.val[1]))};
  }
};

template <> struct simd_traits<int, 1> {
  using type = int;
  static constexpr size_t width = 1;

  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return *ptr;
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return v;
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    *ptr = v;
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
}

template <>
struct simd_traits<int, 4> {
  using type = int32x4_t;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return vld1q_s32(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return vdupq_n_s32(v);
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    vst1q_s32(ptr, v);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return vaddq_s32(a, b);
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return vsubq_s32(a, b);
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return vmulq_s32(a, b);
  }
}

template <>
struct simd_traits<int, 8> {
  using type = int32x4x2_t;
  static constexpr size_t width = 8;

  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return {vld1q_s32(ptr), vld1q_s32(ptr + 4)};
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return {vdupq_n_s32(v), vdupq_n_s32(v)};
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    vst1q_s32(ptr, v.val[0]);
    vst1q_s32(ptr + 4, v.val[1]);
  }
  __attribute__((always_inline)) static inline type add(type a, type b) {
    return {vaddq_s32(a.val[0], b.val[0]), vaddq_s32(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type sub(type a, type b) {
    return {vsubq_s32(a.val[0], b.val[0]), vsubq_s32(a.val[1], b.val[1])};
  }
  __attribute__((always_inline)) static inline type mul(type a, type b) {
    return {vmulq_s32(a.val[0], b.val[0]), vmulq_s32(a.val[1], b.val[1])};
  }
};

} // namespace hpc::simd
#endif
