#pragma once

#ifdef SIMD_NEON
#include <arm_neon.h>
#include <cstddef>
#endif

#ifdef SIMD_NEON
#define SIMD_LOAD(traits, ptr) traits::load(ptr)
#define SIMD_STORE(traits, ptr, value) traits::store(ptr, value)
#define SIMD_DUP(traits, value) traits::duplicate(value)
#define SIMD_SUM(traits, value) traits::sum(value)
#define SIMD_ABS(traits, value) traits::abs(value)
#define SIMD_L1_NORM(traits, value) traits::l1_norm(value)
#define SIMD_ADD(traits, a, b) traits::add(a, b)
#define SIMD_SUB(traits, a, b) traits::sub(a, b)
#define SIMD_MUL(traits, a, b) traits::mul(a, b)
#define SIMD_FMA(traits, a, b, c) traits::fma(a, b, c)
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
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    *ptr = v;
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return v;
  }
  __attribute__((always_inline)) static inline float sum(type v) { return v; }
  __attribute__((always_inline)) static inline type abs(type v) {
    return v > 0 ? v : -v;
  }
  __attribute__((always_inline)) static inline float l1_norm(type v) {
    return v > 0 ? v : -v;
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
    return a * b + c;
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return a / b;
  }
};

template <> struct simd_traits<float, 4> {
  using type = float32x4_t;
  static constexpr size_t width = 4;
  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return vld1q_f32(ptr);
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    vst1q_f32(ptr, v);
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return vdupq_n_f32(v);
  }
  __attribute__((always_inline)) static inline float sum(type v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    float32x2_t t = vpadd_f32(s, s);
    return vget_lane_f32(t, 0);
#endif
  }
  __attribute__((always_inline)) static inline type abs(type v) {
    return vabsq_f32(v);
  }
  __attribute__((always_inline)) static inline float l1_norm(type v) {
    return sum(abs(v));
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FMA)
    return vfmaq_f32(c, a, b);
#else
    return vaddq_f32(vmulq_f32(a, b), c);
#endif
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    float32x4_t r = vrecpeq_f32(b);
    r = vmulq_f32(r, vrecpsq_f32(b, r));
    r = vmulq_f32(r, vrecpsq_f32(b, r));
    return vmulq_f32(a, r);
  }
};

template <> struct simd_traits<float, 8> {
  using type = float32x4x2_t;
  static constexpr size_t width = 8;
  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return {vld1q_f32(ptr), vld1q_f32(ptr + 4)};
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    vst1q_f32(ptr, v.val[0]);
    vst1q_f32(ptr + 4, v.val[1]);
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return {vdupq_n_f32(v), vdupq_n_f32(v)};
  }
  __attribute__((always_inline)) static inline float sum(type v) {
#if defined(__aarch64__)
    return vaddvq_f32(v.val[0]) + vaddvq_f32(v.val[1]);
#else
    float32x2_t s0 = vadd_f32(vget_low_f32(v.val[0]), vget_high_f32(v.val[0]));
    float32x2_t t0 = vpadd_f32(s0, s0);
    float32x2_t s1 = vadd_f32(vget_low_f32(v.val[1]), vget_high_f32(v.val[1]));
    float32x2_t t1 = vpadd_f32(s1, s1);
    return vget_lane_f32(t0, 0) + vget_lane_f32(t1, 0);
#endif
  }
  __attribute__((always_inline)) static inline type abs(type v) {
    return {vabsq_f32(v.val[0]), vabsq_f32(v.val[1])};
  }
  __attribute__((always_inline)) static inline float l1_norm(type v) {
    return sum(abs(v));
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FMA)
    return {vfmaq_f32(c.val[0], a.val[0], b.val[0]),
            vfmaq_f32(c.val[1], a.val[1], b.val[1])};
#else
    return {vaddq_f32(vmulq_f32(a.val[0], b.val[0]), c.val[0]),
            vaddq_f32(vmulq_f32(a.val[1], b.val[1]), c.val[1])};
#endif
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    float32x4_t r0 = vrecpeq_f32(b.val[0]);
    r0 = vmulq_f32(r0, vrecpsq_f32(b.val[0], r0));
    r0 = vmulq_f32(r0, vrecpsq_f32(b.val[0], r0));
    float32x4_t r1 = vrecpeq_f32(b.val[1]);
    r1 = vmulq_f32(r1, vrecpsq_f32(b.val[1], r1));
    r1 = vmulq_f32(r1, vrecpsq_f32(b.val[1], r1));
    return {vmulq_f32(a.val[0], r0), vmulq_f32(a.val[1], r1)};
  }
};

template <> struct simd_traits<double, 1> {
  using type = double;
  static constexpr size_t width = 1;
  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return *ptr;
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    *ptr = v;
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return v;
  }
  __attribute__((always_inline)) static inline double sum(type v) { return v; }
  __attribute__((always_inline)) static inline type abs(type v) {
    return v > 0 ? v : -v;
  }
  __attribute__((always_inline)) static inline double l1_norm(type v) {
    return v > 0 ? v : -v;
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
    return a * b + c;
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    return a / b;
  }
};

template <> struct simd_traits<double, 2> {
  using type = float64x2_t;
  static constexpr size_t width = 2;
  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return vld1q_f64(ptr);
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    vst1q_f64(ptr, v);
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return vdupq_n_f64(v);
  }
  __attribute__((always_inline)) static inline double sum(type v) {
#if defined(__aarch64__)
    return vaddvq_f64(v);
#else
    float64x1_t s = vadd_f64(vget_low_f64(v), vget_high_f64(v));
    return vget_lane_f64(s, 0);
#endif
  }
  __attribute__((always_inline)) static inline type abs(type v) {
    return vabsq_f64(v);
  }
  __attribute__((always_inline)) static inline double l1_norm(type v) {
    return sum(abs(v));
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FMA)
    return vfmaq_f64(c, a, b);
#else
    return vaddq_f64(vmulq_f64(a, b), c);
#endif
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    alignas(16) double aa[2], bb[2], rr[2];
    vst1q_f64(aa, a);
    vst1q_f64(bb, b);
    rr[0] = aa[0] / bb[0];
    rr[1] = aa[1] / bb[1];
    return vld1q_f64(rr);
  }
};

template <> struct simd_traits<double, 4> {
  using type = float64x2x2_t;
  static constexpr size_t width = 4;
  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return {vld1q_f64(ptr), vld1q_f64(ptr + 2)};
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    vst1q_f64(ptr, v.val[0]);
    vst1q_f64(ptr + 2, v.val[1]);
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return {vdupq_n_f64(v), vdupq_n_f64(v)};
  }
  __attribute__((always_inline)) static inline double sum(type v) {
#if defined(__aarch64__)
    return vaddvq_f64(v.val[0]) + vaddvq_f64(v.val[1]);
#else
    float64x1_t s0 = vadd_f64(vget_low_f64(v.val[0]), vget_high_f64(v.val[0]));
    float64x1_t s1 = vadd_f64(vget_low_f64(v.val[1]), vget_high_f64(v.val[1]));
    return vget_lane_f64(s0, 0) + vget_lane_f64(s1, 0);
#endif
  }
  __attribute__((always_inline)) static inline type abs(type v) {
    return {vabsq_f64(v.val[0]), vabsq_f64(v.val[1])};
  }
  __attribute__((always_inline)) static inline double l1_norm(type v) {
    return sum(abs(v));
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FMA)
    return {vfmaq_f64(c.val[0], a.val[0], b.val[0]),
            vfmaq_f64(c.val[1], a.val[1], b.val[1])};
#else
    return {vaddq_f64(vmulq_f64(a.val[0], b.val[0]), c.val[0]),
            vaddq_f64(vmulq_f64(a.val[1], b.val[1]), c.val[1])};
#endif
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    alignas(16) double aa[4], bb[4], rr[4];
    vst1q_f64(aa, a.val[0]);
    vst1q_f64(aa + 2, a.val[1]);
    vst1q_f64(bb, b.val[0]);
    vst1q_f64(bb + 2, b.val[1]);
    rr[0] = aa[0] / bb[0];
    rr[1] = aa[1] / bb[1];
    rr[2] = aa[2] / bb[2];
    rr[3] = aa[3] / bb[3];
    return {vld1q_f64(rr), vld1q_f64(rr + 2)};
  }
};

template <> struct simd_traits<int, 1> {
  using type = int;
  static constexpr size_t width = 1;
  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return *ptr;
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    *ptr = v;
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return v;
  }
  __attribute__((always_inline)) static inline int sum(type v) { return v; }
  __attribute__((always_inline)) static inline type abs(type v) {
    return v > 0 ? v : -v;
  }
  __attribute__((always_inline)) static inline int l1_norm(type v) {
    return v > 0 ? v : -v;
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

template <> struct simd_traits<int, 4> {
  using type = int32x4_t;
  static constexpr size_t width = 4;
  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return vld1q_s32(ptr);
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    vst1q_s32(ptr, v);
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return vdupq_n_s32(v);
  }
  __attribute__((always_inline)) static inline int sum(type v) {
#if defined(__aarch64__)
    return vaddvq_s32(v);
#else
    int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    int32x2_t t = vpadd_s32(s, s);
    return vget_lane_s32(t, 0);
#endif
  }
  __attribute__((always_inline)) static inline type abs(type v) {
    return vabsq_s32(v);
  }
  __attribute__((always_inline)) static inline int l1_norm(type v) {
    return sum(abs(v));
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
    return vaddq_s32(vmulq_s32(a, b), c);
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    alignas(16) int aa[4], bb[4], rr[4];
    vst1q_s32(aa, a);
    vst1q_s32(bb, b);
    rr[0] = aa[0] / bb[0];
    rr[1] = aa[1] / bb[1];
    rr[2] = aa[2] / bb[2];
    rr[3] = aa[3] / bb[3];
    return vld1q_s32(rr);
  }
};

template <> struct simd_traits<int, 8> {
  using type = int32x4x2_t;
  static constexpr size_t width = 8;
  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return {vld1q_s32(ptr), vld1q_s32(ptr + 4)};
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    vst1q_s32(ptr, v.val[0]);
    vst1q_s32(ptr + 4, v.val[1]);
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return {vdupq_n_s32(v), vdupq_n_s32(v)};
  }
  __attribute__((always_inline)) static inline int sum(type v) {
#if defined(__aarch64__)
    return vaddvq_s32(v.val[0]) + vaddvq_s32(v.val[1]);
#else
    int32x2_t s0 = vadd_s32(vget_low_s32(v.val[0]), vget_high_s32(v.val[0]));
    int32x2_t t0 = vpadd_s32(s0, s0);
    int32x2_t s1 = vadd_s32(vget_low_s32(v.val[1]), vget_high_s32(v.val[1]));
    int32x2_t t1 = vpadd_s32(s1, s1);
    return vget_lane_s32(t0, 0) + vget_lane_s32(t1, 0);
#endif
  }
  __attribute__((always_inline)) static inline type abs(type v) {
    return {vabsq_s32(v.val[0]), vabsq_s32(v.val[1])};
  }
  __attribute__((always_inline)) static inline int l1_norm(type v) {
    return sum(abs(v));
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
  __attribute__((always_inline)) static inline type fma(type a, type b,
                                                        type c) {
    return {vaddq_s32(vmulq_s32(a.val[0], b.val[0]), c.val[0]),
            vaddq_s32(vmulq_s32(a.val[1], b.val[1]), c.val[1])};
  }
  __attribute__((always_inline)) static inline type div(type a, type b) {
    alignas(16) int aa[8], bb[8], rr[8];
    vst1q_s32(aa, a.val[0]);
    vst1q_s32(aa + 4, a.val[1]);
    vst1q_s32(bb, b.val[0]);
    vst1q_s32(bb + 4, b.val[1]);
    for (int i = 0; i < 8; ++i)
      rr[i] = aa[i] / bb[i];
    return {vld1q_s32(rr), vld1q_s32(rr + 4)};
  }
};

} // namespace hpc::simd
#endif
