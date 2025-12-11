#pragma once

#ifdef SIMD_NEON
#include <arm_neon.h>
#endif

#ifdef SIMD_NEON
#define SIMD_LOAD(traits, ptr) traits::load(ptr)
#define SIMD_STORE(traits, ptr, value) traits::store(ptr, value)
#define SIMD_DUP(traits, value) traits::duplicate(value)
#define SIMD_SUM(traits, value) traits::sum(value)
#define SIMD_ADD(traits, a, b) traits::add(a, b)
#define SIMD_SUB(traits, a, b) traits::sub(a, b)
#define SIMD_MUL(traits, a, b) traits::mul(a, b)
#define SIMD_FMA(traits, a, b, c) traits::add(traits::mul(a, b), c)
#define SIMD_DIV(traits, a, b) traits::div(a, b)
#endif

// f32
#ifdef SIMD_NEON
#define SIMD_F32_LOAD(ptr) vld1q_f32(ptr)
#define SIMD_F32_STORE(ptr, value) vst1q_f32(ptr, value)
#define SIMD_F32_DUP(value) vdupq_n_f32(value)
#define SIMD_F32_SUM(value)                                                    \
  ({                                                                           \
    float32x2_t sum1 = vadd_f32(vget_low_f32(value), vget_high_f32(value));    \
    float32x2_t sum2 = vpadd_f32(sum1, sum1);                                  \
    vget_lane_f32(sum2, 0);                                                    \
  })
#define SIMD_F32_ADD(a, b) vaddq_f32(a, b)
#define SIMD_F32_SUB(a, b) vsubq_f32(a, b)
#define SIMD_F32_MUL(a, b) vmulq_f32(a, b)
#define SIMD_F32_FMA(a, b, c) vaddq_f32(vmulq_f32(a, b), c)
#define SIMD_F32_DIV(a, b) vmulq_f32(a, vrecpeq_f32(b))
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
  __attribute__((always_inline)) static inline float sum(type v) { return v; }
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
  __attribute__((always_inline)) static inline float sum(type v) {
    float32x2_t sum1 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    float32x2_t sum2 = vpadd_f32(sum1, sum1);
    return vget_lane_f32(sum2, 0);
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
  __attribute__((always_inline)) static inline float sum(type v) {
    float32x2_t sum1 =
        vadd_f32(vget_low_f32(v.val[0]), vget_high_f32(v.val[0]));
    float32x2_t sum2 =
        vadd_f32(vget_low_f32(v.val[1]), vget_high_f32(v.val[1]));
    float32x2_t total = vadd_f32(sum1, sum2);
    float32x2_t final_sum = vpadd_f32(total, total);
    return vget_lane_f32(final_sum, 0);
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
  __attribute__((always_inline)) static inline double sum(type v) { return v; }
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
  __attribute__((always_inline)) static inline double sum(type v) {
    float64x1_t sum1 = vadd_f64(vget_low_f64(v), vget_high_f64(v));
    return vget_lane_f64(sum1, 0);
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
  __attribute__((always_inline)) static inline double sum(type v) {
    float64x1_t sum1 =
        vadd_f64(vget_low_f64(v.val[0]), vget_high_f64(v.val[0]));
    float64x1_t sum2 =
        vadd_f64(vget_low_f64(v.val[1]), vget_high_f64(v.val[1]));
    float64x1_t total = vadd_f64(sum1, sum2);
    return vget_lane_f64(total, 0);
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
  __attribute__((always_inline)) static inline int sum(type v) { return v; }
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
  __attribute__((always_inline)) static inline int sum(type v) {
    int32x2_t sum1 = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    int32x2_t sum2 = vpadd_s32(sum1, sum1);
    return vget_lane_s32(sum2, 0);
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
  __attribute__((always_inline)) static inline int sum(type v) {
    int32x2_t sum1 = vadd_s32(vget_low_s32(v.val[0]), vget_high_s32(v.val[0]));
    int32x2_t sum2 = vadd_s32(vget_low_s32(v.val[1]), vget_high_s32(v.val[1]));
    int32x2_t total = vadd_s32(sum1, sum2);
    int32x2_t final_sum = vpadd_s32(total, total);
    return vget_lane_s32(final_sum, 0);
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
