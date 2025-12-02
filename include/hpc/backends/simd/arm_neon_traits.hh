#pragma once

#ifdef SIMD_NEON
#include <arm_neon.h>
#endif

#ifdef SIMD_NEON
namespace hpc::simd {
template <typename T, size_t Width> struct simd_traits;

template <> struct simd_traits<float, 1> {
  using type = float;
  static constexpr size_t width = 1;

  static type load(const float *ptr) { return *ptr; }
  static type duplicate(float v) { return v; }
  static void store(float *ptr, type v) { *ptr = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
}

template <>
struct simd_traits<float, 4> {
  using type = float32x4_t;
  static constexpr size_t width = 4;

  static type load(const float *ptr) { return vld1q_f32(ptr); }
  static type duplicate(float v) { return vdupq_n_f32(v); }
  static void store(float *ptr, type v) { vst1q_f32(ptr, v); }
  static type add(type a, type b) { return vaddq_f32(a, b); }
  static type sub(type a, type b) { return vsubq_f32(a, b); }
  static type mul(type a, type b) { return vmulq_f32(a, b); }
  static type div(type a, type b) { return vmulq_f32(a, vrecpeq_f32(b)); }
};

template <> struct simd_traits<float, 8> {
  using type = float32x4x2_t;
  static constexpr size_t width = 8;

  static type load(const float *ptr) {
    return {vld1q_f32(ptr), vld1q_f32(ptr + 4)};
  }
  static type duplicate(float v) { return {vdupq_n_f32(v), vdupq_n_f32(v)}; }
  static void store(float *ptr, type v) {
    vst1q_f32(ptr, v.val[0]);
    vst1q_f32(ptr + 4, v.val[1]);
  }
  static type add(type a, type b) {
    return {vaddq_f32(a.val[0], b.val[0]), vaddq_f32(a.val[1], b.val[1])};
  }
  static type sub(type a, type b) {
    return {vsubq_f32(a.val[0], b.val[0]), vsubq_f32(a.val[1], b.val[1])};
  }
  static type mul(type a, type b) {
    return {vmulq_f32(a.val[0], b.val[0]), vmulq_f32(a.val[1], b.val[1])};
  }
  static type div(type a, type b) {
    return {vmulq_f32(a.val[0], vrecpeq_f32(b.val[0])),
            vmulq_f32(a.val[1], vrecpeq_f32(b.val[1]))};
  }
}

template <>
struct simd_traits<double, 1> {
  using type = double;
  static constexpr size_t width = 1;

  static type load(const double *ptr) { return *ptr; }
  static type duplicate(double v) { return v; }
  static void store(double *ptr, type v) { *ptr = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
}

template <>
struct simd_traits<double, 2> {
  using type = float64x2_t;
  static constexpr size_t width = 2;

  static type load(const double *ptr) { return vld1q_f64(ptr); }
  static type duplicate(double v) { return vdupq_n_f64(v); }
  static void store(double *ptr, type v) { vst1q_f64(ptr, v); }
  static type add(type a, type b) { return vaddq_f64(a, b); }
  static type sub(type a, type b) { return vsubq_f64(a, b); }
  static type mul(type a, type b) { return vmulq_f64(a, b); }
  static type div(type a, type b) { return vmulq_f64(a, vrecpeq_f64(b)); }
};

template <> struct simd_traits<double, 4> {
  using type = float64x2x2_t;
  static constexpr size_t width = 4;

  static type load(const double *ptr) {
    return {vld1q_f64(ptr), vld1q_f64(ptr + 2)};
  }
  static type duplicate(double v) { return {vdupq_n_f64(v), vdupq_n_f64(v)}; }
  static void store(double *ptr, type v) {
    vst1q_f64(ptr, v.val[0]);
    vst1q_f64(ptr + 2, v.val[1]);
  }
  static type add(type a, type b) {
    return {vaddq_f64(a.val[0], b.val[0]), vaddq_f64(a.val[1], b.val[1])};
  }
  static type sub(type a, type b) {
    return {vsubq_f64(a.val[0], b.val[0]), vsubq_f64(a.val[1], b.val[1])};
  }
  static type mul(type a, type b) {
    return {vmulq_f64(a.val[0], b.val[0]), vmulq_f64(a.val[1], b.val[1])};
  }
  static type div(type a, type b) {
    return {vmulq_f64(a.val[0], vrecpeq_f64(b.val[0])),
            vmulq_f64(a.val[1], vrecpeq_f64(b.val[1]))};
  }
};

template <> struct simd_traits<int, 1> {
  using type = int;
  static constexpr size_t width = 1;

  static type load(const int *ptr) { return *ptr; }
  static type duplicate(int v) { return v; }
  static void store(int *ptr, type v) { *ptr = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
}

template <>
struct simd_traits<int, 4> {
  using type = int32x4_t;
  static constexpr size_t width = 4;

  static type load(const int *ptr) { return vld1q_s32(ptr); }
  static type duplicate(int v) { return vdupq_n_s32(v); }
  static void store(int *ptr, type v) { vst1q_s32(ptr, v); }
  static type add(type a, type b) { return vaddq_s32(a, b); }
  static type sub(type a, type b) { return vsubq_s32(a, b); }
  static type mul(type a, type b) { return vmulq_s32(a, b); }
}

template <>
struct simd_traits<int, 8> {
  using type = int32x4x2_t;
  static constexpr size_t width = 8;

  static type load(const int *ptr) {
    return {vld1q_s32(ptr), vld1q_s32(ptr + 4)};
  }
  static type duplicate(int v) { return {vdupq_n_s32(v), vdupq_n_s32(v)}; }
  static void store(int *ptr, type v) {
    vst1q_s32(ptr, v.val[0]);
    vst1q_s32(ptr + 4, v.val[1]);
  }
  static type add(type a, type b) {
    return {vaddq_s32(a.val[0], b.val[0]), vaddq_s32(a.val[1], b.val[1])};
  }
  static type sub(type a, type b) {
    return {vsubq_s32(a.val[0], b.val[0]), vsubq_s32(a.val[1], b.val[1])};
  }
  static type mul(type a, type b) {
    return {vmulq_s32(a.val[0], b.val[0]), vmulq_s32(a.val[1], b.val[1])};
  }
};

} // namespace hpc::simd
#endif
