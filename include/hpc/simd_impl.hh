#pragma once

namespace hpc {
template <typename T, size_t Width> struct simd_type;
template <typename T, size_t Width> struct neon_traits;

#ifdef ENABLE_SIMD
#if defined(__APPLE__)

template <> struct simd_type<float, 1> {
  using type = float;
  static constexpr size_t width = 1;
};
template <> struct simd_type<float, 2> {
  using type = simd_float2;
  static constexpr size_t width = 2;
};
template <> struct simd_type<float, 4> {
  using type = simd_float4;
  static constexpr size_t width = 4;
};
template <> struct simd_type<float, 8> {
  using type = simd_float8;
  static constexpr size_t width = 8;
};
template <> struct simd_type<double, 1> {
  using type = double;
  static constexpr size_t width = 1;
};
template <> struct simd_type<double, 2> {
  using type = simd_double2;
  static constexpr size_t width = 2;
};
template <> struct simd_type<double, 4> {
  using type = simd_double4;
  static constexpr size_t width = 4;
};
template <> struct simd_type<int, 1> {
  using type = int;
  static constexpr size_t width = 1;
};
template <> struct simd_type<int, 4> {
  using type = simd_int4;
  static constexpr size_t width = 4;
};
template <> struct simd_type<int, 8> {
  using type = simd_int8;
  static constexpr size_t width = 8;
};

#elif defined(__ARM_NEON)
template <> struct simd_type<float, 1> {
  using type = float;
  static constexpr size_t width = 1;
};
template <> struct neon_traits<float, 1> {
  using type = float;
  static constexpr size_t width = 1;

  static type load(const float *ptr) { return *ptr; }
  static type duplicate(float v) { return v; }
  static void store(float *ptr, type v) { *ptr = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
};

template <> struct simd_type<float, 4> {
  using type = float32x4_t;
  static constexpr size_t width = 4;
};
template <> struct neon_traits<float, 4> {
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

template <> struct simd_type<float, 8> {
  using type = float32x4x2_t;
  static constexpr size_t width = 8;
};
template <> struct neon_traits<float, 8> {
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
};

template <> struct simd_type<double, 1> {
  using type = double;
  static constexpr size_t width = 1;
};
template <> struct neon_traits<double, 1> {
  using type = double;
  static constexpr size_t width = 1;
  static type load(const double *ptr) { return *ptr; }
  static type duplicate(double v) { return v; }
  static void store(double *ptr, type v) { *ptr = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
};

template <> struct simd_type<double, 2> {
  using type = float64x2_t;
  static constexpr size_t width = 2;
};
template <> struct neon_traits<double, 2> {
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

template <> struct simd_type<double, 4> {
  using type = float64x2x2_t;
  static constexpr size_t width = 4;
};

template <> struct neon_traits<double, 4> {
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

template <> struct simd_type<int, 1> {
  using type = int;
  static constexpr size_t width = 1;
};
template <> struct neon_traits<int, 1> {
  using type = int;
  static constexpr size_t width = 1;
  static type load(const int *ptr) { return *ptr; }
  static type duplicate(int v) { return v; }
  static void store(int *ptr, type v) { *ptr = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
};

template <> struct simd_type<int, 4> {
  using type = int32x4_t;
  static constexpr size_t width = 4;
};
template <> struct neon_traits<int, 4> {
  using type = int32x4_t;
  static constexpr size_t width = 4;
  static type load(const int *ptr) { return vld1q_s32(ptr); }
  static type duplicate(int v) { return vdupq_n_s32(v); }
  static void store(int *ptr, type v) { vst1q_s32(ptr, v); }
  static type add(type a, type b) { return vaddq_s32(a, b); }
  static type sub(type a, type b) { return vsubq_s32(a, b); }
  static type mul(type a, type b) { return vmulq_s32(a, b); }
  static type div(type a, type b) {
    float32x4_t fa = vcvtq_f32_s32(a);
    float32x4_t fb = vcvtq_f32_s32(b);
    float32x4_t result = vdivq_f32(fa, fb);
    return vcvtq_s32_f32(result);
  }
};

template <> struct simd_type<int, 8> {
  using type = int32x4x2_t;
  static constexpr size_t width = 8;
};
template <> struct neon_traits<int, 8> {
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
  static type div(type a, type b) {
    float32x4_t fa0 = vcvtq_f32_s32(a.val[0]);
    float32x4_t fb0 = vcvtq_f32_s32(b.val[0]);
    float32x4_t result0 = vdivq_f32(fa0, fb0);
    float32x4_t fa1 = vcvtq_f32_s32(a.val[1]);
    float32x4_t fb1 = vcvtq_f32_s32(b.val[1]);
    float32x4_t result1 = vdivq_f32(fa1, fb1);
    return {vcvtq_s32_f32(result0), vcvtq_s32_f32(result1)};
  }
};

#endif
#endif
} // namespace hpc
