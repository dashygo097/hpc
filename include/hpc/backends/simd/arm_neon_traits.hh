#pragma once

#ifdef SIMD_NEON
#include <arm_neon.h>
#endif

#ifdef SIMD_NEON
namespace hpc::simd {
template <typename T, size_t Width> struct simd_traits;

template <> struct simd_traits<float, 4> {
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

} // namespace hpc::simd
#endif
