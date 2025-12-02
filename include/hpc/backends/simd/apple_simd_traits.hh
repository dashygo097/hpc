#pragma once

#ifdef SIMD_APPLE
#include <simd/simd.h>
#endif

#ifdef SIMD_APPLE
namespace hpc::simd {
template <typename T, size_t Width> struct simd_traits;

template <> struct simd_traits<float, 2> {
  using type = simd_float2;
  static constexpr size_t width = 2;

  static type load(const float *ptr) {
    return simd_make_float2(ptr[0], ptr[1]);
  }
  static type duplicate(float v) { return simd_make_float2(v, v); }
  static void store(float *ptr, type v) { *(type *)(ptr) = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
};

template <> struct simd_traits<float, 4> {
  using type = simd_float4;
  static constexpr size_t width = 4;

  static type load(const float *ptr) {
    return simd_make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
  }
  static type duplicate(float v) { return simd_make_float4(v, v, v, v); }
  static void store(float *ptr, type v) { *(type *)(ptr) = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
};

template <> struct simd_traits<double, 2> {
  using type = simd_double2;
  static constexpr size_t width = 2;

  static type load(const double *ptr) {
    return simd_make_double2(ptr[0], ptr[1]);
  }
  static type duplicate(double v) { return simd_make_double2(v, v); }
  static void store(double *ptr, type v) { *(type *)(ptr) = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
};

template <> struct simd_traits<double, 4> {
  using type = simd_double4;
  static constexpr size_t width = 4;
  static type load(const double *ptr) {
    return simd_make_double4(ptr[0], ptr[1], ptr[2], ptr[3]);
  }
  static type duplicate(double v) { return simd_make_double4(v, v, v, v); }
  static void store(double *ptr, type v) { *(type *)(ptr) = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
  static type div(type a, type b) { return a / b; }
};

template <> struct simd_traits<int, 2> {
  using type = simd_int2;
  static constexpr size_t width = 2;
  static type load(const int *ptr) { return simd_make_int2(ptr[0], ptr[1]); }
  static type duplicate(int v) { return simd_make_int2(v, v); }
  static void store(int *ptr, type v) { *(type *)(ptr) = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
};

template <> struct simd_traits<int, 4> {
  using type = simd_int4;
  static constexpr size_t width = 4;
  static type load(const int *ptr) {
    return simd_make_int4(ptr[0], ptr[1], ptr[2], ptr[3]);
  }
  static type duplicate(int v) { return simd_make_int4(v, v, v, v); }
  static void store(int *ptr, type v) { *(type *)(ptr) = v; }
  static type add(type a, type b) { return a + b; }
  static type sub(type a, type b) { return a - b; }
  static type mul(type a, type b) { return a * b; }
};

} // namespace hpc::simd
#endif
