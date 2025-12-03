#pragma once

#ifdef SIMD_APPLE
#include <simd/simd.h>
#endif

#ifdef SIMD_APPLE
#define SIMD_LOAD(traits, ptr)                                                 \
  (*reinterpret_cast<const typename traits::type *>(ptr))
#define SIMD_STORE(traits, ptr, value)                                         \
  (*(reinterpret_cast<typename traits::type *>(ptr)) = (value))
#define SIMD_DUP(traits, value) traits::duplicate(value)
#define SIMD_ADD(traits, type, a, b) ((a) + (b))
#define SIMD_SUB(traits, type, a, b) ((a) - (b))
#define SIMD_MUL(traits, a, b) ((a) * (b))
#define SIMD_FMA(traits, a, b, c) ((a) * (b) + (c))
#define SIMD_DIV(traits, a, b) ((a) / (b))
#endif

#ifdef SIMD_APPLE
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

template <> struct simd_traits<float, 2> {
  using type = simd_float2;
  static constexpr size_t width = 2;

  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return *(type *)(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return simd_make_float2(v, v);
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    *(type *)(ptr) = v;
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
  using type = simd_float4;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const float *ptr) {
    return *(type *)(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(float v) {
    return simd_make_float4(v, v, v, v);
  }
  __attribute__((always_inline)) static inline void store(float *ptr, type v) {
    *(type *)(ptr) = v;
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

template <> struct simd_traits<double, 2> {
  using type = simd_double2;
  static constexpr size_t width = 2;

  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return *(type *)(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return simd_make_double2(v, v);
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    *(type *)(ptr) = v;
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

template <> struct simd_traits<double, 4> {
  using type = simd_double4;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const double *ptr) {
    return *(type *)(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(double v) {
    return simd_make_double4(v, v, v, v);
  }
  __attribute__((always_inline)) static inline void store(double *ptr, type v) {
    *(type *)(ptr) = v;
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

template <> struct simd_traits<int, 2> {
  using type = simd_int2;
  static constexpr size_t width = 2;

  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return *(type *)(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return simd_make_int2(v, v);
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    *(type *)(ptr) = v;
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
  using type = simd_int4;
  static constexpr size_t width = 4;

  __attribute__((always_inline)) static inline type load(const int *ptr) {
    return *(type *)(ptr);
  }
  __attribute__((always_inline)) static inline type duplicate(int v) {
    return simd_make_int4(v, v, v, v);
  }
  __attribute__((always_inline)) static inline void store(int *ptr, type v) {
    *(type *)(ptr) = v;
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

} // namespace hpc::simd
#endif
