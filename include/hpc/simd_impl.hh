#pragma once

namespace hpc {
template <typename T, size_t Width = SIMD_WIDTH> struct simd_type;

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
template <typename T> using simd_t = typename simd_type<T>::type;

#endif
#endif
} // namespace hpc
