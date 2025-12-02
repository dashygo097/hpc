#pragma once

#include "../backends/backends.hh"

#ifdef ENABLE_SIMD
namespace hpc::l1 {
namespace details {

template <typename T, const size_t SimdWidth>
inline void vadd_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  using simd_t = typename traits::type;

  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    simd_t vd = traits::load(dst + i);
    simd_t vs = traits::load(src + i);
    traits::store(dst + i, traits::add(vd, vs));
  }

  for (size_t i = simd_end; i < n; ++i) {
    dst[i] += src[i];
  }
}

template <typename T, const size_t SimdWidth>
inline void vsub_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);
  const auto scalar_vec = traits::duplicate(scalar);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    auto v = traits::load(dst + i);
    traits::store(dst + i, traits::sub(v, scalar_vec));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] -= scalar;
}

template <typename T, const size_t SimdWidth>
inline void vsub_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    auto vd = traits::load(dst + i);
    auto vs = traits::load(src + i);
    traits::store(dst + i, traits::sub(vd, vs));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] -= src[i];
}

template <typename T, const size_t SimdWidth>
inline void vmul_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);
  const auto scalar_vec = traits::duplicate(scalar);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    auto v = traits::load(dst + i);
    traits::store(dst + i, traits::mul(v, scalar_vec));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= scalar;
}

template <typename T, const size_t SimdWidth>
inline void vmul_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    auto vd = traits::load(dst + i);
    auto vs = traits::load(src + i);
    traits::store(dst + i, traits::mul(vd, vs));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] *= src[i];
}

template <typename T, const size_t SimdWidth>
inline void vdiv_simd(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);
  const auto scalar_vec = traits::duplicate(scalar);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    auto v = traits::load(dst + i);
    traits::store(dst + i, traits::div(v, scalar_vec));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] /= scalar;
}

template <typename T, const size_t SimdWidth>
inline void vdiv_simd(T *__restrict__ dst, const T *__restrict__ src,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    auto vd = traits::load(dst + i);
    auto vs = traits::load(src + i);
    traits::store(dst + i, traits::div(vd, vs));
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] /= src[i];
}

template <typename T, const size_t SimdWidth>
inline void vfill_simd(T *__restrict__ dst, const T &value, size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);
  const auto value_vec = traits::duplicate(value);

  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    traits::store(dst + i, value_vec);
  }
  for (size_t i = simd_end; i < n; ++i)
    dst[i] = value;
}

// l1
template <typename T, const size_t SimdWidth>
inline void axpy_simd(T *__restrict__ y, const T *__restrict__ x, const T &a,
                      size_t n) {
  using traits = simd::simd_traits<T, SimdWidth>;
  const size_t simd_end = n - (n % SimdWidth);
  const auto a_vec = traits::duplicate(a);
  for (size_t i = 0; i < simd_end; i += SimdWidth) {
    auto vy = traits::load(y + i);
    auto vx = traits::load(x + i);
    traits::store(y + i, traits::add(vy, traits::mul(a_vec, vx)));
  }
  for (size_t i = simd_end; i < n; ++i)
    y[i] += a * x[i];
}

} // namespace details
} // namespace hpc::l1
#endif
