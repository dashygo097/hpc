#pragma once

#ifdef ENABLE_ACCELERATE
#include "../../backends/backends.hh"
#endif

#ifdef ENABLE_ACCELERATE
namespace hpc::l1 {
namespace details {

template <typename T>
inline void vadd_acceler(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vsadd(dst, scalar, dst, n);
}

template <typename T>
inline void vadd_acceler(T *__restrict__ dst, const T *__restrict__ src,
                         size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vadd(dst, src, dst, n);
}

template <typename T>
inline void vsub_acceler(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = acceler::dsp_traits<T>;
  T neg_scalar = -scalar;
  traits::vsadd(dst, neg_scalar, dst, n);
}

template <typename T>
inline void vsub_acceler(T *__restrict__ dst, const T *__restrict__ src,
                         size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vsub(dst, src, dst, n);
}

template <typename T>
inline void vmul_acceler(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vsmul(dst, scalar, dst, n);
}

template <typename T>
inline void vmul_acceler(T *__restrict__ dst, const T *__restrict__ src,
                         size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vmul(dst, src, dst, n);
}

template <typename T>
inline void vdiv_acceler(T *__restrict__ dst, const T &scalar, size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vsdiv(dst, scalar, dst, n);
}

template <typename T>
inline void vdiv_acceler(T *__restrict__ dst, const T *__restrict__ src,
                         size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vdiv(dst, src, dst, n);
}

// reduce
template <typename T>
inline void vsum_acceler(const T *__restrict__ src, T &result, size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vsum(src, &result, n);
}

// fill
template <typename T>
inline void vfill_acceler(T *__restrict__ dst, const T &value, size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vfill(value, dst, n);
}

// copy
template <typename T>
inline void vcopy_acceler(T *__restrict__ dst, const T *__restrict__ src,
                          size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::vcopy(src, dst, n);
}

template <typename T>
inline void axpy_acceler(T *__restrict__ y, const T *__restrict__ x,
                         const T &alpha, size_t n) {
  using traits = acceler::dsp_traits<T>;
  traits::axpy(n, alpha, x, y);
}

} // namespace details
} // namespace hpc::l1

#endif
