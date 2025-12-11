#pragma once

#include "../l1/sequential.hh"
#include <cstddef>

namespace hpc::l2 {
namespace details {
template <typename T>
inline void gemv_seq(const size_t &M, const size_t &N, T *__restrict__ y,
                     const T *__restrict__ A, const T *__restrict__ x,
                     const T &alpha, const T &beta) {
  if (alpha == T{}) {
    l1::details::scal_seq(M, y, beta);
    return;
  }

  if (alpha == T{1}) {
    for (size_t i = 0; i < M; ++i) {
      T sum = T{};
      for (size_t j = 0; j < N; ++j) {
        sum += A[i * N + j] * x[j];
      }
      y[i] = sum + beta * y[i];
    }
    return;
  }

  for (size_t i = 0; i < M; ++i) {
    T sum = T{};
    for (size_t j = 0; j < N; ++j) {
      sum += A[i * N + j] * x[j];
    }
    y[i] = alpha * sum + beta * y[i];
  }
}

} // namespace details
} // namespace hpc::l2
