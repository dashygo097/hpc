#pragma once

#include "../constants.hh"

namespace hpc::openmp {
template <typename T, const size_t kBlockSize>
void vadd(T *__restrict__ A, const T &value, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] += value;
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] += value;
    }
  }
}

} // namespace hpc::openmp
