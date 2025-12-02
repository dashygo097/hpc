#pragma once

#include "../constants.hh"

namespace hpc::serial {
template <typename T> void vadd_baseline(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] += value;
  }
}
template <typename T> void vadd_baseline(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] += B[i];
  }
}

template <typename T> void vsub_baseline(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] -= value;
  }
}
template <typename T> void vsub_baseline(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] -= B[i];
  }
}

template <typename T> void vmul_baseline(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] *= value;
  }
}
template <typename T> void vmul_baseline(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] *= B[i];
  }
}

template <typename T> void vdiv_baseline(T *A, const T &value, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] /= value;
  }
}
template <typename T> void vdiv_baseline(T *A, const T *B, size_t N) {
  for (size_t i = 0; i < N; i++) {
    A[i] /= B[i];
  }
}
} // namespace hpc::serial

namespace hpc::openmp {

template <typename T, const size_t kBlockSize>
void vadd_impl(T *__restrict__ A, const T &value, size_t N) {
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
template <typename T, const size_t kBlockSize>
void vadd_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] += B[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] += B[i];
    }
  }
}

template <typename T, const size_t kBlockSize>
void vsub_impl(T *__restrict__ A, const T &value, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] -= value;
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] -= value;
    }
  }
}
template <typename T, const size_t kBlockSize>
void vsub_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] -= B[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] -= B[i];
    }
  }
}

template <typename T, const size_t kBlockSize>
void vmul_impl(T *__restrict__ A, const T &value, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] *= value;
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] *= value;
    }
  }
}
template <typename T, const size_t kBlockSize>
void vmul_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] *= B[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] *= B[i];
    }
  }
}

template <typename T, const size_t kBlockSize>
void vdiv_impl(T *__restrict__ A, const T &value, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] /= value;
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] /= value;
    }
  }
}
template <typename T, const size_t kBlockSize>
void vdiv_impl(T *__restrict__ A, const T *__restrict__ B, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] /= B[i];
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] /= B[i];
    }
  }
}

template <typename T, const size_t kBlockSize>
void vfill_impl(T *__restrict__ A, const T &value, size_t N) {
  if (N > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (size_t block = 0; block < N; block += kBlockSize) {
        size_t end = std::min(block + kBlockSize, N);
        for (size_t i = block; i < end; ++i) {
          A[i] = value;
        }
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      A[i] = value;
    }
  }
}

} // namespace hpc::openmp
