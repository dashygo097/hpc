#pragma once

#ifdef HPC_ENABLE_CUDA
#include "./err.cuh"
#include <cuda_runtime.h>
#include <memory>
#endif

#ifdef HPC_ENABLE_CUDA
namespace hpc::cu {

// Memory Allocation
template <typename T> T *mallocDevice(size_t count) {
  T *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  return ptr;
}

template <typename T> T *mallocHost(size_t count) {
  T *ptr = nullptr;
  CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
  return ptr;
}

template <typename T> T *mallocManaged(size_t count) {
  T *ptr = nullptr;
  CUDA_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
  return ptr;
}

template <typename T> void free(T *ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

template <typename T> void freeHost(T *ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
}

// Memory Copy
template <typename T> void copyH2D(T *dst, const T *src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> void copyD2H(T *dst, const T *src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T> void copyD2D(T *dst, const T *src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T> void memset(T *ptr, int value, size_t count) {
  CUDA_CHECK(cudaMemset(ptr, value, count * sizeof(T)));
}

// RAII Wrappers
template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) : _size(count), _ptr(nullptr) {
    if (count > 0) {
      _ptr = mallocDevice<T>(count);
    }
  }

  ~DeviceBuffer() {
    if (_ptr) {
      cudaFree(_ptr);
    }
  }

  // Delete copy
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  // Allow move
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : _size(other._size), _ptr(other._ptr) {
    other._ptr = nullptr;
    other._size = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (_ptr)
        cudaFree(_ptr);
      _ptr = other._ptr;
      _size = other._size;
      other._ptr = nullptr;
      other._size = 0;
    }
    return *this;
  }

  T *data() { return _ptr; }
  const T *data() const { return _ptr; }
  size_t size() const { return _size; }

  void copyFromHost(const T *src, size_t count = 0) {
    if (count == 0)
      count = _size;
    copyH2D(_ptr, src, count);
  }

  void copyToHost(T *dst, size_t count = 0) const {
    if (count == 0)
      count = _size;
    copyD2H(dst, _ptr, count);
  }

private:
  size_t _size;
  T *_ptr;
};

template <typename T> class PinnedBuffer {
public:
  explicit PinnedBuffer(size_t count) : _size(count), _ptr(nullptr) {
    if (count > 0) {
      _ptr = mallocHost<T>(count);
    }
  }

  ~PinnedBuffer() {
    if (_ptr) {
      cudaFreeHost(_ptr);
    }
  }

  // Delete copy
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  // Allow move
  PinnedBuffer(PinnedBuffer &&other) noexcept
      : _size(other._size), _ptr(other._ptr) {
    other._size = 0;
    other._ptr = nullptr;
  }

  PinnedBuffer &operator=(PinnedBuffer &&other) noexcept {
    if (this != &other) {
      if (_ptr)
        cudaFreeHost(_ptr);
      _ptr = other._ptr;
      _size = other._size;
      other._ptr = nullptr;
      other._size = 0;
    }
    return *this;
  }

  T *data() { return _ptr; }
  const T *data() const { return _ptr; }
  size_t size() const { return _size; }

  T &operator[](size_t idx) { return _ptr[idx]; }
  const T &operator[](size_t idx) const { return _ptr[idx]; }

private:
  size_t _size;
  T *_ptr;
};

} // namespace hpc::cu
#endif
