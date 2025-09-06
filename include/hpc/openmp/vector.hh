#pragma once

#include "../impls/vec_omp_impl.hh"

#ifdef ENABLE_SIMD
#include "../impls/vec_omp_simd_impl.hh"
#endif

namespace hpc::openmp {
template <typename T, const size_t kBlockSize = BLOCK_SIZE,
          const size_t kSimdWidth = SIMD_WIDTH>
class Vector {
public:
  using value_type = T;
#ifdef ENABLE_SIMD
#if defined(__APPLE__)
  using simd_t = typename simd_type<T, kSimdWidth>::type;
#elif defined(__ARM_NEON)
  using traits = neon_traits<T, kSimdWidth>;
  using simd_t = typename traits::type;
#endif
#endif

  Vector() : _data(nullptr), _size(0), _capacity(0) {}

  explicit Vector(size_t size) : _size(size), _capacity(size) {
    _data = std::make_unique<T[]>(_capacity);

    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = T{};
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = T{};
      }
    }
  }

  explicit Vector(size_t size, const T &value) : _size(size), _capacity(size) {
    _data = std::make_unique<T[]>(_capacity);

    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = value;
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = value;
      }
    }
  }

  Vector(std::initializer_list<T> init)
      : _size(init.size()), _capacity(init.size()) {
    _data = std::make_unique<T[]>(_capacity);
    std::copy(init.begin(), init.end(), _data.get());
  }

  ~Vector() = default;

  Vector(const Vector &other) : _size(other._size), _capacity(other._capacity) {
    _data = std::make_unique<T[]>(_capacity);

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = other_data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = other_data[i];
      }
    }
  }

  Vector(Vector &&other) noexcept
      : _data(std::move(other._data)), _size(other._size),
        _capacity(other._capacity) {
    other._size = 0;
    other._capacity = 0;
  }

  Vector &operator=(const Vector &other) {
    if (this != &other) {
      if (_capacity < other._size) {
        _data = std::make_unique<T[]>(other._capacity);
        _capacity = other._capacity;
      }
      _size = other._size;

      T *__restrict__ this_data = _data.get();
      const T *__restrict__ other_data = other._data.get();
      const size_t block_size = kBlockSize;

      if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
        {
#pragma omp for schedule(static)
          for (size_t block = 0; block < _size; block += block_size) {
            size_t end = std::min(block + block_size, _size);
            for (size_t i = block; i < end; ++i) {
              this_data[i] = other_data[i];
            }
          }
        }
      } else {
        for (size_t i = 0; i < _size; ++i) {
          this_data[i] = other_data[i];
        }
      }
    }
    return *this;
  }

  Vector &operator=(Vector &&other) noexcept {
    if (this != &other) {
      _data = std::move(other._data);
      _size = other._size;
      _capacity = other._capacity;
      other._size = 0;
      other._capacity = 0;
    }
    return *this;
  }

  [[nodiscard]] size_t size() const noexcept { return _size; }
  [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
  [[nodiscard]] bool empty() const noexcept { return _size == 0; }

  [[nodiscard]] T &operator[](size_t idx) noexcept { return _data[idx]; }
  [[nodiscard]] const T &operator[](size_t idx) const noexcept {
    return _data[idx];
  }

  [[nodiscard]] T &at(size_t idx) {
    if (idx >= _size) {
      throw std::out_of_range("Index out of range in vector access.");
    }
    return _data[idx];
  }
  [[nodiscard]] const T &at(size_t idx) const {
    if (idx >= _size) {
      throw std::out_of_range("Index out of range in vector access.");
    }
    return _data[idx];
  }

  [[nodiscard]] T *begin() noexcept { return _data.get(); }
  [[nodiscard]] const T *begin() const noexcept { return _data.get(); }
  [[nodiscard]] T *end() noexcept { return _data.get() + _size; }
  [[nodiscard]] const T *end() const noexcept { return _data.get() + _size; }
  [[nodiscard]] T *data() noexcept { return _data.get(); }
  [[nodiscard]] const T *data() const noexcept { return _data.get(); }

  void clear() noexcept {
    _size = 0;
    _capacity = 0;
    _data.reset();
  }

  void reserve(size_t new_capacity) {
    if (new_capacity > _capacity) {
      reallocate(new_capacity);
    }
  }

  void resize(size_t new_size, const T &value = T{}) {
    if (new_size > _capacity) {
      reallocate(new_size);
    }

    if (new_size > _size) {
      T *__restrict__ this_data = _data.get();
      const size_t block_size = kBlockSize;

      if (new_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
        {
#pragma omp for schedule(static)
          for (size_t block = _size; block < new_size; block += block_size) {
            size_t end = std::min(block + block_size, new_size);
            for (size_t i = block; i < end; ++i) {
              this_data[i] = value;
            }
          }
        }
      } else {
        for (size_t i = _size; i < new_size; ++i) {
          this_data[i] = value;
        }
      }
    }
    _size = new_size;
  }

  template <typename Func> Vector &assign(const Vector &vec1, Func &&func) {
#ifndef ENABLE_SIMD
    if (_size != vec1._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for assignment.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ vec1_data = vec1._data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = func(vec1_data[i]);
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = func(vec1_data[i]);
      }
    }
#else
    assign_unary_simd(vec1, func);
#endif

    return *this;
  }

  template <typename Func>
  Vector &assign(const Vector &vec1, const Vector &vec2, Func &&func) {
#ifndef ENABLE_SIMD
    if (_size != vec1._size || _size != vec2._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for assignment.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ vec1_data = vec1.data();
    const T *__restrict__ vec2_data = vec2.data();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = func(vec1_data[i], vec2_data[i]);
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = func(vec1_data[i], vec2_data[i]);
      }
    }
#else
    assign_binary_simd(vec1, vec2, func);
#endif

    return *this;
  }

  Vector &operator+=(const T &value) {
    T *__restrict__ this_data = _data.get();
#ifndef ENABLE_SIMD
    vadd_impl<T, kBlockSize>(this_data, value, _size);
#else
    vadd_simd_impl<T, kBlockSize, kSimdWidth>(this_data, value, _size);
#endif
    return *this;
  }

  Vector &operator-=(const T &value) { return *this += -value; }

  Vector &operator*=(const T &value) {
    T *__restrict__ this_data = _data.get();
#ifndef ENABLE_SIMD
    vmul_impl<T, kBlockSize>(this_data, value, _size);
#else
    vmul_simd_impl<T, kBlockSize, kSimdWidth>(this_data, value, _size);
#endif
    return *this;
  }

  Vector &operator/=(const T &value) { return *this *= (1 / value); }

  Vector &operator+=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for addition.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();

#ifndef ENABLE_SIMD
    vadd_impl<T, kBlockSize>(this_data, other_data, _size);
#else
    vadd_simd_impl<T, kBlockSize, kSimdWidth>(this_data, other_data, _size);
#endif
    return *this;
  }

  Vector &operator-=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for subtraction.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
#ifndef ENABLE_SIMD
    vsub_impl<T, kBlockSize>(this_data, other_data, _size);
#else
    vsub_simd_impl<T, kBlockSize, kSimdWidth>(this_data, other_data, _size);
#endif
    return *this;
  }

  Vector &operator*=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for multiplication.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
#ifndef ENABLE_SIMD
    vmul_impl<T, kBlockSize>(this_data, other_data, _size);
#else
    vmul_simd_impl<T, kBlockSize, kSimdWidth>(this_data, other_data, _size);
#endif
    return *this;
  }

  Vector &operator/=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for division.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
#ifndef ENABLE_SIMD
    vdiv_impl<T, kBlockSize>(this_data, other_data, _size);
#else
    vdiv_simd_impl<T, kBlockSize, kSimdWidth>(this_data, other_data, _size);
#endif

    return *this;
  }

  void fill(const T &value) {
    T *__restrict__ this_data = _data.get();
#ifndef ENABLE_SIMD
    vfill_impl<T, kBlockSize>(this_data, value, _size);
#else
    vfill_simd_impl<T, kBlockSize, kSimdWidth>(this_data, value, _size);
#endif
  }

  T dot(const Vector &other) const {
    if (_size != other._size) {
      throw std::runtime_error("Vectors must be of the same size for dot "
                               "product.");
    }

    T result = T{};
    const T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel reduction(+ : result)
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          T local_result = T{};
          for (size_t i = block; i < end; ++i) {
            local_result += this_data[i] * other_data[i];
          }
          result += local_result;
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        result += this_data[i] * other_data[i];
      }
    }

    return result;
  }

  T sum() const {
    if (_size == 0) {
      throw std::runtime_error("Cannot compute sum of an empty vector.");
    }

    T result = T{};
    const T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel reduction(+ : result)
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          T local_result = T{};
          for (size_t i = block; i < end; ++i) {
            local_result += this_data[i];
          }
          result += local_result;
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        result += this_data[i];
      }
    }
    return result;
  }

private:
  std::unique_ptr<T[], std::default_delete<T[]>> _data;
  size_t _size;
  size_t _capacity;

  void reallocate(size_t new_capacity) {
    auto new_data = std::make_unique<T[]>(new_capacity);

    T *__restrict__ new_ptr = new_data.get();
    const T *__restrict__ old_ptr = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            new_ptr[i] = std::move(old_ptr[i]);
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        new_ptr[i] = std::move(old_ptr[i]);
      }
    }

    _data = std::move(new_data);
    _capacity = new_capacity;
  }

#ifdef ENABLE_SIMD

  template <typename Func>
  void assign_unary_simd(const Vector &vec1, Func &&func) {
    if (_size != vec1._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for assignment.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ vec1_data = vec1._data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t vi = *((simd_t *)(vec1_data + i));
            *((simd_t *)(this_data + i)) = func(vi);
          }
          if (_size % kSimdWidth != 0) {
            simd_t vi = *((simd_t *)(vec1_data + simd_size));
            simd_t result = func(vi);
            for (size_t i = 0; i < _size % kSimdWidth; ++i) {
              *((this_data + simd_size + i)) = result[i];
            }
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t vi = *((simd_t *)(vec1_data + i));
        *((simd_t *)(this_data + i)) = func(vi);
      }
      if (_size % kSimdWidth != 0) {
        simd_t vi = *((simd_t *)(vec1_data + simd_size));
        simd_t result = func(vi);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }

#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t vi = traits::load(vec1_data + i);
            traits::store(this_data + i, func(vi));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t vi = traits::load(vec1_data + simd_size);
          simd_t result = func(vi);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t vi = traits::load(vec1_data + i);
        traits::store(this_data + i, func(vi));
      }
      if (_size % kSimdWidth != 0) {
        simd_t vi = traits::load(vec1_data + simd_size);
        simd_t result = func(vi);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `assign` function for "
                 "this platform."
              << std::endl;
#endif
  }

  template <typename Func>
  void assign_binary_simd(const Vector &vec1, const Vector &vec2, Func &&func) {
    if (_size != vec1._size || _size != vec2._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for assignment.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ vec1_data = vec1.data();
    const T *__restrict__ vec2_data = vec2.data();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v1 = *((simd_t *)(vec1_data + i));
            simd_t v2 = *((simd_t *)(vec2_data + i));
            *((simd_t *)(this_data + i)) = func(v1, v2);
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = *((simd_t *)(vec1_data + simd_size));
          simd_t v2 = *((simd_t *)(vec2_data + simd_size));
          simd_t result = func(v1, v2);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v1 = *((simd_t *)(vec1_data + i));
        simd_t v2 = *((simd_t *)(vec2_data + i));
        *((simd_t *)(this_data + i)) = func(v1, v2);
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = *((simd_t *)(vec1_data + simd_size));
        simd_t v2 = *((simd_t *)(vec2_data + simd_size));
        simd_t result = func(v1, v2);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v1 = traits::load(vec1_data + i);
            simd_t v2 = traits::load(vec2_data + i);
            traits::store(this_data + i, func(v1, v2));
          }
        }

        if (_size % kSimdWidth != 0) {
          simd_t v1 = traits::load(vec1_data + simd_size);
          simd_t v2 = traits::load(vec2_data + simd_size);
          simd_t result = func(v1, v2);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v1 = traits::load(vec1_data + i);
        simd_t v2 = traits::load(vec2_data + i);
        traits::store(this_data + i, func(v1, v2));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = traits::load(vec1_data + simd_size);
        simd_t v2 = traits::load(vec2_data + simd_size);
        simd_t result = func(v1, v2);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `assign` function for "
                 "this platform."
              << std::endl;
#endif
  }

#endif
};

} // namespace hpc::openmp
