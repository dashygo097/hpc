#pragma once

#ifdef ENABLE_OPENMP
#include <memory>
#endif

#ifdef ENABLE_SIMD
#include "../../simd_impl.hh"
#endif

namespace hpc::openmp {
#ifdef ENABLE_OPENMP
template <typename T> class Vector {
public:
  using value_type = T;
#ifdef ENABLE_SIMD
#if defined(__APPLE__)
  using simd_type = simd_t<T>;
#endif
#endif

  Vector() : _data(nullptr), _size(0), _capacity(0) {}

  explicit Vector(size_t size) : _size(size), _capacity(size) {
    _data = std::make_unique<T[]>(_capacity);

    T *__restrict__ this_data = _data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
      const size_t block_dim = BLOCK_DIM;

      if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
        {
#pragma omp for schedule(static)
          for (size_t block = 0; block < _size; block += block_dim) {
            size_t end = std::min(block + block_dim, _size);
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
      const size_t block_dim = BLOCK_DIM;

      if (new_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
        {
#pragma omp for schedule(static)
          for (size_t block = _size; block < new_size; block += block_dim) {
            size_t end = std::min(block + block_dim, new_size);
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

  void fill(const T &value) {
#ifndef ENABLE_SIMD
    T *__restrict__ this_data = _data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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

#else
    fill_simd(value);
#endif
  }

  template <typename Func> Vector &assign(const Vector &vec1, Func &&func) {
#ifndef ENABLE_SIMD
    if (_size != vec1._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for assignment.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ vec1_data = vec1._data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
#ifndef ENABLE_SIMD
    T *__restrict__ this_data = _data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] += value;
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] += value;
      }
    }
#else
    add_simd(value);
#endif

    return *this;
  }

  Vector &operator-=(const T &value) { return *this += -value; }

  Vector &operator*=(const T &value) {
#ifndef ENABLE_SIMD
    T *__restrict__ this_data = _data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] *= value;
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] *= value;
      }
    }
#else
    return mul_simd(value);
#endif

    return *this;
  }

  Vector &operator/=(const T &value) { return *this *= (1 / value); }

  Vector &operator+=(const Vector &other) {
#ifndef ENABLE_SIMD
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for addition.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] += other_data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] += other_data[i];
      }
    }
#else
    add_vec_simd(other);
#endif

    return *this;
  }

  Vector &operator-=(const Vector &other) {
#ifndef ENABLE_SIMD
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for subtraction.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] -= other_data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] -= other_data[i];
      }
    }
#else
    sub_vec_simd(other);
#endif
    return *this;
  }

  Vector &operator*=(const Vector &other) {
#ifndef ENABLE_SIMD
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for multiplication.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] *= other_data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] *= other_data[i];
      }
    }
#else
    mul_vec_simd(other);
#endif

    return *this;
  }

  Vector &operator/=(const Vector &other) {
#ifndef ENABLE_SIMD
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for division.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
          for (size_t i = block; i < end; ++i) {
            if (other_data[i] == 0) {
              throw std::runtime_error("Division by zero in vector division.");
            }
            this_data[i] /= other_data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        if (other_data[i] == 0) {
          throw std::runtime_error("Division by zero in vector division.");
        }
        this_data[i] /= other_data[i];
      }
    }
#else
    div_vec_simd(other);
#endif

    return *this;
  }

  T dot(const Vector &other) const {
    if (_size != other._size) {
      throw std::runtime_error("Vectors must be of the same size for dot "
                               "product.");
    }

    T result = T{};
    const T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel reduction(+ : result)
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel reduction(+ : result)
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
    const size_t block_dim = BLOCK_DIM;

    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_dim) {
          size_t end = std::min(block + block_dim, _size);
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
  void fill_simd(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            *((simd_type *)(this_data + i)) = simd_type(value);
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v = *((simd_type *)(this_data + simd_size));
          simd_type result = simd_type(value);
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        *((simd_type *)(this_data + i)) = simd_type(value);
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type v = *((simd_type *)(this_data + simd_size));
        simd_type result = simd_type(value);
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `fill` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

  void add_simd(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            *((simd_type *)(this_data + i)) += simd_type(value);
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v = *((simd_type *)(this_data + simd_size));
          simd_type result = v + simd_type(value);
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        *((simd_type *)(this_data + i)) += simd_type(value);
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type v = *((simd_type *)(this_data + simd_size));
        simd_type result = v + simd_type(value);
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `+=` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

  void mul_simd(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            *((simd_type *)(this_data + i)) *= simd_type(value);
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v = *((simd_type *)(this_data + simd_size));
          simd_type result = v * simd_type(value);
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        *((simd_type *)(this_data + i)) *= simd_type(value);
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type v = *((simd_type *)(this_data + simd_size));
        simd_type result = v * simd_type(value);
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `*=` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

  void add_vec_simd(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for addition.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            *((simd_type *)(this_data + i)) += *((simd_type *)(other_data + i));
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v1 = *((simd_type *)(this_data + simd_size));
          simd_type v2 = *((simd_type *)(other_data + simd_size));
          simd_type result = v1 + v2;
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        *((simd_type *)(this_data + i)) += *((simd_type *)(other_data + i));
      }

      if (_size % SIMD_WIDTH != 0) {
        simd_type v1 = *((simd_type *)(this_data + simd_size));
        simd_type v2 = *((simd_type *)(other_data + simd_size));
        simd_type result = v1 + v2;
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `+=` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

  void sub_vec_simd(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for subtraction.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            *((simd_type *)(this_data + i)) -= *((simd_type *)(other_data + i));
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v1 = *((simd_type *)(this_data + simd_size));
          simd_type v2 = *((simd_type *)(other_data + simd_size));
          simd_type result = v1 - v2;
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        *((simd_type *)(this_data + i)) -= *((simd_type *)(other_data + i));
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type v1 = *((simd_type *)(this_data + simd_size));
        simd_type v2 = *((simd_type *)(other_data + simd_size));
        simd_type result = v1 - v2;
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `-=` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

  void div_vec_simd(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for division.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            *((simd_type *)(this_data + i)) /= *((simd_type *)(other_data + i));
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v1 = *((simd_type *)(this_data + simd_size));
          simd_type v2 = *((simd_type *)(other_data + simd_size));
          simd_type result = v1 / v2;
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        *((simd_type *)(this_data + i)) /= *((simd_type *)(other_data + i));
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type v1 = *((simd_type *)(this_data + simd_size));
        simd_type v2 = *((simd_type *)(other_data + simd_size));
        simd_type result = v1 / v2;
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `-=` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

  void mul_vec_simd(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for multiplication.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            *((simd_type *)(this_data + i)) *= *((simd_type *)(other_data + i));
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v1 = *((simd_type *)(this_data + simd_size));
          simd_type v2 = *((simd_type *)(other_data + simd_size));
          simd_type result = v1 * v2;
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        *((simd_type *)(this_data + i)) *= *((simd_type *)(other_data + i));
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type v1 = *((simd_type *)(this_data + simd_size));
        simd_type v2 = *((simd_type *)(other_data + simd_size));
        simd_type result = v1 * v2;
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `*=` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

  template <typename Func>
  void assign_unary_simd(const Vector &vec1, Func &&func) {
    if (_size != vec1._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for assignment.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ vec1_data = vec1._data.get();
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            simd_type vi = *((simd_type *)(vec1_data + i));
            *((simd_type *)(this_data + i)) = func(vi);
          }
          if (_size % SIMD_WIDTH != 0) {
            simd_type vi = *((simd_type *)(vec1_data + simd_size));
            simd_type result = func(vi);
            for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
              *((this_data + simd_size + i)) = result[i];
            }
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        simd_type vi = *((simd_type *)(vec1_data + i));
        *((simd_type *)(this_data + i)) = func(vi);
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type vi = *((simd_type *)(vec1_data + simd_size));
        simd_type result = func(vi);
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `assign` function for "
                 "non-Apple platforms."
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
    const size_t block_dim = BLOCK_DIM;
    const size_t simd_size = _size - _size % SIMD_WIDTH;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_1D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_dim) {
          size_t i_end = std::min(block_idx + block_dim, simd_size);
          for (size_t i = block_idx; i < i_end; i += SIMD_WIDTH) {
            simd_type v1 = *((simd_type *)(vec1_data + i));
            simd_type v2 = *((simd_type *)(vec2_data + i));
            *((simd_type *)(this_data + i)) = func(v1, v2);
          }
        }
        if (_size % SIMD_WIDTH != 0) {
          simd_type v1 = *((simd_type *)(vec1_data + simd_size));
          simd_type v2 = *((simd_type *)(vec2_data + simd_size));
          simd_type result = func(v1, v2);
          for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += SIMD_WIDTH) {
        simd_type v1 = *((simd_type *)(vec1_data + i));
        simd_type v2 = *((simd_type *)(vec2_data + i));
        *((simd_type *)(this_data + i)) = func(v1, v2);
      }
      if (_size % SIMD_WIDTH != 0) {
        simd_type v1 = *((simd_type *)(vec1_data + simd_size));
        simd_type v2 = *((simd_type *)(vec2_data + simd_size));
        simd_type result = func(v1, v2);
        for (size_t i = 0; i < _size % SIMD_WIDTH; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `assign` function for "
                 "non-Apple platforms."
              << std::endl;
#endif
  }

#endif
};

#endif
} // namespace hpc::openmp
