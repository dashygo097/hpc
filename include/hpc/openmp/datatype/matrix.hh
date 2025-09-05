#pragma once

#ifdef ENABLE_OPENMP
#include "../../mmul_impl.hh"
#include "./vector.hh"
#endif

#ifdef ENABLE_SIMD
#include "../../simd_impl.hh"
#endif

namespace hpc::openmp {
#ifdef ENABLE_OPENMP
template <typename T, const size_t kBlockSize = BLOCK_SIZE,
          const size_t kSimdWidth = SIMD_WIDTH>
class Matrix {
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

  Matrix() : _data(nullptr), _size(0), _rows(0), _cols(0), _capacity(0) {}
  explicit Matrix(size_t rows, size_t cols)
      : _size(rows * cols), _rows(rows), _cols(cols), _capacity(rows * cols) {
    _data = std::make_unique<T[]>(_capacity);

    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_2D) {
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

  explicit Matrix(size_t rows, size_t cols, const T &value)
      : _size(rows * cols), _rows(rows), _cols(cols), _capacity(rows * cols) {
    _data = std::make_unique<T[]>(_capacity);

    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_2D) {
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

  Matrix(std::initializer_list<T> init)
      : _size(init.size()), _rows(1), _cols(init.size()),
        _capacity(init.size()) {
    _data = std::make_unique<T[]>(_capacity);

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ init_data = init.begin();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = init_data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = init_data[i];
      }
    }
  }

  ~Matrix() = default;

  Matrix(Matrix &&other) noexcept
      : _data(std::move(other._data)), _rows(other._rows), _cols(other._cols),
        _size(other._size), _capacity(other._capacity) {
    other._size = 0;
    other._rows = 0;
    other._cols = 0;
    other._capacity = 0;
  }

  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      if (_capacity < other._size) {
        _data = std::make_unique<T[]>(other._capacity);
        _capacity = other._capacity;
      }
      _size = other._size;

      T *__restrict__ this_data = _data.get();
      const T *__restrict__ other_data = other._data.get();
      const size_t block_size = kBlockSize;

      if (_size > PARALLEL_THRESHOLD_2D) {
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

  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      _data = std::move(other._data);
      _size = other._size;
      _rows = other._rows;
      _cols = other._cols;
      _capacity = other._capacity;
      other._size = 0;
      other._capacity = 0;
    }
    return *this;
  }

  [[nodiscard]] size_t rows() const noexcept { return _rows; }
  [[nodiscard]] size_t cols() const noexcept { return _cols; }
  [[nodiscard]] size_t size() const noexcept { return _size; }
  [[nodiscard]] size_t capacity() const noexcept { return _capacity; }

  [[nodiscard]] T &operator[](size_t idx) noexcept { return _data[idx]; }
  [[nodiscard]] const T &operator[](size_t idx) const noexcept {
    return _data[idx];
  }

  [[nodiscard]] T &at(size_t row, size_t col) noexcept {
    return _data[row * _cols + col];
  }
  [[nodiscard]] const T &at(size_t row, size_t col) const noexcept {
    return _data[row * _cols + col];
  }

  [[nodiscard]] Vector<T> row(size_t row) const noexcept {
    Vector<T> result(_cols);
    T *__restrict__ result_data = result.data();
    const T *__restrict__ this_data = _data.get() + row * _cols;
    const size_t block_size = kBlockSize;
    if (_cols > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _cols; block += block_size) {
          size_t end = std::min(block + block_size, _cols);
          for (size_t i = block; i < end; ++i) {
            result_data[i] = this_data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _cols; ++i) {
        result_data[i] = this_data[i];
      }
    }
    return result;
  }

  [[nodiscard]] Vector<T> col(size_t col) const noexcept {
    Vector<T> result(_rows);
    T *__restrict__ result_data = result.data();
    const T *__restrict__ this_data = _data.get() + col;
    const size_t block_size = kBlockSize;
    if (_rows > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _rows; block += block_size) {
          size_t end = std::min(block + block_size, _rows);
          for (size_t i = block; i < end; ++i) {
            result_data[i] = this_data[i * _cols];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _rows; ++i) {
        result_data[i] = this_data[i * _cols];
      }
    }
    return result;
  }

  [[nodiscard]] T *begin() noexcept { return _data.get(); }
  [[nodiscard]] const T *begin() const noexcept { return _data.get(); }
  [[nodiscard]] T *end() noexcept { return _data.get() + _size; }
  [[nodiscard]] const T *end() const noexcept { return _data.get() + _size; }
  [[nodiscard]] T *data() noexcept { return _data.get(); }
  [[nodiscard]] const T *data() const noexcept { return _data.get(); }

  void clear() noexcept {
    _rows = 0;
    _cols = 0;
    _size = 0;
    _capacity = 0;
    _data.reset();
  }

  void reserve(size_t new_capacity) {
    if (new_capacity > _capacity) {
      reallocate(new_capacity);
    }
  }

  void resize(size_t new_rows, size_t new_cols, const T &value = T{}) {
    size_t new_size = new_rows * new_cols;
    if (new_size > _capacity) {
      reallocate(new_size);
    }
    if (new_size > _size) {
      T *__restrict__ this_data = _data.get();
      const size_t old_size = _size;
      const size_t block_size = kBlockSize;
      if (new_size > PARALLEL_THRESHOLD_2D) {

#pragma omp parallel
        {
#pragma omp for schedule(static)
          for (size_t block = old_size; block < new_size; block += block_size) {
            size_t end = std::min(block + block_size, new_size);
            for (size_t i = block; i < end; ++i) {
              this_data[i] = value;
            }
          }
        }
      } else {
        for (size_t i = old_size; i < new_size; ++i) {
          this_data[i] = value;
        }
      }
    }
  }

  template <typename Func> Matrix &assign(const Matrix &mat1, Func &&func) {
    if (_rows != mat1._rows && _cols != mat1._cols) {
      throw std::runtime_error(
          "Matrixes must be of the same size for assignment.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ mat1_data = mat1._data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = func(mat1_data[i]);
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = func(mat1_data[i]);
      }
    }
    return *this;
  }

  template <typename Func>
  Matrix &assign(const Matrix &mat1, const Matrix &mat2, Func &&func) {
    if (_rows != mat1._rows || _cols != mat1._cols || _rows != mat2._rows ||
        _cols != mat2._cols) {
      throw std::runtime_error(
          "Matrixes must be of the same size for assignment.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ mat1_data = mat1._data.get();
    const T *__restrict__ mat2_data = mat2._data.get();
    const size_t block_size = kBlockSize;
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] = func(mat1_data[i], mat2_data[i]);
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] = func(mat1_data[i], mat2_data[i]);
      }
    }
    return *this;
  }

  Matrix &operator+=(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
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
    return *this;
  }

  Matrix &operator-=(const T &value) { return *this += -value; }

  Matrix &operator*=(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
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
    return *this;
  }

  Matrix &operator/=(const T &value) { return *this *= (1 / value); }

  void fill(const T &value) {
#ifndef ENABLE_SIMD
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_2D) {
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

#else
    fill_simd(value);
#endif
  }

  Matrix &operator+=(const Matrix &other) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] += other._data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] += other._data[i];
      }
    }

    return *this;
  }

  Matrix &operator-=(const Matrix &other) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] -= other._data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] -= other._data[i];
      }
    }
    return *this;
  }

  Matrix &operator*=(const Matrix &other) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] *= other._data[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] *= other._data[i];
      }
    }

    return *this;
  }

  Matrix &operator/=(const Matrix &other) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block = 0; block < _size; block += block_size) {
          size_t end = std::min(block + block_size, _size);
          for (size_t i = block; i < end; ++i) {
            this_data[i] /= other._data[i];
          }
        }
      }

    } else {
      for (size_t i = 0; i < _size; ++i) {
        this_data[i] /= other._data[i];
      }
    }

    return *this;
  }

private:
  std::unique_ptr<T[], std::default_delete<T[]>> _data;
  size_t _size;
  size_t _rows;
  size_t _cols;
  size_t _capacity;

  void reallocate(size_t new_capacity) {
    auto new_data = std::make_unique<T[]>(new_capacity);

    T *__restrict__ new_ptr = new_data.get();
    const T *__restrict__ old_ptr = _data.get();
    const size_t block_size = kBlockSize;

    if (_size > PARALLEL_THRESHOLD_2D) {
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
  void fill_simd(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            *((simd_t *)(this_data + i)) = simd_t(value);
          }
        }
        if (_size % kSimdWidth != 0) {
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = value;
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        *((simd_t *)(this_data + i)) = simd_t(value);
      }
      if (_size % kSimdWidth != 0) {
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = value;
        }
      }
    }

#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v = traits::duplicate(value);
            traits::store(this_data + i, v);
          }
        }
        if (_size % kSimdWidth != 0) {
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = value;
          }
        }
      }
    } else {
      for (size_t i = 0; i < simd_size; i += kSimdWidth) {
        simd_t v = traits::duplicate(value);
        traits::store(this_data + i, v);
      }
      if (_size % kSimdWidth != 0) {
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = value;
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `fill` function for "
                 "this platform."
              << std::endl;
#endif
  }

  void add_simd(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            *((simd_t *)(this_data + i)) += simd_t(value);
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v = *((simd_t *)(this_data + simd_size));
          simd_t result = v + simd_t(value);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        *((simd_t *)(this_data + i)) += simd_t(value);
      }
      if (_size % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(this_data + simd_size));
        simd_t result = v + simd_t(value);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }

#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v = traits::load(this_data + i);
            traits::store(this_data + i,
                          traits::add(v, traits::duplicate(value)));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v = traits::load(this_data + simd_size);
          simd_t result = traits::add(v, traits::duplicate(value));
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v = traits::load(this_data + i);
        traits::store(this_data + i, traits::add(v, traits::duplicate(value)));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v = traits::load(this_data + simd_size);
        simd_t result = traits::add(v, traits::duplicate(value));
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `+=` function for "
                 "this platform."
              << std::endl;
#endif
  }

  void mul_simd(const T &value) {
    T *__restrict__ this_data = _data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            *((simd_t *)(this_data + i)) *= simd_t(value);
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v = *((simd_t *)(this_data + simd_size));
          simd_t result = v * simd_t(value);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        *((simd_t *)(this_data + i)) *= simd_t(value);
      }
      if (_size % kSimdWidth != 0) {
        simd_t v = *((simd_t *)(this_data + simd_size));
        simd_t result = v * simd_t(value);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }

#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v = traits::load(this_data + i);
            traits::store(this_data + i,
                          traits::mul(v, traits::duplicate(value)));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v = traits::load(this_data + simd_size);
          simd_t result = traits::mul(v, traits::duplicate(value));
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v = traits::load(this_data + i);
        traits::store(this_data + i, traits::mul(v, traits::duplicate(value)));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v = traits::load(this_data + simd_size);
        simd_t result = traits::mul(v, traits::duplicate(value));
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }
#else
    std::cerr << "Not Implement SIMD for `*=` function for "
                 "this platform."
              << std::endl;
#endif
  }

  void add_vec_simd(const Matrix &other) {
    if (_rows != other._rows && _cols != other._cols) {
      throw std::runtime_error(
          "Matrixs must be of the same size for addition.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            *((simd_t *)(this_data + i)) += *((simd_t *)(other_data + i));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = *((simd_t *)(this_data + simd_size));
          simd_t v2 = *((simd_t *)(other_data + simd_size));
          simd_t result = v1 + v2;
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        *((simd_t *)(this_data + i)) += *((simd_t *)(other_data + i));
      }

      if (_size % kSimdWidth != 0) {
        simd_t v1 = *((simd_t *)(this_data + simd_size));
        simd_t v2 = *((simd_t *)(other_data + simd_size));
        simd_t result = v1 + v2;
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v1 = traits::load(this_data + i);
            simd_t v2 = traits::load(other_data + i);
            traits::store(this_data + i, traits::add(v1, v2));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = traits::load(this_data + simd_size);
          simd_t v2 = traits::load(other_data + simd_size);
          simd_t result = traits::add(v1, v2);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v1 = traits::load(this_data + i);
        simd_t v2 = traits::load(other_data + i);
        traits::store(this_data + i, traits::add(v1, v2));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = traits::load(this_data + simd_size);
        simd_t v2 = traits::load(other_data + simd_size);
        simd_t result = traits::add(v1, v2);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `+=` function for "
                 "this platform."
              << std::endl;
#endif
  }

  void sub_vec_simd(const Matrix &other) {
    if (_rows != other._rows && _cols != other._cols) {
      throw std::runtime_error(
          "Matrixs must be of the same size for addition.");
    }

    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            *((simd_t *)(this_data + i)) -= *((simd_t *)(other_data + i));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = *((simd_t *)(this_data + simd_size));
          simd_t v2 = *((simd_t *)(other_data + simd_size));
          simd_t result = v1 - v2;
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        *((simd_t *)(this_data + i)) -= *((simd_t *)(other_data + i));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = *((simd_t *)(this_data + simd_size));
        simd_t v2 = *((simd_t *)(other_data + simd_size));
        simd_t result = v1 - v2;
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }

#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v1 = traits::load(this_data + i);
            simd_t v2 = traits::load(other_data + i);
            traits::store(this_data + i, traits::sub(v1, v2));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = traits::load(this_data + simd_size);
          simd_t v2 = traits::load(other_data + simd_size);
          simd_t result = traits::sub(v1, v2);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v1 = traits::load(this_data + i);
        simd_t v2 = traits::load(other_data + i);
        traits::store(this_data + i, traits::sub(v1, v2));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = traits::load(this_data + simd_size);
        simd_t v2 = traits::load(other_data + simd_size);
        simd_t result = traits::sub(v1, v2);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `-=` function for "
                 "this platform."
              << std::endl;
#endif
  }

  void mul_vec_simd(const Matrix &other) {
    if (_rows != other._rows && _cols != other._cols) {
      throw std::runtime_error(
          "Matrixs must be of the same size for division.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;

#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            *((simd_t *)(this_data + i)) *= *((simd_t *)(other_data + i));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = *((simd_t *)(this_data + simd_size));
          simd_t v2 = *((simd_t *)(other_data + simd_size));
          simd_t result = v1 * v2;
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        *((simd_t *)(this_data + i)) *= *((simd_t *)(other_data + i));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = *((simd_t *)(this_data + simd_size));
        simd_t v2 = *((simd_t *)(other_data + simd_size));
        simd_t result = v1 * v2;
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v1 = traits::load(this_data + i);
            simd_t v2 = traits::load(other_data + i);
            traits::store(this_data + i, traits::mul(v1, v2));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = traits::load(this_data + simd_size);
          simd_t v2 = traits::load(other_data + simd_size);
          simd_t result = traits::mul(v1, v2);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v1 = traits::load(this_data + i);
        simd_t v2 = traits::load(other_data + i);
        traits::store(this_data + i, traits::mul(v1, v2));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = traits::load(this_data + simd_size);
        simd_t v2 = traits::load(other_data + simd_size);
        simd_t result = traits::mul(v1, v2);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `*=` function for "
                 "this platform."
              << std::endl;
#endif
  }

  void div_vec_simd(const Matrix &other) {
    if (_rows != other._rows && _cols != other._cols) {
      throw std::runtime_error(
          "Matrixs must be of the same size for division.");
    }
    T *__restrict__ this_data = _data.get();
    const T *__restrict__ other_data = other._data.get();
    const size_t block_size = kBlockSize;
    const size_t simd_size = _size - _size % kSimdWidth;
#if defined(__APPLE__)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            *((simd_t *)(this_data + i)) /= *((simd_t *)(other_data + i));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = *((simd_t *)(this_data + simd_size));
          simd_t v2 = *((simd_t *)(other_data + simd_size));
          simd_t result = v1 / v2;
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *((this_data + simd_size + i)) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        *((simd_t *)(this_data + i)) /= *((simd_t *)(other_data + i));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = *((simd_t *)(this_data + simd_size));
        simd_t v2 = *((simd_t *)(other_data + simd_size));
        simd_t result = v1 / v2;
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *((this_data + simd_size + i)) = result[i];
        }
      }
    }
#elif defined(__ARM_NEON)
    if (_size > PARALLEL_THRESHOLD_2D) {
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for (size_t block_idx = 0; block_idx < _size; block_idx += block_size) {
          size_t i_end = std::min(block_idx + block_size, simd_size);
          for (size_t i = block_idx; i < i_end; i += kSimdWidth) {
            simd_t v1 = traits::load(this_data + i);
            simd_t v2 = traits::load(other_data + i);
            traits::store(this_data + i, traits::div(v1, v2));
          }
        }
        if (_size % kSimdWidth != 0) {
          simd_t v1 = traits::load(this_data + simd_size);
          simd_t v2 = traits::load(other_data + simd_size);
          simd_t result = traits::div(v1, v2);
          for (size_t i = 0; i < _size % kSimdWidth; ++i) {
            *(this_data + simd_size + i) = result[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < _size; i += kSimdWidth) {
        simd_t v1 = traits::load(this_data + i);
        simd_t v2 = traits::load(other_data + i);
        traits::store(this_data + i, traits::div(v1, v2));
      }
      if (_size % kSimdWidth != 0) {
        simd_t v1 = traits::load(this_data + simd_size);
        simd_t v2 = traits::load(other_data + simd_size);
        simd_t result = traits::div(v1, v2);
        for (size_t i = 0; i < _size % kSimdWidth; ++i) {
          *(this_data + simd_size + i) = result[i];
        }
      }
    }

#else
    std::cerr << "Not Implement SIMD for `/=` function for "
                 "this platform."
              << std::endl;
#endif
  }

#endif
};

template <typename T>
Matrix<T> naive_mmul(const Matrix<T> &mat1, const Matrix<T> &mat2) {
  if (mat1.cols() != mat2.rows()) {
    throw std::runtime_error("Matrix multiplication dimension mismatch: (" +
                             std::to_string(mat1.rows()) + ", " +
                             std::to_string(mat1.cols()) + ") x (" +
                             std::to_string(mat2.rows()) + ", " +
                             std::to_string(mat2.cols()) + ")");
  }
  Matrix<T> result(mat1.rows(), mat2.cols());
  naive_mmul_impl(result.data(), mat1.data(), mat2.data(), mat1.rows(),
                  mat1.cols(), mat2.cols());
  return result;
}

template <typename T, const size_t kGemmTileSize = GEMM_TILE_SIZE>
Matrix<T> tiled_mmul(const Matrix<T> &mat1, const Matrix<T> &mat2) {
  if (mat1.cols() != mat2.rows()) {
    throw std::runtime_error("Matrix multiplication dimension mismatch: (" +
                             std::to_string(mat1.rows()) + ", " +
                             std::to_string(mat1.cols()) + ") x (" +
                             std::to_string(mat2.rows()) + ", " +
                             std::to_string(mat2.cols()) + ")");
  }
  Matrix<T> result(mat1.rows(), mat2.cols());
  tiled_mmul_impl(result.data(), mat1.data(), mat2.data(), mat1.rows(),
                  mat1.cols(), mat2.cols(), kGemmTileSize);
  return result;
}

#ifdef ENABLE_SIMD
template <typename T, const size_t kSimdWidth = SIMD_WIDTH>
Matrix<T> naive_mmul_simd_1xk(const Matrix<T> &mat1, const Matrix<T> &mat2) {
  if (mat1.cols() != mat2.rows()) {
    throw std::runtime_error("Matrix multiplication dimension mismatch: (" +
                             std::to_string(mat1.rows()) + ", " +
                             std::to_string(mat1.cols()) + ") x (" +
                             std::to_string(mat2.rows()) + ", " +
                             std::to_string(mat2.cols()) + ")");
  }
  Matrix<T> result(mat1.rows(), mat2.cols());
  naive_mmul_simd_1xk_impl<T, kSimdWidth>(result.data(), mat1.data(),
                                          mat2.data(), mat1.rows(), mat1.cols(),
                                          mat2.cols());
  return result;
}

#endif

#endif

} // namespace hpc::openmp
