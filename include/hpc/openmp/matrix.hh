#pragma once

#include "../impls/mmul_impl.hh"
#include "../impls/vec_omp_impl.hh"
#include "./vector.hh"

#ifdef ENABLE_SIMD
#include "../impls/vec_omp_simd_impl.hh"
#endif

namespace hpc::openmp {
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
#ifndef ENABLE_SIMD
    vadd_impl<T, kBlockSize>(this_data, value, _size);
#else
    vadd_simd_impl<T, kBlockSize, kSimdWidth>(this_data, value, _size);
#endif
    return *this;
  }

  Matrix &operator-=(const T &value) { return *this += -value; }

  Matrix &operator*=(const T &value) {
    T *__restrict__ this_data = _data.get();
#ifndef ENABLE_SIMD
    vmul_impl<T, kBlockSize>(this_data, value, _size);
#else
    vmul_simd_impl<T, kBlockSize, kSimdWidth>(this_data, value, _size);
#endif
    return *this;
  }

  Matrix &operator/=(const T &value) { return *this *= (1 / value); }

  Matrix &operator+=(const Matrix &other) {
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

  Matrix &operator-=(const Matrix &other) {
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

  Matrix &operator*=(const Matrix &other) {
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

  Matrix &operator/=(const Matrix &other) {
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
  naive_mmul_impl<T>(result.data(), mat1.data(), mat2.data(), mat1.rows(),
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
  tiled_mmul_impl<T, kGemmTileSize>(result.data(), mat1.data(), mat2.data(),
                                    mat1.rows(), mat1.cols(), mat2.cols());
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

} // namespace hpc::openmp
