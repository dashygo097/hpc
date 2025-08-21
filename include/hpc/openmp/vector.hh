#pragma once

#ifdef _OPENMP
#include <memory>
#include <omp.h>
#endif

namespace hpc::openmp {
#ifdef _OPENMP
template <typename T> class Vector {
public:
  Vector() : _data(nullptr), _size(0), _capacity(0) {}

  explicit Vector(size_t size) : _size(size), _capacity(size) {
    _data = std::make_unique<T[]>(_capacity);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      _data[i] = T{};
    }
  }

  Vector(const T &value, size_t size) : _size(size), _capacity(size) {
    _data = std::make_unique<T[]>(_capacity);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      _data[i] = value;
    }
  }

  Vector(std::initializer_list<T> init)
      : _size(init.size()), _capacity(init.size()) {
    _data = std::make_unique<T[]>(_capacity);
    std::copy(init.begin(), init.end(), _data.get());
  }

  ~Vector() = default;

  [[nodiscard]] size_t size() const noexcept { return _size; }
  [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
  [[nodiscard]] bool empty() const noexcept { return _size == 0; }

  [[nodiscard]] T &operator[](size_t idx) { return _data[idx]; }
  [[nodiscard]] const T &operator[](size_t idx) const { return _data[idx]; }

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

  T *begin() noexcept { return _data.get(); }
  const T *begin() const noexcept { return _data.get(); }
  T *end() noexcept { return _data.get() + _size; }
  const T *end() const noexcept { return _data.get() + _size; }
  T *data() noexcept { return _data.get(); }
  const T *data() const noexcept { return _data.get(); }

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

  void resize(size_t new_size, const T &calue = T{}) {
    if (new_size > _capacity) {
      reallocate(new_size);
    }

    if (new_size > _size) {
#pragma omp parallel for if (new_size > PARALLEL_THRESHOLD)
      for (size_t i = _size; i < new_size; ++i) {
        _data[i] = calue;
      }
    }
    _size = new_size;
  }

  Vector operator+(const T &value) const {
    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result._data[i] = _data[i] + value;
    }
    return result;
  }

  Vector operator-(const T &value) const {
    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result._data[i] = _data[i] - value;
    }
    return result;
  }

  Vector operator*(const T &value) const {
    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result._data[i] = _data[i] * value;
    }
    return result;
  }

  Vector operator/(const T &value) const {
    if (value == 0) {
      throw std::runtime_error("Division by zero in vector division.");
    }
    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result._data[i] = _data[i] / value;
    }
    return result;
  }

  Vector operator+(const Vector &other) const {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for addition.");
    }

    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result._data[i] = _data[i] + other._data[i];
    }

    return result;
  }

  Vector operator-(const Vector &other) const {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for subtraction.");
    }

    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result._data[i] = _data[i] - other._data[i];
    }

    return result;
  }

  Vector operator*(const Vector &other) const {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for multiplication.");
    }

    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result._data[i] = _data[i] * other._data[i];
    }

    return result;
  }

  Vector operator/(const Vector &other) const {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for division.");
    }

    Vector result(_size);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      if (other._data[i] == 0) {
        throw std::runtime_error("Division by zero in vector division.");
      }
      result._data[i] = _data[i] / other._data[i];
    }
    return result;
  }

  Vector &operator+=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for addition.");
    }

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      _data[i] += other._data[i];
    }
    return *this;
  }

  Vector &operator-=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for subtraction.");
    }

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      _data[i] -= other._data[i];
    }
    return *this;
  }

  Vector &operator*=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for multiplication.");
    }

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      _data[i] *= other._data[i];
    }

    return *this;
  }

  Vector &operator/=(const Vector &other) {
    if (_size != other._size) {
      throw std::runtime_error(
          "Vectors must be of the same size for division.");
    }

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      if (other._data[i] == 0) {
        throw std::runtime_error("Division by zero in vector division.");
      }
      _data[i] /= other._data[i];
    }
    return *this;
  }

  T dot(const Vector &other) const {
    if (_size != other._size) {
      throw std::runtime_error("Vectors must be of the same size for dot "
                               "product.");
    }

    T result = T{};

#pragma omp parallel for reduction(+ : result) if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result += _data[i] * other._data[i];
    }

    return result;
  }

  T sum() const {
    if (_size == 0) {
      throw std::runtime_error("Cannot compute sum of an empty vector.");
    }

    T result = T{};

#pragma omp parallel for reduction(+ : result) if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      result += _data[i];
    }
    return result;
  }

private:
  std::unique_ptr<T[], std::default_delete<T[]>> _data;
  size_t _size;
  size_t _capacity;

  static constexpr size_t PARALLEL_THRESHOLD = 4096;

  void reallocate(size_t new_capacity) {
    auto new_data = std::make_unique<T[]>(new_capacity);

#pragma omp parallel for if (_size > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < _size; ++i) {
      new_data[i] = std::move(_data[i]);
    }

    _data = std::move(new_data);
    _capacity = new_capacity;
  }
};

#endif
} // namespace hpc::openmp
