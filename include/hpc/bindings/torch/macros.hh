#pragma once

#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(func##, &func, func##);
#define TORCH_CHECK_TENSOR_DTYPE(T, dtype)                                     \
  if (((T).options().dtype() != (dtype))) {                                    \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    std::cout << "Expected dtype: " << (dtype) << std::endl;                   \
    throw std::runtime_error("Tensor dtype mismatch");                         \
  }
#define TORCH_CHECK_TENSOR_SHAPE(T, ...)                                       \
  if (!((T).sizes() == torch::IntArrayRef{__VA_ARGS__})) {                     \
    std::cout << "Tensor Shape:" << (T).sizes() << std::endl;                  \
    throw std::runtime_error("values must have shape " #__VA_ARGS__);          \
  }
