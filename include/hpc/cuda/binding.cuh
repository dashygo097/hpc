#pragma once

#include "../constants.hh"
#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));
#define TORCH_BINDING_TEMPLATE_EXTENSION(func_name, func_impl)                 \
  m.def(func_name, &func_impl, func_name);

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

namespace hpc::cuda {
template <typename ElementType, const size_t NumElements,
          const size_t kBlockSize = hpc::CBLOCK_SIZE_1D>
torch::Tensor act_wrapper(torch::Tensor input, torch::Dtype expected_dtype,
                          void (*kernel_func)(ElementType *,
                                              const ElementType *, size_t)) {
  CHECK_TORCH_TENSOR_DTYPE(input, expected_dtype);
  auto output = torch::empty_like(input);
  const int ndim = input.dim();
  int N = 1;
  for (int i = 0; i < ndim; ++i) {
    N *= input.size(i);
  }

  if (ndim != 2) {
    dim3 block(kBlockSize / NumElements);
    dim3 grid((N + kBlockSize - 1) / kBlockSize);
    kernel_func<<<grid, block>>>(
        reinterpret_cast<ElementType *>(output.data_ptr()),
        reinterpret_cast<ElementType *>(input.data_ptr()), N);
  } else {
    const int S = input.size(0);
    const int K = input.size(1);
    const int total_N = S * K;

    if ((K / NumElements) <= hpc::CBLOCK_SIZE_1D_MAXIMUM) {
      dim3 block(K / NumElements);
      dim3 grid(S);
      kernel_func<<<grid, block>>>(
          reinterpret_cast<ElementType *>(output.data_ptr()),
          reinterpret_cast<ElementType *>(input.data_ptr()), total_N);
    } else {
      dim3 block(kBlockSize / NumElements);
      dim3 grid((N + kBlockSize - 1) / kBlockSize);
      kernel_func<<<grid, block>>>(
          reinterpret_cast<ElementType *>(output.data_ptr()),
          reinterpret_cast<ElementType *>(input.data_ptr()), N);
    }
  }
  return output;
}

template <typename ElementType, const size_t NumElements,
          const size_t kBlockSize = hpc::CBLOCK_SIZE_1D>
torch::Tensor reduce_wrapper(torch::Tensor input, torch::Dtype expected_dtype,
                             void (*kernel_func)(ElementType *,
                                                 const ElementType *, size_t)) {
  CHECK_TORCH_TENSOR_DTYPE(input, expected_dtype);
  const int ndim = input.dim();

  int N = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    N *= input.size(i);
  }

  auto output = torch::empty({1}, input.options());

  if (ndim != 2) {
    dim3 block(kBlockSize / NumElements);
    dim3 grid((N + kBlockSize - 1) / kBlockSize);
    kernel_func<<<grid, block>>>(
        reinterpret_cast<ElementType *>(output.data_ptr()),
        reinterpret_cast<ElementType *>(input.data_ptr()), N);
  } else {
    const int S = input.size(0);
    const int K = input.size(1);
    const int total_N = S * K;

    if ((K / NumElements) <= hpc::CBLOCK_SIZE_1D_MAXIMUM) {
      dim3 block(K / NumElements);
      dim3 grid(S);
      kernel_func<<<grid, block>>>(
          reinterpret_cast<ElementType *>(output.data_ptr()),
          reinterpret_cast<ElementType *>(input.data_ptr()), total_N);
    } else {
      dim3 block(kBlockSize / NumElements);
      dim3 grid((N + kBlockSize - 1) / kBlockSize);
      kernel_func<<<grid, block>>>(
          reinterpret_cast<ElementType *>(output.data_ptr()),
          reinterpret_cast<ElementType *>(input.data_ptr()), N);
    }
  }
}
} // namespace hpc::cuda
