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

namespace hpc::cu {
template <typename ElementType, const size_t NumElements,
          const size_t kBlockSize = hpc::CBLOCK_SIZE_1D>
torch::Tensor act_wrapper(torch::Tensor in, torch::Dtype expected_dtype,
                          void (*kernel_func)(ElementType *,
                                              const ElementType *, size_t)) {
  CHECK_TORCH_TENSOR_DTYPE(in, expected_dtype);
  auto out = torch::empty_like(in);
  const int ndim = in.dim();
  int N = 1;
  for (int i = 0; i < ndim; ++i) {
    N *= in.size(i);
  }

  if (ndim != 2) {
    dim3 block(kBlockSize / NumElements);
    dim3 grid((N + kBlockSize - 1) / kBlockSize);
    kernel_func<<<grid, block>>>(
        reinterpret_cast<ElementType *>(out.data_ptr()),
        reinterpret_cast<ElementType *>(in.data_ptr()), N);
  } else {
    const int S = in.size(0);
    const int K = in.size(1);
    const int total_N = S * K;

    if ((K / NumElements) <= hpc::CBLOCK_SIZE_1D_MAXIMUM) {
      dim3 block(K / NumElements);
      dim3 grid(S);
      kernel_func<<<grid, block>>>(
          reinterpret_cast<ElementType *>(out.data_ptr()),
          reinterpret_cast<ElementType *>(in.data_ptr()), total_N);
    } else {
      dim3 block(kBlockSize / NumElements);
      dim3 grid((N + kBlockSize - 1) / kBlockSize);
      kernel_func<<<grid, block>>>(
          reinterpret_cast<ElementType *>(out.data_ptr()),
          reinterpret_cast<ElementType *>(in.data_ptr()), N);
    }
  }
  return out;
}

template <typename ElementType, const size_t NumElements,
          const size_t kBlockSize = hpc::CBLOCK_SIZE_1D>
torch::Tensor reduce_wrapper(torch::Tensor in, torch::Dtype expected_dtype,
                             void (*kernel_func)(ElementType *,
                                                 const ElementType *, size_t)) {
  CHECK_TORCH_TENSOR_DTYPE(in, expected_dtype);
  const int ndim = in.dim();
  if (ndim != 1) {
    throw std::runtime_error("Input tensor must be 1D");
  }
  int N = 1;
  for (int i = 0; i < ndim; ++i) {
    N *= in.size(i);
  }

  auto out = torch::empty({1}, in.options());

  dim3 block(kBlockSize / NumElements);
  dim3 grid((N + kBlockSize - 1) / kBlockSize);
  kernel_func<<<grid, block>>>(
      reinterpret_cast<ElementType *>(out.data_ptr()),
      reinterpret_cast<ElementType *>(in.data_ptr()), N);
  return out;
}

template <typename ElementType, const size_t NumElements,
          const size_t kBlockSize = hpc::CBLOCK_SIZE_1D>
torch::Tensor dot_wrapper(torch::Tensor in1, torch::Tensor in2,
                          torch::Dtype expected_dtype,
                          void (*kernel_func)(ElementType *, ElementType *,
                                              ElementType *, size_t)) {
  CHECK_TORCH_TENSOR_DTYPE(in1, expected_dtype);
  CHECK_TORCH_TENSOR_DTYPE(in2, expected_dtype);
  const int ndim1 = in1.dim();
  const int ndim2 = in2.dim();
  if (ndim1 != 1 && ndim2 != 1) {
    throw std::runtime_error("Input tensors must be 1D");
  }
  if (ndim1 != ndim2) {
    throw std::runtime_error("Input tensors must have the same number of dims");
  }

  int N = 1;
  for (int i = 0; i < ndim1; ++i) {
    if (in1.size(i) != in2.size(i)) {
      throw std::runtime_error("Input tensors must have the same shape");
    }
    N *= in1.size(i);
  }

  auto out = torch::empty({1}, in1.options());

  dim3 block(kBlockSize / NumElements);
  dim3 grid((N + kBlockSize - 1) / kBlockSize);
  kernel_func<<<grid, block>>>(
      reinterpret_cast<ElementType *>(out.data_ptr()),
      reinterpret_cast<ElementType *>(in1.data_ptr()),
      reinterpret_cast<ElementType *>(in2.data_ptr()), N);

  return out;
}

} // namespace hpc::cu
