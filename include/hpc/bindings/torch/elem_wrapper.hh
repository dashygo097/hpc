#pragma once

#if defined(ENABLE_PYTORCH) && defined(__CUDACC__)
#include "./macros.hh"
#include <torch/extension.h>
#include <torch/types.h>
#endif

#if defined(ENABLE_PYTORCH) && defined(__CUDACC__)
namespace hpc::bindings::torch {
template <typename ElementType, const size_t NumElements,
          const size_t BlockSize, typename Func>
::torch::Tensor elem_wrapper(Func kernel_func, ::torch::Tensor input,
                             ::torch::Dtype dtype) {
  TORCH_CHECK_TENSOR_DTYPE(input, dtype);
  auto output = ::torch::empty_like(input);

  const int ndim = input.dim();
  int n = 1;
  for (int i = 0; i < ndim; ++i) {
    n *= output.size(i);
  }

  dim3 block(BlockSize / NumElements);
  dim3 grid((n + BlockSize - 1) / BlockSize);

  kernel_func<<<grid, block>>>(
      reinterpret_cast<ElementType *>(output.data_ptr()),
      reinterpret_cast<const ElementType *>(n, input.data_ptr()));

  return output;
}
} // namespace hpc::bindings::torch
#endif
