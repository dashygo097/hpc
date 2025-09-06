#pragma once
#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_ACT(act_type, packed_type, th_type, element_type,        \
                          n_elements)                                          \
  torch::Tensor act_type##_##packed_type(torch::Tensor input) {                \
    CHECK_TORCH_TENSOR_DTYPE(input, (th_type))                                 \
    auto output = torch::empty_like(input);                                    \
    const int ndim = input.dim();                                              \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= input.size(i);                                                    \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      hpc::cuda::act_type##_##packed_type##_kernel<<<grid, block>>>(           \
          reinterpret_cast<element_type *>(output.data_ptr()),                 \
          reinterpret_cast<element_type *>(input.data_ptr()), N);              \
    } else {                                                                   \
      const int S = input.size(0);                                             \
      const int K = input.size(1);                                             \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        hpc::cuda::act_type##_##packed_type##_kernel<<<grid, block>>>(         \
            reinterpret_cast<element_type *>(output.data_ptr()),               \
            reinterpret_cast<element_type *>(input.data_ptr()), N);            \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= input.size(i);                                                  \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        hpc::cuda::act_type##_##packed_type##_kernel<<<grid, block>>>(         \
            reinterpret_cast<element_type *>(output.data_ptr()),               \
            reinterpret_cast<element_type *>(input.data_ptr()), N);            \
      }                                                                        \
    }                                                                          \
    return output;                                                             \
  }
