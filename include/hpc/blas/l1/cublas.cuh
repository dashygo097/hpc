#pragma once

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS) && defined(__CUDACC__)
#include "../../backends/backends.hh"
#endif

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS) && defined(__CUDACC__)
#define ENABLE_CUBLAS_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_CUBLAS_BINARY_BRANCH(name)
#define ENABLE_CUBLAS_UNARY_BRANCH(name)
#define ENABLE_CUBLAS_SCALAR_BRANCH(name)
#define ENABLE_CUBLAS_REDUCE_BRANCH(name)
#define ENABLE_CUBLAS_REDUCE2_BRANCH(name)
// #define ENABLE_CUBLAS_VECTOR_SCALAR_BRANCH(name) \
//   else if constexpr (backend == Backend::CUBLAS) { \
//     details::name##_cuda<T, BackendParams...>(n, dst, src, scalar); \
//   }
// #define ENABLE_CUBLAS_BINARY_BRANCH(name) \
//   else if constexpr (backend == Backend::CUBLAS) { \
//     details::name##_cuda<T, BackendParams...>(n, dst, src1, src2); \
//   }
// #define ENABLE_CUBLAS_UNARY_BRANCH(name) \
//   else if constexpr (backend == Backend::CUBLAS) { \
//     details::name##_cuda<T, BackendParams...>(n, dst, src); \
//   }
// #define ENABLE_CUBLAS_SCALAR_BRANCH(name) \
//   else if constexpr (backend == Backend::CUBLAS) { \
//     details::name##_cuda<T, BackendParams...>(n, dst, scalar); \
//   }
// #define ENABLE_CUBLAS_REDUCE_BRANCH(name) \
//   else if constexpr (backend == Backend::CUBLAS) { \
//     return details::name##_cuda<T, BackendParams...>(n, src); \
//   }
// #define ENABLE_CUBLAS_REDUCE2_BRANCH(name) \
//   else if constexpr (backend == Backend::CUBLAS) { \
//     return details::name##_cuda<T, BackendParams...>(n, src1, src2); \
//   }
#else
#define ENABLE_CUBLAS_VECTOR_SCALAR_BRANCH(name)
#define ENABLE_CUBLAS_BINARY_BRANCH(name)
#define ENABLE_CUBLAS_UNARY_BRANCH(name)
#define ENABLE_CUBLAS_SCALAR_BRANCH(name)
#define ENABLE_CUBLAS_REDUCE_BRANCH(name)
#define ENABLE_CUBLAS_REDUCE2_BRANCH(name)
#endif

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS) && defined(__CUDACC__)
namespace hpc::l1 {
namespace details {}
} // namespace hpc::l1
#endif
