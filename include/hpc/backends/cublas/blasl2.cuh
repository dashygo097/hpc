#pragma once

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS)
#include <cublas_v2.h>
#endif

#if defined(ENABLE_CUDA) && defined(ENABLE_CUBLAS)
namespace hpc::cublas {} // namespace hpc::cublas
#endif
