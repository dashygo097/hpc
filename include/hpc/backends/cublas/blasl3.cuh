#pragma once

#if defined(HPC_ENABLE_CUDA) && defined(HPC_ENABLE_CUBLAS)
#include <cublas_v2.h>
#endif

#if defined(HPC_ENABLE_CUDA) && defined(HPC_ENABLE_CUBLAS)
namespace hpc::cublas {} // namespace hpc::cublas
#endif
