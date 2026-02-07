#pragma once

#ifdef HPC_ENABLE_CUDA
#define HPC_ENABLE_GEMM_CUDA_BRANCH(name)
#else
#define HPC_ENABLE_GEMM_CUDA_BRANCH(name)
#endif
