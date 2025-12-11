#pragma once

#ifdef ENABLE_CUDA
#define ENABLE_GEMM_CUDA_BRANCH(name)
#else
#define ENABLE_GEMM_CUDA_BRANCH(name)
#endif
