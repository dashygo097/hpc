#pragma once

#ifdef ENABLE_CUDA
#include "./err.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_CUDA
#define FLOAT2(ptr) (*reinterpret_cast<float2 *>(ptr))
#define FLOAT4(ptr) (*reinterpret_cast<float4 *>(ptr))
#define INT2(ptr) (*reinterpret_cast<int2 *>(ptr))
#define INT4(ptr) (*reinterpret_cast<int4 *>(ptr))

#define MAX_EXP_FP32 88.3762626647949f
#define MIN_EXP_FP32 -88.3762626647949f

#define HALF2(ptr) (*reinterpret_cast<half2 *>(ptr))
#define BFLOAT2(ptr) (*reinterpret_cast<__nv_bfloat162 *>(ptr))
#define MAX_EXP_FP16 __float2half(11.089866488461016f)
#define MIN_EXP_FP16 __float2half(-9.704060527839234f)
#endif
