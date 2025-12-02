#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define INT2(value) *reinterpret_cast<int2 *>(value)
#define INT4(value) *reinterpret_cast<int4 *>(value)
#define FLOAT2(value) *reinterpret_cast<float2 *>(value)
#define FLOAT4(value) *reinterpret_cast<float4 *>(value)
#define HALF2(value) *reinterpret_cast<half2 *>(value)
#define BFLOAT2(value) *reinterpret_cast<__nv_bfloat162 *>(value)

#define MAX_EXP_FP32 88.3762626647949f
#define MIN_EXP_FP32 -88.3762626647949f
#define MAX_EXP_FP16 __float2half(11.089866488461016f)
#define MIN_EXP_FP16 __float2half(-9.704060527839234f)
