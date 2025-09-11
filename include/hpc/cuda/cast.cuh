#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define INT2(value) *reinterpret_cast<int2 *>(value)
#define INT4(value) *reinterpret_cast<int4 *>(value)
#define FLOAT2(value) *reinterpret_cast<float2 *>(value)
#define FLOAT4(value) *reinterpret_cast<float4 *>(value)
#define HALF2(value) *reinterpret_cast<half2 *>(value)
#define BFLOAT2(value) *reinterpret_cast<__nv_bfloat162 *>(value)
