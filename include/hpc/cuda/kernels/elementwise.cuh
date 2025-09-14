#pragma once

#include <cuda_fp16.h>
#include <memory>

namespace hpc::cu {
__global__ void elementwise_add_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N);
__global__ void elementwise_add_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N);
__global__ void elementwise_sub_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N);
__global__ void elementwise_sub_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N);
__global__ void elementwise_mul_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N);
__global__ void elementwise_mul_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N);
__global__ void elementwise_div_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N);
__global__ void elementwise_div_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N);
} // namespace hpc::cu
