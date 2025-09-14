#include "hpc/cuda/kernels/elementwise.cuh"

namespace hpc::cu {
__global__ void elementwise_add_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = in1[idx] + in2[idx];
  }
}
__global__ void elementwise_add_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = __hadd(in1[idx], in2[idx]);
  }
}

__global__ void elementwise_sub_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = in1[idx] - in2[idx];
  }
}
__global__ void elementwise_sub_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = __hsub(in1[idx], in2[idx]);
  }
}

__global__ void elementwise_mul_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = in1[idx] * in2[idx];
  }
}
__global__ void elementwise_mul_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = __hmul(in1[idx], in2[idx]);
  }
}

__global__ void elementwise_div_fp32_kernel(float *out, float *in1, float *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = in1[idx] / in2[idx];
  }
}
__global__ void elementwise_div_fp16_kernel(half *out, half *in1, half *in2,
                                            size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = __hdiv(in1[idx], in2[idx]);
  }
}

} // namespace hpc::cu
