#pragma once

#ifdef ENABLE_CUDA
#endif

namespace hpc::cuda {
#ifdef ENABLE_CUDA
__global__ void relu_forward_f32(float *output, const float *input,
                                 size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void relu_forward_f16(__half *output, const __half *input,
                                 size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] =
        __hgt(input[idx], __float2half(0.0f)) ? input[idx] : __float2half(0.0f);
  }
}

__global__ void relu_backwards_f32(float *output, const float *input,
                                   size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0f ? 1.0f : 0.0f;
  }
}

__global__ void relu_backwawrds_f16(__half *output, const __half *input,
                                    size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __hgt(input[idx], __float2half(0.0f)) ? __float2half(1.0f)
                                                        : __float2half(0.0f);
  }
}

#endif
} // namespace hpc::cuda
