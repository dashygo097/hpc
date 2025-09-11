#include "hpc/cuda/cast.cuh"
#include "hpc/cuda/kernels/dot_prod.cuh"

namespace hpc::cu {
template <const size_t kBlockSize>
__global__ void dot_prod_fp32_kernel(float *out, float *in1, float *in2,
                                     size_t N) {
  __shared__ float sdata[kBlockSize / CWARP_SIZE];
  size_t tid = threadIdx.x;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t lane = threadIdx.x % CWARP_SIZE;
  size_t wid = threadIdx.x / CWARP_SIZE;
  size_t mask = 0xffffffff;

  float sum = 0.0f;
}

template <const size_t kBlockSize>
__global__ void dot_prod_fp16_kernel(half *out, half *in1, half *in2,
                                     size_t N) {
  __shared__ float sdata[kBlockSize / CWARP_SIZE];
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t lane = threadIdx.x % CWARP_SIZE;
  size_t wid = threadIdx.x / CWARP_SIZE;
  size_t mask = 0xffffffff;

  float sum = 0.0f;
}

} // namespace hpc::cu
