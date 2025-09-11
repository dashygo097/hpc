#include <hpc.hh>

using namespace hpc;

const size_t DSIZE = 102400;

int main() {
  float *h_o = cu::cudaMallocH<float>(1);
  float *h_i = cu::cudaMallocH<float>(DSIZE);
  float *d_o = cu::cudaMallocD<float>(1);
  float *d_i = cu::cudaMallocD<float>(DSIZE);
  for (size_t i = 0; i < DSIZE; i++) {
    h_i[i] = 1.0f;
  }
  cudaMemcpy(d_i, h_i, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckLast("Memory copy host to device");

  CUDA_LAUNCH_1D(cu::reduce_sum_fp32_kernel<CBLOCK_SIZE_1D>, DSIZE,
                 CBLOCK_SIZE_1D, d_o, d_i, DSIZE);

  cudaMemcpy(h_o, d_o, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckLast("Memory copy device to host");

  if ((h_o[0] - DSIZE) > 1e-5) {
    std::cout << "Error at index " << 0 << ": " << h_o[0]
              << " != " << float(DSIZE) << std::endl;
    return -1;
  }

  return 0;
}
