#include <hpc.hh>

using namespace hpc;

const size_t DSIZE = 10000000;

int main() {
  float *h_o = cu::cudaMallocH<float>(DSIZE);
  float *h_i = cu::cudaMallocH<float>(DSIZE);
  float *d_o = cu::cudaMallocD<float>(DSIZE);
  float *d_i = cu::cudaMallocD<float>(DSIZE);
  for (size_t i = 0; i < DSIZE; i++) {
    h_i[i] = (i % 2 == 0) ? float(i) : float(-i);
  }
  cudaMemcpy(d_i, h_i, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckLast("Memory copy host to device");

  CUDA_LAUNCH_1D(cu::relu_fp32x4_kernel, DSIZE / 4, CBLOCK_SIZE_1D, d_o, d_i,
                 DSIZE);

  cudaMemcpy(h_o, d_o, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckLast("Memory copy device to host");

  for (size_t i = 0; i < DSIZE; i++) {
    if (h_o[i] != std::max(0.0f, h_i[i])) {
      std::cout << "Error at index " << i << ": " << h_o[i]
                << " != " << std::max(0.0f, h_i[i]) << std::endl;
      return -1;
    }
  }
  std::cout << "Success" << std::endl;
  

  return 0;
}
