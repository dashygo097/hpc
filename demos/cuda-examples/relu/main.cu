#include <hpc.hh>

using namespace hpc;

const size_t DSIZE = 1000000;

int main() {
  float *output = cu::cudaMallocDevice<float>(DSIZE);
  float *input = cu::cudaMallocDevice<float>(DSIZE);
  for (size_t i = 0; i < DSIZE; i++) {
    input[i] = (i % 2 == 0) ? float(i) : float(-i);
  }
  CUDA_LAUNCH_1D(cu::relu_fp32x4_kernel, DSIZE, 256, output, input, DSIZE);

  for (size_t i = 0; i < 10; i++) {
    if (output[i] != std::max(0.0f, input[i])) {
      std::cout << "Error at index " << i << ": " << output[i]
                << " != " << std::max(0.0f, input[i]) << std::endl;
      return -1;
    }
  }
  return 0;
}
