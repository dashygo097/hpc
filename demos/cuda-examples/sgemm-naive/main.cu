#include <hpc.hh>

using namespace hpc;

const size_t DSIZE = 4096;

int main() {
  float *h_A = cu::cudaMallocH<float>(DSIZE * DSIZE);
  float *h_B = cu::cudaMallocH<float>(DSIZE * DSIZE);
  float *h_C = cu::cudaMallocH<float>(DSIZE * DSIZE);

  float *d_A = cu::cudaMallocD<float>(DSIZE * DSIZE);
  float *d_B = cu::cudaMallocD<float>(DSIZE * DSIZE);
  float *d_C = cu::cudaMallocD<float>(DSIZE * DSIZE);
  
  for (size_t i = 0; i < DSIZE * DSIZE; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f; 
    h_C[i] = 0.0f;
  }
  
  cu::cudaMemcpyH2D<float>(d_A, h_A, DSIZE * DSIZE);
  cudaCheckLast("Memcpy A from host to device");
  cu::cudaMemcpyH2D<float>(d_B, h_B, DSIZE * DSIZE);
  cudaCheckLast("Memcpy B from host to device");
  
  CUDA_LAUNCH_2D(cu::sgemm_naive_kernel, DSIZE, CBLOCK_SIZE_2D, CBLOCK_SIZE_2D, d_C, d_A, d_B, DSIZE, DSIZE, DSIZE, 1.0f, 0.0f);
  
  cu::cudaMemcpyD2H<float>(h_C, d_C, DSIZE * DSIZE);
  cudaCheckLast("Memcpy C from device to host");
  
  for (size_t i = 0; i < DSIZE * DSIZE; i++) {
    if (h_C[i] != 2.0f * DSIZE) {
      fprintf(stderr, "Error: h_C[%zu] = %f\n", i, h_C[i]);
      exit(1);
    }
  }
  std::cout << "Success!" << std::endl;

  return 0;
}
