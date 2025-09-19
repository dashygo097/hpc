#include <hpc.hh>

using namespace hpc;

const size_t M = 1024;
const size_t K = 512; 
const size_t N = 1024;

int main() {
  float *h_A = cu::cudaMallocH<float>(M * K);
  float *h_B = cu::cudaMallocH<float>(K * N);
  float *h_C = cu::cudaMallocH<float>(M * N);

  float *d_A = cu::cudaMallocD<float>(M * K);
  float *d_B = cu::cudaMallocD<float>(K * N);
  float *d_C = cu::cudaMallocD<float>(M * N);

  for (size_t i = 0; i < M * K; i++) {
    h_A[i] = 1.0f;
  }
  for (size_t i = 0; i < K * N; i++) {
    h_B[i] = 2.0f;
  }
  for (size_t i = 0; i < M * N; i++) {
    h_C[i] = 0.0f;
  }

  cu::cudaMemcpyH2D<float>(d_A, h_A, M * K);
  cudaCheckLast("Memcpy A from host to device");
  cu::cudaMemcpyH2D<float>(d_B, h_B, K * N);
  cudaCheckLast("Memcpy B from host to device");

  cu::CudaKernelLauncher()
    .autoConfig2D(M, N, CBLOCK_SIZE_2D, CBLOCK_SIZE_2D)
    .launch(cu::sgemm_smem_kernel<CBLOCK_SIZE_2D>,
            d_C, d_A, d_B, M, K, N, 1.0f, 0.0f);

  cu::cudaMemcpyD2H<float>(h_C, d_C, M * N);
  cudaCheckLast("Memcpy C from device to host");

  for (size_t i = 0; i < M * N; i++) {
    if (h_C[i] != 2.0f * K) {
      fprintf(stderr, "Error: h_C[%zu] = %f\n", i, h_C[i]);
      exit(1);
    }
  }
  std::cout << "Success!" << std::endl;

  return 0;
}
