#include <hpc.hh>

using namespace hpc;

const size_t DSIZE = 10000000;

int main() {
    float *h_o = cu::cudaMallocH<float>(1);
    float *h_i1 = cu::cudaMallocH<float>(DSIZE);
    float *h_i2 = cu::cudaMallocH<float>(DSIZE);
    float *d_o = cu::cudaMallocD<float>(1);
    float *d_i1 = cu::cudaMallocD<float>(DSIZE);
    float *d_i2 = cu::cudaMallocD<float>(DSIZE);

    for (size_t i = 0; i < DSIZE; i++) {
      h_i1[i] = 1.0f;
      h_i2[i] = 1.0f;
    }
    cudaMemcpy(d_i1, h_i1, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckLast("Memory copy host to device");
    cudaMemcpy(d_i2, h_i2, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckLast("Memory copy host to device");
    
    CUDA_LAUNCH_1D(cu::dot_prod_fp32_kernel<CBLOCK_SIZE_1D>, DSIZE,
                   CBLOCK_SIZE_1D, d_o, d_i1, d_i2, DSIZE);
    
    cudaMemcpy(h_o, d_o, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckLast("Memory copy device to host");
    
    if ((h_o[0] - DSIZE) > 1e-5) {
        std::cerr << "Error: incorrect result " << h_o[0] << " != " << DSIZE
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Success: " << h_o[0] << " == " << DSIZE << std::endl;
    
    return 0;
}