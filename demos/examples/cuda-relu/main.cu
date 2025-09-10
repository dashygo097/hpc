#include <hpc.hh>

using namespace hpc;

const size_t DSIZE = 1000000;

int main() {
    float* output = cuda::cudaMallocDevice<float>(DSIZE);
    float* input  = cuda::cudaMallocDevice<float>(DSIZE);
    CUDA_LAUNCH_1D(cuda::relu_fp32x4_kernel, DSIZE, 256, output, input, DSIZE);
    return 0; 
}
