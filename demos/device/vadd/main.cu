#include <hpc.hh>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void vadd_kernel(float* dst, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::cout << "=== 简单CUDA向量加法测试 ===\n" << std::endl;
    
    std::cout << "1. 初始化CUDA..." << std::endl;
    try {
        hpc::cu::init(0, true);
    } catch (const std::exception& e) {
        std::cerr << "CUDA初始化失败: " << e.what() << std::endl;
        return 1;
    }
    
    const int N = 1024 * 1024; 
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    std::cout << "\n2. 测试参数:" << std::endl;
    std::cout << "   向量大小: " << N << " 个元素" << std::endl;
    std::cout << "   块大小: " << block_size << " 线程" << std::endl;
    std::cout << "   网格大小: " << grid_size << " 个块" << std::endl;
    
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N, 0.0f);
    std::vector<float> h_expected(N);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
        h_expected[i] = h_a[i] + h_b[i];
    }
    
    std::cout << "\n3. 分配设备内存..." << std::endl;
    hpc::cu::DeviceBuffer<float> d_a(N);
    hpc::cu::DeviceBuffer<float> d_b(N);
    hpc::cu::DeviceBuffer<float> d_c(N);
    
    std::cout << "   设备缓冲区创建成功" << std::endl;
    
    std::cout << "\n4. 复制数据到设备..." << std::endl;
    d_a.copyFromHost(h_a.data());
    d_b.copyFromHost(h_b.data());
    std::cout << "   数据复制完成" << std::endl;
    
    std::cout << "\n5. 启动CUDA内核..." << std::endl;
    
    hpc::cu::sync();
    CUDA_LAUNCH(vadd_kernel, N, block_size, d_c.data(), d_a.data(), d_b.data(), N);
    
    hpc::cu::sync();

    std::cout << "\n6. 复制结果回主机..." << std::endl;
    d_c.copyToHost(h_c.data());
    
    std::cout << "\n7. 验证结果..." << std::endl;
    bool passed = true;
    int errors = 0;
    const int max_errors_to_show = 10;
    
    for (int i = 0; i < N; i++) {
        float diff = std::abs(h_c[i] - h_expected[i]);
        if (diff > 1e-5f) {
            errors++;
            if (errors <= max_errors_to_show) {
                std::cout << "   错误 #" << errors << " 在索引 " << i 
                          << ": 期望值=" << h_expected[i] 
                          << ", 实际值=" << h_c[i] 
                          << ", 差值=" << diff << std::endl;
            }
            passed = false;
        }
    }
    
    std::cout << "\n8. 测试结果:" << std::endl;
    if (passed) {
        std::cout << "   ✓ 测试通过! 所有 " << N << " 个元素计算正确" << std::endl;
    } else {
        std::cout << "   ✗ 测试失败!" << std::endl;
        std::cout << "   总共发现 " << errors << " 个错误" << std::endl;
        if (errors > max_errors_to_show) {
            std::cout << "   (只显示前 " << max_errors_to_show << " 个错误)" << std::endl;
        }
    }
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    
    return passed ? 0 : 1;
}