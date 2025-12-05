#include <hpc.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
  std::cout << "=== CUDA Vector Addition Test (Using HPC L1 API) ===\n" << std::endl;

  std::cout << "1.  Initializing CUDA..." << std::endl;
  try {
    hpc::cu::init(0, true);
  } catch (const std::exception &e) {
    std::cerr << "CUDA initialization failed: " << e. what() << std::endl;
    return 1;
  }

  const int N = 1024 * 1024;
  std::cout << "\n2. Test parameters:" << std::endl;
  std::cout << "   Vector size: " << N << " elements" << std::endl;

  // Allocate host memory
  std::vector<float> h_a(N);
  std::vector<float> h_b(N);
  std::vector<float> h_c(N, 0.0f);
  std::vector<float> h_expected(N);

  // Initialize data
  std::cout << "\n3.  Initializing test data..." << std::endl;
  for (int i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(2 * i);
    h_expected[i] = h_a[i] + h_b[i];
  }
  std::cout << "   Data initialization complete" << std::endl;

  // Allocate device memory
  std::cout << "\n4. Allocating device memory..." << std::endl;
  hpc::cu::DeviceBuffer<float> d_a(N);
  hpc::cu::DeviceBuffer<float> d_b(N);
  hpc::cu::DeviceBuffer<float> d_c(N);
  std::cout << "   Device buffers created successfully" << std::endl;

  // Copy data to device
  std::cout << "\n5.  Copying data to device..." << std::endl;
  d_a.copyFromHost(h_a.data());
  d_b.copyFromHost(h_b.data());
  std::cout << "   Data copy complete" << std::endl;

  // Launch kernel using HPC L1 API
  std::cout << "\n6.  Launching CUDA kernel via HPC L1 API..." << std::endl;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  try {
    hpc::l1::vadd<float, hpc::Backend::CUDA, 256>(
      d_c.data(), d_a.data(), N
    );
    hpc::l1::vadd<float, hpc::Backend::CUDA, 256>(
      d_c.data(), d_b.data(), N
    );
 
  } catch (const std::exception &e) {
    std::cerr << "Kernel launch failed: " << e. what() << std::endl;
    return 1;
  }
  
  hpc::cu::sync();
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  std::cout << "   Kernel execution time: " << duration.count() / 1000.0 << " ms" << std::endl;
  
  // Copy result back to host
  std::cout << "\n7. Copying results back to host..." << std::endl;
  d_c.copyToHost(h_c.data());

  // Verify results
  std::cout << "\n8.  Verifying results..." << std::endl;
  bool passed = true;
  int errors = 0;
  const int max_errors_to_show = 10;
  double max_diff = 0.0;

  for (int i = 0; i < N; i++) {
    double diff = std::abs(h_c[i] - h_expected[i]);
    max_diff = std::max(max_diff, diff);
    
    if (diff > 1e-5) {
      errors++;
      if (errors <= max_errors_to_show) {
        std::cout << "   Error #" << errors << " at index " << i
                  << ": expected=" << h_expected[i] 
                  << ", actual=" << h_c[i]
                  << ", diff=" << diff << std::endl;
      }
      passed = false;
    }
  }

  // Print results
  std::cout << "\n9. Test results:" << std::endl;
  std::cout << "   Total elements: " << N << std::endl;
  std::cout << "   Max difference: " << max_diff << std::endl;
  
  if (passed) {
    std::cout << "   ✓ TEST PASSED! All " << N << " elements computed correctly" << std::endl;
  } else {
    std::cout << "   ✗ TEST FAILED!" << std::endl;
    std::cout << "   Total errors found: " << errors << std::endl;
    if (errors > max_errors_to_show) {
      std::cout << "   (only showing first " << max_errors_to_show << " errors)" << std::endl;
    }
  }
  
  // Performance metrics
  std::cout << "\n10. Performance metrics:" << std::endl;
  double bandwidth_gb = (3.0 * N * sizeof(float)) / (duration.count() * 1e-6) / 1e9;
  std::cout << "   Bandwidth: " << bandwidth_gb << " GB/s" << std::endl;

  std::cout << "\n=== Test Complete ===" << std::endl;

  return passed ? 0 : 1;
}