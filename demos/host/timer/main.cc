#include <hpc.hh>
#include <iostream>

void work(size_t n) {
  volatile double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += std::sin(i * 0.001);
  }
}

int main() {
  const size_t N = 100'000'000;

  std::cout << "Timing different backends\n";
  std::cout << std::string(50, '=') << "\n\n";

  // Sequential
  {
    __TIME_SEQUENTIAL("Sequential work");
    work(N);
  }

#ifdef HAS_OPENMP
  // OpenMP
  {
    __TIME_OPENMP("OpenMP work");
#pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
      volatile double x = std::sin(i * 0.001);
    }
  }
#endif

  // Manual timer for comparison
  {
    hpc::Timer timer(hpc::Backend::SEQUENTIAL, "Manual comparison");
    timer.start();
    work(N / 10);
    timer.stop();

    std::cout << "Throughput: " << (N / 10 / timer.elapsed() / 1e6)
              << " Mops/s\n";
  }

  return 0;
}
