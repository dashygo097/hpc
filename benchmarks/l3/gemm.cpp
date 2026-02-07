#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <hpc.hh>
#include <random>
#include <vector>

// Helper functions
template <typename T>
void init_matrix(T *mat, size_t rows, size_t cols, T value) {
  for (size_t i = 0; i < rows * cols; ++i)
    mat[i] = value;
}

template <typename T>
void init_matrix_random(T *mat, size_t rows, size_t cols, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dis(-1.0, 1.0);
  for (size_t i = 0; i < rows * cols; ++i)
    mat[i] = dis(gen);
}

template <typename T>
bool verify_result(const T *C, const T *C_ref, size_t M, size_t N,
                   T tol = 1e-3) {
  for (size_t i = 0; i < M * N; ++i) {
    T diff = std::abs(C[i] - C_ref[i]);
    T max_val = std::max(std::abs(C[i]), std::abs(C_ref[i]));
    if (diff > tol * max_val && diff > tol)
      return false;
  }
  return true;
}

template <typename T>
void gemm_ref(size_t M, size_t K, size_t N, T *C, const T *A, const T *B) {
  for (size_t i = 0; i < M * N; ++i)
    C[i] = T{0};
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Fixture
template <typename T> class GEMMFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    M = state.range(0);
    K = state.range(1);
    N = state.range(2);
    A.resize(M * K);
    B.resize(K * N);
    C.resize(M * N);
    C_ref.resize(M * N);
    init_matrix_random(A.data(), M, K);
    init_matrix_random(B.data(), K, N);
    init_matrix(C.data(), M, N, T{0});
  }

  void TearDown(const ::benchmark::State &) override {
    A.clear();
    B.clear();
    C.clear();
    C_ref.clear();
  }

protected:
  size_t M, K, N;
  std::vector<T> A, B, C, C_ref;
};

using GEMMFixtureFloat = GEMMFixture<float>;

// Macro for benchmark definition
#define DEFINE_GEMM_BENCHMARK(Name, Function, ...)                             \
  BENCHMARK_DEFINE_F(GEMMFixtureFloat, Name)(benchmark::State & state) {       \
    for (auto _ : state) {                                                     \
      Function<float, ##__VA_ARGS__>(M, K, N, C.data(), A.data(), B.data());   \
      benchmark::DoNotOptimize(C.data());                                      \
      benchmark::ClobberMemory();                                              \
    }                                                                          \
    if (state.thread_index() == 0) {                                           \
      gemm_ref(M, K, N, C_ref.data(), A.data(), B.data());                     \
      if (!verify_result(C.data(), C_ref.data(), M, N)) {                      \
        state.SkipWithError("Verification failed!");                           \
      }                                                                        \
    }                                                                          \
    double ops = 2.0 * M * K * N;                                              \
    state.counters["GFLOPS"] =                                                 \
        benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate, \
                           benchmark::Counter::kIs1000);                       \
  }                                                                            \
  BENCHMARK_REGISTER_F(GEMMFixtureFloat, Name)                                 \
      ->Args({256, 256, 256})                                                  \
      ->Args({512, 512, 512})                                                  \
      ->Args({1024, 1024, 1024})                                               \
      ->Unit(benchmark::kMillisecond);

// Naive
BENCHMARK_DEFINE_F(GEMMFixtureFloat, Naive)(benchmark::State &state) {
  for (auto _ : state) {
    gemm_ref(M, K, N, C.data(), A.data(), B.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  double ops = 2.0 * M * K * N;
  state.counters["GFLOPS"] =
      benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}
BENCHMARK_REGISTER_F(GEMMFixtureFloat, Naive)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Unit(benchmark::kMillisecond);

// Sequential
DEFINE_GEMM_BENCHMARK(Seq_32, hpc::l3::gemm, hpc::Backend::SEQUENTIAL, 32, 64)
DEFINE_GEMM_BENCHMARK(Seq_64, hpc::l3::gemm, hpc::Backend::SEQUENTIAL, 64, 64)
DEFINE_GEMM_BENCHMARK(Seq_128, hpc::l3::gemm, hpc::Backend::SEQUENTIAL, 128, 64)

// OpenMP
#ifdef HPC_ENABLE_OPENMP
DEFINE_GEMM_BENCHMARK(OpenMP_32, hpc::l3::gemm, hpc::Backend::OPENMP, 32, 64)
DEFINE_GEMM_BENCHMARK(OpenMP_64, hpc::l3::gemm, hpc::Backend::OPENMP, 64, 64)
DEFINE_GEMM_BENCHMARK(OpenMP_128, hpc::l3::gemm, hpc::Backend::OPENMP, 128, 64)
#endif

// SIMD
#ifdef HPC_ENABLE_SIMD
DEFINE_GEMM_BENCHMARK(SIMD_32_4, hpc::l3::gemm, hpc::Backend::SIMD, 32, 4, 64)
DEFINE_GEMM_BENCHMARK(SIMD_64_4, hpc::l3::gemm, hpc::Backend::SIMD, 64, 4, 64)
DEFINE_GEMM_BENCHMARK(SIMD_128_4, hpc::l3::gemm, hpc::Backend::SIMD, 128, 4, 64)
#endif

// OpenMP + SIMD
#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
DEFINE_GEMM_BENCHMARK(OpenMP_SIMD_32_4, hpc::l3::gemm,
                      hpc::Backend::OPENMP_SIMD, 32, 4, 64)
DEFINE_GEMM_BENCHMARK(OpenMP_SIMD_64_4, hpc::l3::gemm,
                      hpc::Backend::OPENMP_SIMD, 64, 4, 64)
DEFINE_GEMM_BENCHMARK(OpenMP_SIMD_128_4, hpc::l3::gemm,
                      hpc::Backend::OPENMP_SIMD, 128, 4, 64)
#endif

// OpenBLAS
#ifdef HPC_ENABLE_OPENBLAS
DEFINE_GEMM_BENCHMARK(OpenBLAS, hpc::l3::gemm, hpc::Backend::OPENBLAS)
#endif

// Accelerate
#ifdef HPC_ENABLE_ACCELERATE
DEFINE_GEMM_BENCHMARK(Accelerate, hpc::l3::gemm, hpc::Backend::ACCELERATE)
#endif

BENCHMARK_MAIN();
