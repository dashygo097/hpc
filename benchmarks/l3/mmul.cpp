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
void mmul_ref(T *C, const T *A, const T *B, size_t M, size_t K, size_t N) {
  for (size_t i = 0; i < M * N; ++i)
    C[i] = T{};
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Fixture
template <typename T> class MMulFixture : public benchmark::Fixture {
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
    init_matrix(C.data(), M, N, T(0));
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

using MMulFixtureFloat = MMulFixture<float>;

// Macro for benchmark definition
#define DEFINE_MMUL_BENCHMARK(Name, Function, ...)                             \
  BENCHMARK_DEFINE_F(MMulFixtureFloat, Name)(benchmark::State & state) {       \
    for (auto _ : state) {                                                     \
      Function<float, ##__VA_ARGS__>(C.data(), A.data(), B.data(), M, K, N);   \
      benchmark::DoNotOptimize(C.data());                                      \
      benchmark::ClobberMemory();                                              \
    }                                                                          \
    if (state.thread_index() == 0) {                                           \
      mmul_ref(C_ref.data(), A.data(), B.data(), M, K, N);                     \
      if (!verify_result(C.data(), C_ref.data(), M, N)) {                      \
        state.SkipWithError("Verification failed!");                           \
      }                                                                        \
    }                                                                          \
    double ops = 2.0 * M * K * N;                                              \
    state.counters["GFLOPS"] =                                                 \
        benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate, \
                           benchmark::Counter::kIs1000);                       \
  }                                                                            \
  BENCHMARK_REGISTER_F(MMulFixtureFloat, Name)                                 \
      ->Args({256, 256, 256})                                                  \
      ->Args({512, 512, 512})                                                  \
      ->Args({1024, 1024, 1024})                                               \
      ->Unit(benchmark::kMillisecond);

// Naive
BENCHMARK_DEFINE_F(MMulFixtureFloat, Naive)(benchmark::State &state) {
  for (auto _ : state) {
    mmul_ref(C.data(), A.data(), B.data(), M, K, N);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  double ops = 2.0 * M * K * N;
  state.counters["GFLOPS"] =
      benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}
BENCHMARK_REGISTER_F(MMulFixtureFloat, Naive)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Unit(benchmark::kMillisecond);

// Sequential tiled benchmarks
DEFINE_MMUL_BENCHMARK(Seq_32, hpc::l3::details::mmul_seq, 32, 64)
DEFINE_MMUL_BENCHMARK(Seq_64, hpc::l3::details::mmul_seq, 64, 64)
DEFINE_MMUL_BENCHMARK(Seq_128, hpc::l3::details::mmul_seq, 128, 64)
DEFINE_MMUL_BENCHMARK(SeqAPI, hpc::l3::mmul, hpc::Backend::SEQUENTIAL, 128, 64)

// OpenMP benchmarks
#ifdef ENABLE_OPENMP
DEFINE_MMUL_BENCHMARK(OpenMP_32, hpc::l3::details::mmul_omp, 32, 64)
DEFINE_MMUL_BENCHMARK(OpenMP_64, hpc::l3::details::mmul_omp, 64, 64)
DEFINE_MMUL_BENCHMARK(OpenMP_128, hpc::l3::details::mmul_omp, 128, 64)
DEFINE_MMUL_BENCHMARK(OpenMPAPI, hpc::l3::mmul, hpc::Backend::OPENMP, 128, 64)
#endif

#ifdef ENABLE_SIMD
DEFINE_MMUL_BENCHMARK(SIMD_32_4, hpc::l3::details::mmul_simd, 32, 4, 64)
DEFINE_MMUL_BENCHMARK(SIMD_64_4, hpc::l3::details::mmul_simd, 64, 4, 64)
DEFINE_MMUL_BENCHMARK(SIMD_128_4, hpc::l3::details::mmul_simd, 128, 4, 64)
DEFINE_MMUL_BENCHMARK(SIMDAPI, hpc::l3::mmul, hpc::Backend::SIMD, 128, 4, 64)
#endif

#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
DEFINE_MMUL_BENCHMARK(OpenMP_SIMD_32_4, hpc::l3::details::mmul_omp_simd, 32, 4,
                      64)
DEFINE_MMUL_BENCHMARK(OpenMP_SIMD_64_4, hpc::l3::details::mmul_omp_simd, 64, 4,
                      64)
DEFINE_MMUL_BENCHMARK(OpenMP_SIMD_128_4, hpc::l3::details::mmul_omp_simd, 128,
                      4, 64)
DEFINE_MMUL_BENCHMARK(OpenMP_SIMD_API, hpc::l3::mmul, hpc::Backend::OPENMP_SIMD,
                      128, 4, 64)
#endif

#ifdef ENABLE_ACCELERATE
DEFINE_MMUL_BENCHMARK(Accelerate, hpc::l3::details::mmul_acceler)
DEFINE_MMUL_BENCHMARK(AccelerateAPI, hpc::l3::mmul, hpc::Backend::ACCELERATE)
#endif

BENCHMARK_MAIN();
