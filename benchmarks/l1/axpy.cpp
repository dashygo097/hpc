#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <hpc.hh>
#include <random>
#include <vector>

// Helper functions
template <typename T> void init_vector(T *vec, size_t n) {
  for (size_t i = 0; i < n; ++i)
    vec[i] = static_cast<T>(i % 100);
}

template <typename T>
void init_vector_random(T *vec, size_t n, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dis(-1.0, 1.0);
  for (size_t i = 0; i < n; ++i)
    vec[i] = dis(gen);
}

template <typename T>
bool verify_result(const T *vec, const T *vec_ref, size_t n, T tol = 1e-5) {
  for (size_t i = 0; i < n; ++i) {
    T diff = std::abs(vec[i] - vec_ref[i]);
    T max_val = std::max(std::abs(vec[i]), std::abs(vec_ref[i]));
    if (diff > tol * max_val && diff > tol) {
      return false;
    }
  }
  return true;
}

// Reference implementation
template <typename T> void axpy_ref(size_t n, T *dst, const T *src, T alpha) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] += alpha * src[i];
  }
}

// Fixture
template <typename T> class AXPYFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    N = state.range(0);
    src.resize(N);
    dst.resize(N);
    dst_ref.resize(N);
    init_vector_random(src.data(), N);
    init_vector_random(dst.data(), N);
    std::copy(dst.begin(), dst.end(), dst_ref.begin());
  }

  void TearDown(const ::benchmark::State &) override {
    src.clear();
    dst.clear();
    dst_ref.clear();
  }

protected:
  size_t N;
  std::vector<T> src, dst, dst_ref;
};

using AXPYFixtureFloat = AXPYFixture<float>;

#define DEFINE_AXPY_VEC_BENCHMARK(Name, Function, ...)                         \
  BENCHMARK_DEFINE_F(AXPYFixtureFloat, Name)(benchmark::State & state) {       \
    std::vector<float> dst_copy = dst;                                         \
                                                                               \
    for (auto _ : state) {                                                     \
      Function<float, ##__VA_ARGS__>(N, dst_copy.data(), src.data(), 2.0);     \
      benchmark::DoNotOptimize(dst_copy.data());                               \
      benchmark::ClobberMemory();                                              \
    }                                                                          \
                                                                               \
    if (state.thread_index() == 0) {                                           \
      std::vector<float> test_dst = dst;                                       \
      std::vector<float> test_ref = dst;                                       \
                                                                               \
      Function<float, ##__VA_ARGS__>(N, test_dst.data(), src.data(), 2.0);     \
      axpy_ref<float>(N, test_ref.data(), src.data(), 2.0);                    \
                                                                               \
      if (!verify_result(test_dst.data(), test_ref.data(), N)) {               \
        state.SkipWithError("Verification failed!");                           \
      }                                                                        \
    }                                                                          \
                                                                               \
    double bytes =                                                             \
        3.0 * N * sizeof(float); /* read src, read dst, write dst */           \
    state.counters["GB/s"] = benchmark::Counter(                               \
        bytes, benchmark::Counter::kIsIterationInvariantRate,                  \
        benchmark::Counter::kIs1000);                                          \
    state.counters["N"] = N;                                                   \
  }                                                                            \
  BENCHMARK_REGISTER_F(AXPYFixtureFloat, Name)                                 \
      ->Args({256 * 256})                                                      \
      ->Args({1024 * 1024})                                                    \
      ->Args({4096 * 4096})                                                    \
      ->Unit(benchmark::kMillisecond);

// Naive Benchmark
BENCHMARK_DEFINE_F(AXPYFixtureFloat, Naive)(benchmark::State &state) {
  std::vector<float> dst_copy = dst;

  for (auto _ : state) {
    axpy_ref<float>(N, dst_copy.data(), src.data(), 2.0);
    benchmark::DoNotOptimize(dst_copy.data());
    benchmark::ClobberMemory();
  }

  double bytes = 3.0 * N * sizeof(float);
  state.counters["GB/s"] =
      benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

BENCHMARK_REGISTER_F(AXPYFixtureFloat, Naive)
    ->Args({256 * 256})
    ->Args({1024 * 1024})
    ->Args({4096 * 4096})
    ->Unit(benchmark::kMillisecond);

// Sequential
DEFINE_AXPY_VEC_BENCHMARK(Seq_8192, hpc::l1::axpy, hpc::Backend::SEQUENTIAL,
                          8192)

// OpenMP
#ifdef HPC_ENABLE_OPENMP
DEFINE_AXPY_VEC_BENCHMARK(OpenMP_8192, hpc::l1::axpy, hpc::Backend::OPENMP,
                          8192)
#endif

// SIMD
#ifdef HPC_ENABLE_SIMD
DEFINE_AXPY_VEC_BENCHMARK(SIMD_4, hpc::l1::axpy, hpc::Backend::SIMD, 4)
#endif

// OpenMP + SIMD
#if defined(HPC_ENABLE_OPENMP) && defined(HPC_ENABLE_SIMD)
DEFINE_AXPY_VEC_BENCHMARK(OpenMP_SIMD_8192_4, hpc::l1::axpy,
                          hpc::Backend::OPENMP_SIMD, 8192, 4)
#endif

// OpenBLAS
#ifdef HPC_ENABLE_OPENBLAS
DEFINE_AXPY_VEC_BENCHMARK(OpenBLAS, hpc::l1::axpy, hpc::Backend::OPENBLAS)
#endif

// Apple Accelerate
#ifdef HPC_ENABLE_ACCELERATE
DEFINE_AXPY_VEC_BENCHMARK(Accelerate, hpc::l1::axpy, hpc::Backend::ACCELERATE)
#endif

BENCHMARK_MAIN();
