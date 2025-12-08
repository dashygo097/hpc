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

// Reference implementation
template <typename T> T dot_ref(size_t n, const T *src1, const T *src2) {
  T result = 0;
  for (size_t i = 0; i < n; ++i) {
    result += src1[i] * src2[i];
  }
  return result;
}

// Fixture
template <typename T> class DotFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    N = state.range(0);
    src1.resize(N);
    src2.resize(N);
    init_vector_random(src1.data(), N, 42);
    init_vector_random(src2.data(), N, 123);
  }

  void TearDown(const ::benchmark::State &) override {
    src1.clear();
    src2.clear();
  }

protected:
  size_t N;
  std::vector<T> src1, src2;
};

using DotFixtureFloat = DotFixture<float>;

#define DEFINE_DOT_BENCHMARK(Name, Function, ...)                              \
  BENCHMARK_DEFINE_F(DotFixtureFloat, Name)(benchmark::State & state) {        \
    float result = 0.0f;                                                       \
                                                                               \
    for (auto _ : state) {                                                     \
      result = Function<float, ##__VA_ARGS__>(N, src1.data(), src2.data());    \
      benchmark::DoNotOptimize(result);                                        \
    }                                                                          \
                                                                               \
    if (state.thread_index() == 0) {                                           \
      float test_result =                                                      \
          Function<float, ##__VA_ARGS__>(N, src1.data(), src2.data());         \
      float ref_result = dot_ref<float>(N, src1.data(), src2.data());          \
                                                                               \
      float diff = std::abs(test_result - ref_result);                         \
      float max_val = std::max(std::abs(test_result), std::abs(ref_result));   \
      if (diff > 1e-3 * max_val && diff > 1e-3) {                              \
        state.SkipWithError("Verification failed!");                           \
      }                                                                        \
    }                                                                          \
                                                                               \
    double bytes = 2.0 * N * sizeof(float); /* read src1, read src2 */         \
    state.counters["GB/s"] = benchmark::Counter(                               \
        bytes, benchmark::Counter::kIsIterationInvariantRate,                  \
        benchmark::Counter::kIs1000);                                          \
    state.counters["N"] = N;                                                   \
    state.counters["Result"] = result;                                         \
  }                                                                            \
  BENCHMARK_REGISTER_F(DotFixtureFloat, Name)                                  \
      ->Args({256 * 256})                                                      \
      ->Args({1024 * 1024})                                                    \
      ->Args({4096 * 4096})                                                    \
      ->Unit(benchmark::kMillisecond);

// Naive Benchmark
BENCHMARK_DEFINE_F(DotFixtureFloat, Naive)(benchmark::State &state) {
  float result = 0.0f;

  for (auto _ : state) {
    result = dot_ref<float>(N, src1.data(), src2.data());
    benchmark::DoNotOptimize(result);
  }

  double bytes = 2.0 * N * sizeof(float);
  state.counters["GB/s"] =
      benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["Result"] = result;
}

BENCHMARK_REGISTER_F(DotFixtureFloat, Naive)
    ->Args({256 * 256})
    ->Args({1024 * 1024})
    ->Args({4096 * 4096})
    ->Unit(benchmark::kMillisecond);

// Sequential
DEFINE_DOT_BENCHMARK(Seq, hpc::l1::details::dot_seq, 1024)
DEFINE_DOT_BENCHMARK(SeqAPI, hpc::l1::dot, hpc::Backend::SEQUENTIAL, 1024)

// OpenMP
#ifdef ENABLE_OPENMP
DEFINE_DOT_BENCHMARK(OpenMP, hpc::l1::details::dot_omp, 1024)
DEFINE_DOT_BENCHMARK(OpenMPAPI, hpc::l1::dot, hpc::Backend::OPENMP, 1024)
#endif

// SIMD
#ifdef ENABLE_SIMD
DEFINE_DOT_BENCHMARK(SIMD_4, hpc::l1::details::dot_simd, 1024, 4)
DEFINE_DOT_BENCHMARK(SIMDAPI_4, hpc::l1::dot, hpc::Backend::SIMD, 1024, 4)
#endif

// OpenMP + SIMD
#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
DEFINE_DOT_BENCHMARK(OpenMP_SIMD_4, hpc::l1::details::dot_omp_simd, 1024, 4)
DEFINE_DOT_BENCHMARK(OpenMP_SIMDAPI_4, hpc::l1::dot, hpc::Backend::OPENMP_SIMD,
                     1024, 4)
#endif

// Apple Accelerate
#ifdef ENABLE_ACCELERATE
DEFINE_DOT_BENCHMARK(Accelerate, hpc::l1::details::dot_acceler)
DEFINE_DOT_BENCHMARK(AccelerateAPI, hpc::l1::dot, hpc::Backend::ACCELERATE)
#endif

BENCHMARK_MAIN();
