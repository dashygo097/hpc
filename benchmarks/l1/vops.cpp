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
template <typename T> void vadd_ref(T *dst, const T *src, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    dst[i] += src[i];
  }
}

// Fixture
template <typename T> class VaddFixture : public benchmark::Fixture {
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

using VaddFixtureFloat = VaddFixture<float>;

#define DEFINE_VADD_VEC_BENCHMARK(Name, Function, ...)                         \
  BENCHMARK_DEFINE_F(VaddFixtureFloat, Name)(benchmark::State & state) {       \
    std::vector<float> dst_copy = dst;                                         \
                                                                               \
    for (auto _ : state) {                                                     \
      Function<float, ##__VA_ARGS__>(dst_copy.data(), src.data(), N);          \
      benchmark::DoNotOptimize(dst_copy.data());                               \
      benchmark::ClobberMemory();                                              \
    }                                                                          \
                                                                               \
    if (state.thread_index() == 0) {                                           \
      std::vector<float> test_dst = dst;                                       \
      std::vector<float> test_ref = dst;                                       \
                                                                               \
      Function<float, ##__VA_ARGS__>(test_dst.data(), src.data(), N);          \
      vadd_ref(test_ref.data(), src.data(), N);                                \
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
  BENCHMARK_REGISTER_F(VaddFixtureFloat, Name)                                 \
      ->Args({256 * 256})                                                      \
      ->Args({1024 * 1024})                                                    \
      ->Args({4096 * 4096})                                                    \
      ->Unit(benchmark::kMillisecond);

// Naive Benchmark
BENCHMARK_DEFINE_F(VaddFixtureFloat, Naive)(benchmark::State &state) {
  std::vector<float> dst_copy = dst;

  for (auto _ : state) {
    vadd_ref(dst_copy.data(), src.data(), N);
    benchmark::DoNotOptimize(dst_copy.data());
    benchmark::ClobberMemory();
  }

  double bytes = 3.0 * N * sizeof(float);
  state.counters["GB/s"] =
      benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

BENCHMARK_REGISTER_F(VaddFixtureFloat, Naive)
    ->Args({256 * 256})
    ->Args({1024 * 1024})
    ->Args({4096 * 4096})
    ->Unit(benchmark::kMillisecond);

// Sequential
DEFINE_VADD_VEC_BENCHMARK(Seq, hpc::l1::details::vadd_seq)
DEFINE_VADD_VEC_BENCHMARK(SeqAPI, hpc::l1::vadd, hpc::Backend::SEQUENTIAL)
  
// OpenMP
#ifdef ENABLE_OPENMP
DEFINE_VADD_VEC_BENCHMARK(OpenMP, hpc::l1::details::vadd_omp, 1024)
DEFINE_VADD_VEC_BENCHMARK(OpenMPAPI, hpc::l1::vadd, hpc::Backend::OPENMP)
#endif
  
// SIMD
#ifdef ENABLE_SIMD
DEFINE_VADD_VEC_BENCHMARK(SIMD_4, hpc::l1::details::vadd_simd, 4)
DEFINE_VADD_VEC_BENCHMARK(SIMDAPI_4, hpc::l1::vadd, hpc::Backend::SIMD, 4)
#endif

// OpenMP + SIMD
#if defined(ENABLE_OPENMP) && defined(ENABLE_SIMD)
DEFINE_VADD_VEC_BENCHMARK(OpenMP_SIMD_4, hpc::l1::details::vadd_omp_simd, 1024,
                          4)
DEFINE_VADD_VEC_BENCHMARK(OpenMP_SIMDAPI_4, hpc::l1::vadd,
                          hpc::Backend::OPENMP_SIMD, 1024, 4)
#endif

// Apple Accelerate
#ifdef ENABLE_ACCELERATE
DEFINE_VADD_VEC_BENCHMARK(Accelerate, hpc::l1::details::vadd_acceler)
DEFINE_VADD_VEC_BENCHMARK(AccelerateAPI, hpc::l1::vadd,
                          hpc::Backend::ACCELERATE)
#endif
  
BENCHMARK_MAIN();
