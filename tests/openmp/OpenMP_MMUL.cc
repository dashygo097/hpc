#define ENABLE_OPENMP
#define ENABLE_SIMD
#include <gtest/gtest.h>
#include <hpc.hh>

using namespace hpc;
const size_t DSIZE = 512;

class OpenMPMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {

    a_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    b_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    c_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));

#if defined(__APPLE__)
    a_acc = (__LAPACK_float_complex *)malloc(DSIZE * DSIZE *
                                             sizeof(__LAPACK_float_complex));
    b_acc = (__LAPACK_float_complex *)malloc(DSIZE * DSIZE *
                                             sizeof(__LAPACK_float_complex));
    c_acc = (__LAPACK_float_complex *)malloc(DSIZE * DSIZE *
                                             sizeof(__LAPACK_float_complex));
#endif

    for (size_t i = 0; i < DSIZE * DSIZE; ++i) {
      a_serial[i] = 1.0f;
      b_serial[i] = 2.0f;
      c_serial[i] = 0.0f;
#if defined(__APPLE__)
      a_acc[i] = 1.0f;
      b_acc[i] = 2.0f;
      c_acc[i] = 0.0f;
#endif
    }

    a = openmp::Matrix<float>(DSIZE, DSIZE, 1.0f);
    b = openmp::Matrix<float>(DSIZE, DSIZE, 2.0f);
    c = openmp::Matrix<float>(DSIZE, DSIZE, 0.0f);
  }

  void TearDown() override {
    free(a_serial);
    free(b_serial);
    free(c_serial);
#if defined(__APPLE__)
    free(a_acc);
    free(b_acc);
    free(c_acc);
#endif
  }

  void checkCorrectness(float *c_src, float *c_tgt) {
    for (size_t i = 0; i < DSIZE * DSIZE; ++i) {
      EXPECT_FLOAT_EQ(c_src[i], c_tgt[i]) << "Mismatch at index " << i;
    }
  }

#if defined(__APPLE__)
  void checkCorrectness(float *c_src, __LAPACK_float_complex *c_tgt) {
    for (size_t i = 0; i < DSIZE * DSIZE; ++i) {
      EXPECT_FLOAT_EQ(c_src[i], (float)(c_tgt[i].real()))
          << "Mismatch at index " << i;
    }
  }
#endif

  void reset() {
    memset(c_serial, 0, DSIZE * DSIZE * sizeof(float));
#if defined(__APPLE__)
    for (size_t i = 0; i < DSIZE * DSIZE; ++i) {
      c_acc[i] = 0.0f;
    }
#endif
    c.fill(0.0f);
  }

  void computeBaseline() {
    serial::mmul_baseline<float>(c_serial, a_serial, b_serial, DSIZE, DSIZE,
                                 DSIZE);
  }

#if defined(__APPLE__)
  void computeAccelerate() {
    __LAPACK_float_complex alpha = __LAPACK_float_complex(1.0f, 0.0f);
    __LAPACK_float_complex beta = __LAPACK_float_complex(0.0f, 0.0f);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, DSIZE, DSIZE, DSIZE,
                &alpha, (const __LAPACK_float_complex *)a_acc, DSIZE,
                (const __LAPACK_float_complex *)b_acc, DSIZE, &beta,
                (__LAPACK_float_complex *)c_acc, DSIZE);
  }
#endif

  void computeOpenMPNaive() { c = openmp::naive_mmul<float>(a, b); }
  void computeOpenMPTiled() { c = openmp::tiled_mmul<float>(a, b); }

#if defined(__APPLE) || defined(__ARM_NEON)
  void computeOpenMPNaiveSimd() {
    c = openmp::naive_mmul_simd_1xk<float>(a, b);
  }
#endif

  float *a_serial;
  float *b_serial;
  float *c_serial;

#if defined(__APPLE__)
  __LAPACK_float_complex *a_acc;
  __LAPACK_float_complex *b_acc;
  __LAPACK_float_complex *c_acc;
#endif

  openmp::Matrix<float> a;
  openmp::Matrix<float> b;
  openmp::Matrix<float> c;
};

TEST_F(OpenMPMatrixTest, ParallelNaiveCorrectness) {
  computeBaseline();
  computeOpenMPNaive();
  checkCorrectness(c_serial, c.data());
  reset();
}

TEST_F(OpenMPMatrixTest, ParallelTiledCorrectness) {
  computeBaseline();
  computeOpenMPTiled();
  checkCorrectness(c_serial, c.data());
  reset();
}

#if defined(__APPLE__) || defined(__ARM_NEON)
TEST_F(OpenMPMatrixTest, ParallelNaiveSimdCorrectness) {
  computeBaseline();
  computeOpenMPNaiveSimd();
  checkCorrectness(c_serial, c.data());
  reset();
}
#endif

TEST_F(OpenMPMatrixTest, PerformanceBenchmark) {
  // Warm-up
  computeBaseline();
  computeOpenMPNaive();
  computeOpenMPTiled();
#if defined(__APPLE__) || defined(__ARM_NEON)
  computeOpenMPNaiveSimd();
#endif
#if defined(__APPLE__)
  computeAccelerate();
#endif

  reset();

  ProgTimer timer_serial(Backend::SERIAL, "Baseline");
  ProgTimer timer_openmp(Backend::OPENMP, "OpenMP Naive");
  ProgTimer timer_openmp_tiled(Backend::OPENMP, "OpenMP Tiled");
  ProgTimer timer_openmp_simd(Backend::OPENMP, "OpenMP Naive + SIMD");
#if defined(__APPLE__)
  ProgTimer timer_accel(Backend::SERIAL, "Accelerate Lib");
#endif

  // Benchmark
  timer_serial.start();
  computeBaseline();
  timer_serial.stop();
  timer_serial.report();

  timer_openmp.start();
  computeOpenMPNaive();
  timer_openmp.stop();
  timer_openmp.report();
  checkCorrectness(c_serial, c.data());

  timer_openmp_tiled.start();
  computeOpenMPTiled();
  timer_openmp_tiled.stop();
  timer_openmp_tiled.report();
  checkCorrectness(c_serial, c.data());

#if defined(__APPLE__) || defined(__ARM_NEON)
  timer_openmp_simd.start();
  computeOpenMPNaiveSimd();
  timer_openmp_simd.stop();
  timer_openmp_simd.report();
  checkCorrectness(c_serial, c.data());
#endif

#if defined(__APPLE__)
  timer_accel.start();
  computeAccelerate();
  timer_accel.stop();
  timer_accel.report();
  checkCorrectness(c_serial, c_acc);
#endif

  std::cout << "[INFO] OpenMP Naive achieves speedup of "
            << timer_serial.elapsed_seconds() / timer_openmp.elapsed_seconds()
            << "x over baseline;" << std::endl;
  ;

  std::cout << "[INFO] OpenMP Tiled achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_openmp_tiled.elapsed_seconds()
            << "x over baseline;" << std::endl;

#if defined(__APPLE__) || defined(__ARM_NEON)
  std::cout << "[INFO] OpenMP Naive + SIMD achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_openmp_simd.elapsed_seconds()
            << "x over baseline;" << std::endl;
#endif

#if defined(__APPLE__)
  std::cout << "[INFO] Accelerate achieves speedup of "
            << timer_serial.elapsed_seconds() / timer_accel.elapsed_seconds()
            << "x over baseline;" << std::endl;
#endif
}
