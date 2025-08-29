#define ENABLE_OPENMP
#include <gtest/gtest.h>
#include <hpc.hh>
#include <hpc/mmul_impl.hh>

using namespace hpc;
const size_t DSIZE = 512;

class OpenMPMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    a_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    b_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    c_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));

    for (size_t i = 0; i < DSIZE * DSIZE; ++i) {
      a_serial[i] = 1.0f;
      b_serial[i] = 1.0f;
      c_serial[i] = 0.0f;
    }

    a = openmp::Matrix<float>(DSIZE, DSIZE, 1.0f);
    b = openmp::Matrix<float>(DSIZE, DSIZE, 1.0f);
    c = openmp::Matrix<float>(DSIZE, DSIZE, 0.0f);
  }

  void TearDown() override {
    free(a_serial);
    free(b_serial);
    free(c_serial);
  }

  void checkCorrectness(float *c_src, float *c_tgt) {
    for (size_t i = 0; i < DSIZE * DSIZE; ++i) {
      EXPECT_FLOAT_EQ(c_src[i], c_tgt[i]) << "Mismatch at index " << i;
    }
  }

  void reset() {
    memset(c_serial, 0, DSIZE * DSIZE * sizeof(float));
    c.fill(0.0f);
  }

  void computeSerialNaive() {
    serial::naive_mmul<float>(c_serial, a_serial, b_serial, DSIZE, DSIZE,
                              DSIZE);
  }

  void computeOpenMPNaive() { c = openmp::naive_mmul<float>(a, b); }

  float *a_serial;
  float *b_serial;
  float *c_serial;

  openmp::Matrix<float> a;
  openmp::Matrix<float> b;
  openmp::Matrix<float> c;
};

TEST_F(OpenMPMatrixTest, ParallelNaiveCorrectness) {
  computeSerialNaive();
  computeOpenMPNaive();
  checkCorrectness(c_serial, c.data());
  reset();
}

TEST_F(OpenMPMatrixTest, PerformanceBenchmark) {
  // Warm-up
  computeSerialNaive();
  computeOpenMPNaive();

  reset();

  ProgTimer timer_serial(Backend::SERIAL, "Serial Naive");
  ProgTimer timer_openmp(Backend::OPENMP, "OpenMP Naive");

  // Benchmark
  timer_serial.start();
  computeSerialNaive();
  timer_serial.stop();
  timer_serial.report();

  timer_openmp.start();
  computeOpenMPNaive();
  timer_openmp.stop();
  timer_openmp.report();
  checkCorrectness(c_serial, c.data());

  std::cout << "[INFO] OpenMP Naive achieves speedup of "
            << timer_serial.elapsed_seconds() / timer_openmp.elapsed_seconds()
            << "x over Serial;" << std::endl;
  ;
}
