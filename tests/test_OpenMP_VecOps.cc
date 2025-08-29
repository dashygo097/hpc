#define ENABLE_OPENMP
#include <gtest/gtest.h>
#include <hpc.hh>
#include <simd/simd.h>

using namespace hpc;
const size_t DSIZE = 3200000;
const size_t TSIZE = 5;

class OpenMPVectorTest : public ::testing::Test {
protected:
  void SetUp() override {
    a_serial = (float *)malloc(DSIZE * sizeof(float));
    b_serial = (float *)malloc(DSIZE * sizeof(float));
    c_serial = (float *)malloc(DSIZE * sizeof(float));

    a_data = (float *)malloc(DSIZE * sizeof(float));
    b_data = (float *)malloc(DSIZE * sizeof(float));
    c_data = (float *)malloc(DSIZE * sizeof(float));

    a = openmp::Vector<float>(DSIZE, 1.0f);
    b = openmp::Vector<float>(DSIZE, 2.0f);
    c = openmp::Vector<float>(DSIZE, 0.0f);

    for (size_t i = 0; i < DSIZE; ++i) {
      a_serial[i] = 1.0f;
      b_serial[i] = 2.0f;
      c_serial[i] = 0.0f;
      a_data[i] = 1.0f;
      b_data[i] = 2.0f;
      c_data[i] = 0.0f;
    }
  }

  void TearDown() override {
    free(a_serial);
    free(b_serial);
    free(c_serial);
    free(a_data);
    free(b_data);
    free(c_data);
  }

  void checkCorrectness(float *a, float *b) {
    for (size_t i = 0; i < DSIZE; ++i) {
      EXPECT_FLOAT_EQ(a[i], b[i]) << "Mismatch at index " << i;
    }
  }

  void computeSerial() {
    for (size_t i = 0; i < DSIZE; ++i) {
      float temp = a_serial[i] + b_serial[i] - 1.0f;
      c_serial[i] = temp * temp;
    }
  }

  void computeBaseOpenMP() {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < DSIZE; ++i) {
      float temp = a_data[i] + b_data[i] - 1.0f;
      c_data[i] = temp * temp;
    }
  }

  void computeSIMDOpenMP() {
    size_t simd_count = DSIZE - (DSIZE % 4);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_count; i += 4) {
      simd_float4 va = *((simd_float4 *)(a_data + i));
      simd_float4 vb = *((simd_float4 *)(b_data + i));
      simd_float4 vr = va + vb - simd_float4(1.0f);
      vr = vr * vr;
      *((simd_float4 *)(c_data + i)) = vr;
    }

    if (simd_count < DSIZE) {
      for (size_t i = simd_count; i < DSIZE; ++i) {
        c_data[i] = a_data[i] + b_data[i] - 1.0f;
        c_data[i] = c_data[i] * c_data[i];
      }
    }
  }

  void computeImplOpenMP() {
    c.assign(a, b, [](const float &x, const float &y) {
      float temp = (x + y - 1.0f);
      return temp * temp;
    });
  }

  void cleanUp() {
    memset(c_serial, 0, DSIZE * sizeof(float));
    memset(c_data, 0, DSIZE * sizeof(float));
    c.fill(0.0f);
  }

  float *a_serial;
  float *b_serial;
  float *c_serial;

  float *a_data;
  float *b_data;
  float *c_data;

  openmp::Vector<float> a;
  openmp::Vector<float> b;
  openmp::Vector<float> c;
};

TEST_F(OpenMPVectorTest, BaseParallelComputationCorrectness) {
  computeSerial();
  computeBaseOpenMP();
  checkCorrectness(c_serial, c_data);
  cleanUp();
}

TEST_F(OpenMPVectorTest, SIMDParallelComputationCorrectness) {
  computeSerial();
  computeSIMDOpenMP();
  checkCorrectness(c_serial, c_data);
  cleanUp();
}

TEST_F(OpenMPVectorTest, ParallelizedVectorAssignComputationCorrectness) {
  computeSerial();
  computeImplOpenMP();
  checkCorrectness(c_serial, const_cast<float *>(c.data()));
  cleanUp();
}

TEST_F(OpenMPVectorTest, PerformanceBenchmark) {
  // Warm-up
  computeSerial();
  computeBaseOpenMP();
  computeSIMDOpenMP();
  computeImplOpenMP();

  cleanUp();

  ProgTimer timer_serial(Backend::SERIAL, "Serial");
  ProgTimer timer_base_openmp(Backend::OPENMP, "Base OpenMP");
  ProgTimer timer_simd_openmp(Backend::OPENMP, "SIMD OpenMP");
  ProgTimer timer_impl_openmp(Backend::OPENMP, "Impled OpenMP");

  // Benchmark
  timer_serial.start();
  for (size_t i = 0; i < TSIZE; ++i)
    computeSerial();
  timer_serial.stop();
  timer_serial.report();

  timer_base_openmp.start();
  for (size_t i = 0; i < TSIZE; ++i)
    computeBaseOpenMP();
  timer_base_openmp.stop();
  timer_base_openmp.report();
  checkCorrectness(c_serial, c_data);

  memset(c_data, 0, DSIZE * sizeof(float));
  timer_simd_openmp.start();
  for (size_t i = 0; i < TSIZE; ++i)
    computeSIMDOpenMP();
  timer_simd_openmp.stop();
  timer_simd_openmp.report();
  checkCorrectness(c_serial, c_data);

  timer_impl_openmp.start();
  for (size_t i = 0; i < TSIZE; ++i)
    computeImplOpenMP();
  timer_impl_openmp.stop();
  timer_impl_openmp.report();
  checkCorrectness(c_serial, const_cast<float *>(c.data()));

  std::cout << "[INFO] Base OpenMP achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_base_openmp.elapsed_seconds()
            << "x over Serial;" << std::endl;
  ;

  std::cout << "[INFO] SIMD OpenMP achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_simd_openmp.elapsed_seconds()
            << "x over Serial;" << std::endl;

  std::cout << "[INFO] Impled OpenMP achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_impl_openmp.elapsed_seconds()
            << "x over Serial;" << std::endl;
}
