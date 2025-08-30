#define ENABLE_OPENMP
#define ENABLE_SIMD
#include <gtest/gtest.h>
#include <hpc.hh>
#if defined(__APPLE__)
#include <simd/simd.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace hpc;
const size_t DSIZE = 10000000;

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

  void checkCorrectness(float *c_src, float *c_tgt) {
    for (size_t i = 0; i < DSIZE; ++i) {
      EXPECT_FLOAT_EQ(c_src[i], c_tgt[i]) << "Mismatch at index " << i;
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

#if defined(__APPLE__)
  void computeSIMDOpenMP() {
    size_t simd_count = DSIZE - (DSIZE % 4);
    using simd_type = simd_float4;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_count; i += 4) {
      simd_type va = *((simd_type *)(a_data + i));
      simd_type vb = *((simd_type *)(b_data + i));
      simd_type vr = va + vb - simd_type(1.0f);
      vr = vr * vr;
      *((simd_type *)(c_data + i)) = vr;
    }

    if (simd_count < DSIZE) {
      for (size_t i = simd_count; i < DSIZE; ++i) {
        c_data[i] = a_data[i] + b_data[i] - 1.0f;
        c_data[i] = c_data[i] * c_data[i];
      }
    }
  }
#elif defined(__ARM_NEON)
  void computeSIMDOpenMP() {
    size_t simd_count = DSIZE - (DSIZE % 4);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < simd_count; i += 4) {
      float32x4_t va = vld1q_f32(a_data + i);
      float32x4_t vb = vld1q_f32(b_data + i);
      float32x4_t vr = vsubq_f32(vaddq_f32(va, vb), vdupq_n_f32(1.0f));
      vr = vmulq_f32(vr, vr);
      vst1q_f32(c_data + i, vr);
    }
    if (simd_count < DSIZE) {
      for (size_t i = simd_count; i < DSIZE; ++i) {
        c_data[i] = a_data[i] + b_data[i] - 1.0f;
        c_data[i] = c_data[i] * c_data[i];
      }
    }
  }
#endif

#if defined(__APPLE__)
  void computeImplOpenMP() {
    using simd_type = simd_float4;
    c.assign(a, b, [](const simd_type &x, const simd_type &y) {
      auto temp = x + y - simd_type(1.0f);
      return temp * temp;
    });
  }
#elif defined(__ARM_NEON)
  void computeImplOpenMP() {
    c.assign(a, b, [](const float32x4_t &x, const float32x4_t &y) {
      auto temp = vsubq_f32(vaddq_f32(x, y), vdupq_n_f32(1.0f));
      return vmulq_f32(temp, temp);
    });
  }
#endif

  void reset() {
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
  reset();
}

#if defined(__APPLE__) || defined(__ARM_NEON)
TEST_F(OpenMPVectorTest, SIMDParallelComputationCorrectness) {
  computeSerial();
  computeSIMDOpenMP();
  checkCorrectness(c_serial, c_data);
  reset();
}

TEST_F(OpenMPVectorTest, ParallelizedVectorAssignComputationCorrectness) {
  computeSerial();
  computeImplOpenMP();
  checkCorrectness(c_serial, const_cast<float *>(c.data()));
  reset();
}
#endif

TEST_F(OpenMPVectorTest, PerformanceBenchmark) {
  // Warm-up
  computeSerial();
  computeBaseOpenMP();
#if defined(__APPLE__) || defined(__ARM_NEON)
  computeSIMDOpenMP();
  computeImplOpenMP();
#endif

  reset();

  ProgTimer timer_serial(Backend::SERIAL, "Serial");
  ProgTimer timer_base_openmp(Backend::OPENMP, "Base OpenMP");
#if defined(__APPLE__) || defined(__ARM_NEON)
  ProgTimer timer_simd_openmp(Backend::OPENMP, "SIMD OpenMP");
  ProgTimer timer_impl_openmp(Backend::OPENMP, "Impled OpenMP");
#endif

  // Benchmark
  timer_serial.start();
  computeSerial();
  timer_serial.stop();
  timer_serial.report();

  timer_base_openmp.start();
  computeBaseOpenMP();
  timer_base_openmp.stop();
  timer_base_openmp.report();
  checkCorrectness(c_serial, c_data);

#if defined(__APPLE__) || defined(__ARM_NEON)
  timer_simd_openmp.start();
  computeSIMDOpenMP();
  timer_simd_openmp.stop();
  timer_simd_openmp.report();
  checkCorrectness(c_serial, c_data);

  timer_impl_openmp.start();
  computeImplOpenMP();
  timer_impl_openmp.stop();
  timer_impl_openmp.report();
#endif

  std::cout << "[INFO] Base OpenMP achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_base_openmp.elapsed_seconds()
            << "x over Serial;" << std::endl;
  ;

#if defined(__APPLE__) || defined(__ARM_NEON)
  std::cout << "[INFO] SIMD OpenMP achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_simd_openmp.elapsed_seconds()
            << "x over Serial;" << std::endl;

  std::cout << "[INFO] Impled OpenMP achieves speedup of "
            << timer_serial.elapsed_seconds() /
                   timer_impl_openmp.elapsed_seconds()
            << "x over Serial;" << std::endl;
#endif
}
