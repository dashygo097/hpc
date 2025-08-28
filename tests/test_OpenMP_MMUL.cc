#define ENABLE_OPENMP
#include <gtest/gtest.h>
#include <hpc.hh>

using namespace hpc;
const size_t DSIZE = 2048;
const size_t TSIZE = 5;

class OpenMPMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    a_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    b_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    c_serial = (float *)malloc(DSIZE * DSIZE * sizeof(float));

    a_data = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    b_data = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    c_data = (float *)malloc(DSIZE * DSIZE * sizeof(float));

    a = openmp::Matrix<float>(DSIZE, DSIZE, 1.0f);
    b = openmp::Matrix<float>(DSIZE, DSIZE, 2.0f);
    c = openmp::Matrix<float>(DSIZE, DSIZE, 0.0f);

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
    for (size_t time = 0; time < TSIZE; ++time) {
      for (size_t i = 0; i < DSIZE; ++i) {
        float temp = a_serial[i] + b_serial[i] - 1.0f;
        c_serial[i] = temp * temp;
      }
    }
  }

  void computeBaseOpenMP() {
    for (size_t time = 0; time < TSIZE; ++time) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < DSIZE; ++i) {
        float temp = a_data[i] + b_data[i] - 1.0f;
        c_data[i] = temp * temp;
      }
    }
  }

  void computeImplOpenMP() {
    for (size_t time = 0; time < TSIZE; ++time) {
      c.assign(a, b, [](const float &x, const float &y) {
        float temp = (x + y - 1.0f);
        return temp * temp;
      });
    }
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

  openmp::Matrix<float> a;
  openmp::Matrix<float> b;
  openmp::Matrix<float> c;
};
