#define ENABLE_OPENMP
#include <hpc.hh>

using namespace hpc;
const size_t DSIZE = 10000000;
const size_t TSIZE = 50;

int main() {
  ProgTimer timer_serial(Backend::SERIAL, "Serial");
  ProgTimer timer_openmp(Backend::OPENMP, "OpenMP");

  float *a_serial = (float *)malloc(DSIZE * sizeof(float));
  float *b_serial = (float *)malloc(DSIZE * sizeof(float));
  float *c_serial = (float *)malloc(DSIZE * sizeof(float));

  float *a_data = (float *)malloc(DSIZE * sizeof(float));
  float *b_data = (float *)malloc(DSIZE * sizeof(float));
  float *c_data = (float *)malloc(DSIZE * sizeof(float));

  openmp::Vector<float> a(DSIZE, 1.0f);
  openmp::Vector<float> b(DSIZE, 2.0f);
  openmp::Vector<float> c(DSIZE, 0.0f);

  for (size_t i = 0; i < DSIZE; ++i) {
    a_serial[i] = 1.0f;
    b_serial[i] = 2.0f;
    c_serial[i] = 0.0f;
    a_data[i] = 1.0f;
    b_data[i] = 2.0f;
    c_data[i] = 0.0f;
  }

  timer_serial.start();
  for (size_t time = 0; time < TSIZE; ++time) {
    for (size_t i = 0; i < DSIZE; ++i) {
      c_serial[i] = a_serial[i] + b_serial[i] - 1.0f;
      c_serial[i] = c_serial[i] * c_serial[i];
    }
  }

  timer_serial.stop();
  timer_serial.report();

  timer_openmp.start();
  for (size_t time = 0; time < TSIZE; ++time) {
#pragma omp parallel for
    for (size_t i = 0; i < DSIZE; ++i) {
      c_data[i] = a_data[i] + b_data[i] - 1.0f;
      c_data[i] = c_data[i] * c_data[i];
    }
  }

  timer_openmp.stop();
  timer_openmp.report();

  for (size_t i = 0; i < DSIZE; ++i) {
    if (c_serial[i] != c_data[i]) {
      std::cerr << "Error at index " << i << ": " << c_serial[i]
                << " != " << c_data[i] << std::endl;
      return 1;
    }
  }

  timer_openmp.start();
  for (size_t time = 0; time < TSIZE; ++time) {
    c.assign(a, b, [](const float &x, const float &y) {
      float temp = (x + y - 1.0f);
      return temp * temp;
    });
  }

  timer_openmp.stop();
  timer_openmp.report();

  for (size_t i = 0; i < DSIZE; ++i) {
    if (c_serial[i] != c.data()[i]) {
      std::cerr << "Error at index " << i << ": " << c_serial[i]
                << " != " << c.data()[i] << std::endl;
      return 1;
    }
  }

  free(a_serial);
  free(b_serial);
  free(c_serial);
  free(a_data);
  free(b_data);
  free(c_data);

  return 0;
}
