#define ENABLE_OPENMP
#include <hpc.hh>

using namespace hpc;
const size_t DSIZE = 10000000;

int main() {

  float *a_serial = (float *)malloc(DSIZE * sizeof(float));
  float *b_serial = (float *)malloc(DSIZE * sizeof(float));
  float *c_serial = (float *)malloc(DSIZE * sizeof(float));

  openmp::Vector<float> a(DSIZE);
  openmp::Vector<float> b(DSIZE);
  openmp::Vector<float> c(DSIZE);

  float *a_data = a.data();
  float *b_data = b.data();
  float *c_data = c.data();

  for (size_t i = 0; i < DSIZE; ++i) {
    a_serial[i] = 1.0f;
    b_serial[i] = 2.0f;
    c_serial[i] = 0.0f;
  }
  for (size_t i = 0; i < DSIZE; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
    c[i] = 0.0f;
  }
  ProgTimer timer_serial(Backend::SERIAL, "Serial");
  float tmp;
  for (size_t i = 0; i < DSIZE; ++i) {
    tmp = a_serial[i] + b_serial[i];
  }
  timer_serial.stop();
  timer_serial.report();

  ProgTimer timer_openmp(Backend::OPENMP, "OpenMP");
#pragma omp parallel for
  for (size_t i = 0; i < DSIZE; ++i) {
    c_data[i] = a_data[i] + b_data[i];
  }

  timer_openmp.stop();
  timer_openmp.report();

  timer_openmp.start();
  c.assign(a, b, std::plus<float>());
  timer_openmp.stop();
  timer_openmp.report();

  return 0;
}
