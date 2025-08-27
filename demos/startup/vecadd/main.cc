#define ENABLE_OPENMP
#include <hpc.hh>

using namespace hpc;

const int DSIZE = 10000000;

int main() {
  float *a_data = (float *)malloc(DSIZE * sizeof(float));
  openmp::Vector<float> a(DSIZE);

  ProgTimer timer_serial(Backend::SERIAL, "Serial");
  timer_serial.start();
  for (size_t i = 0; i < DSIZE; ++i) {
    a_data[i] += 1.0f;
  }
  timer_serial.stop();
  timer_serial.report();

  ProgTimer timer_openmp(Backend::OPENMP, "OpenMP");
  timer_openmp.start();
  a += 1.0f;
  timer_openmp.stop();
  timer_openmp.report();

  return 0;
}
