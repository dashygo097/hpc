#include <hpc.hh>

int main() {
  __TICK(SEQUENTIAL);
  float a[1000];
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 1000; j++) {
      a[j] = static_cast<float>(i * j);
    }
  }
  __TOCK(SEQUENTIAL);

  return 0;
}
