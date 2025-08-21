#define USE_MPI
#include <hpc.hh>

#ifdef USE_MPI
#include <mpi.h>
#endif

int main() {
  int iterations = 100000000;
  {
    hpc::ProgTimer t(hpc::Backend::SERIAL, "Serial");
    for (volatile int i = 0; i < iterations; ++i)
      ;
    t.stop();
    t.report();
  }

#ifdef _OPENMP
  {
    hpc::ProgTimer t(hpc::Backend::OPENMP, "OpenMP");
#pragma omp parallel for
    for (int i = 0; i < iterations; ++i)
      ;
    t.stop();
    t.report();
  }
#endif

#ifdef USE_MPI
  MPI_Init(nullptr, nullptr);
  {
    hpc::ProgTimer t(hpc::Backend::OPENMPI, "MPI");
    MPI_Barrier(MPI_COMM_WORLD);
    t.stop();
    t.report();
  }
  MPI_Finalize();
#endif

#ifdef USE_CUDA
  {
    hpc::ProgTimer t(hpc::Backend::CUDA, "CUDA");
    // kernel<<<...>>>(); cudaDeviceSynchronize();
    t.stop();
    t.report();
  }
#endif

  return 0;
}
