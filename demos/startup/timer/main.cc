#include <hpc.hpp>
#include <mpi.h>

int main() {
  {
    hpc::ProgTimer t(HPC_TIMER_SERIAL, "Serial");
    for (volatile int i = 0; i < 1e8; ++i)
      ;
    t.stop();
    t.report();
  }

#ifdef _OPENMP
  {
    hpc::ProgTimer t(HPC_TIMER_OPENMP, "OpenMP");
#pragma omp parallel for
    for (int i = 0; i < 1e8; ++i)
      ;
    t.stop();
    t.report();
  }
#endif

#ifdef USE_MPI
  MPI_Init(nullptr, nullptr);
  {
    hpc::ProgTimer t(HPC_TIMER_OPENMPI, "MPI");
    MPI_Barrier(MPI_COMM_WORLD);
    t.stop();
    t.report();
  }
  MPI_Finalize();
#endif

#ifdef USE_CUDA
  {
    hpc::ProgTimer t(HPC_TIMER_CUDA, "CUDA");
    // kernel<<<...>>>(); cudaDeviceSynchronize();
    t.stop();
    t.report();
  }
#endif

  return 0;
}
