#pragma once

#include <chrono>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "hpc/constants.hpp"

namespace hpc {
class ProgTimer {
public:
  explicit ProgTimer(Backend mode = Backend::SERIAL,
                     const std::string &name = "ProgramTimer")
      : backend(mode), label(name) {
    if (backend == Backend::SERIAL) {
#ifdef USE_MPI
      int flag = 0;
      MPI_Initialized(&flag);
      if (flag) {
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif
    }
    start();
  }

  void start() {
    switch (backend) {
    case Backend::SERIAL:
      start_std = std::chrono::high_resolution_clock::now();
      break;
#ifdef _OPENMP
    case Backend::OPENMP:
      start_omp = omp_get_wtime();
      break;
#endif
#ifdef USE_MPI
    case Backend::OPENMPI:
      start_mpi = MPI_Wtime();
      break;
#endif
#ifdef USE_CUDA
    case Backend::CUDA: {
      cudaEventCreate(&start_event);
      cudaEventCreate(&stop_event);
      cudaEventRecord(start_event);
      break;
#else
    default:
      std::cerr << "Unsupported backend: " << static_cast<int>(backend)
                << std::endl;
      throw std::runtime_error("Unsupported backend");
#endif
    }
    }

    void stop() {
      switch (backend) {
      case Backend::SERIAL:
        elapsed = std::chrono::duration<double>(
                      std::chrono::high_resolution_clock::now() - start_std)
                      .count();
        break;
#ifdef _OPENMP
      case Backend::OPENMP:
        elapsed = omp_get_wtime() - start_omp;
        break;
#endif
#ifdef USE_MPI
      case Backend::OPENMPI:
        elapsed = MPI_Wtime() - start_mpi;
        break;
#endif
#ifdef USE_CUDA
      case Backend::CUDA: {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        elapsed = ms / 1000.0;
        break;
#else
    default:
      std::cerr << "Unsupported backend: " << static_cast<int>(backend)
                << std::endl;
      throw std::runtime_error("Unsupported backend");
#endif
      }
      }

      double elapsed_seconds() const { return elapsed; }

      void report() const {
        std::cout << "[TIMER][" << label << "] " << elapsed << " sec\n";
      }

      ~ProgTimer() {
#ifdef USE_CUDA
        if (backend == Backend::CUDA) {
          cudaEventDestroy(start_event);
          cudaEventDestroy(stop_event);
        }
#endif
      }

    private:
      Backend backend;
      std::string label;
      double elapsed = 0.0;

      // Backend-specific start times
      std::chrono::high_resolution_clock::time_point start_std;

#ifdef _OPENMP
      double start_omp = 0.0;
#endif
#ifdef USE_MPI
      double start_mpi = 0.0;
#endif
#ifdef USE_CUDA
      cudaEvent_t start_event = nullptr, stop_event = nullptr;
#endif
    };

  } // namespace hpc
