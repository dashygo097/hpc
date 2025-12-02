#pragma once

#include "./backends/backends.hh"
#include <chrono>
#include <iostream>
#include <string>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#define __TICK(backend)                                                        \
  static hpc::Timer __timer_##backend(hpc::Backend::backend, #backend);        \
  __timer_##backend.start();
#define __TOCK(backend)                                                        \
  __timer_##backend.stop();                                                    \
  __timer_##backend.report();

namespace hpc {

// Main Timer Class
class Timer {
public:
  explicit Timer(Backend mode = Backend::SEQUENTIAL,
                 const std::string &name = "Timer")
      : backend_(mode), label_(name), elapsed_(0.0) {
#ifdef ENABLE_MPI
    if (backend_ == Backend::MPI) {
      int initialized = 0;
      MPI_Initialized(&initialized);
      if (initialized) {
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
#endif
  }

  ~Timer() {
#ifdef ENABLE_CUDA
    if (backend_ == Backend::CUDA && start_event_ != nullptr) {
      cudaEventDestroy(start_event_);
      cudaEventDestroy(stop_event_);
    }
#endif
  }

  // Start timing
  void start() {
    switch (backend_) {
    case Backend::SEQUENTIAL: {
      start_time_ = std::chrono::high_resolution_clock::now();
      break;
    }

#ifdef ENABLE_OPENMP
    case Backend::OPENMP: {
      start_omp_ = omp_get_wtime();
      break;
    }
#endif

#ifdef ENABLE_MPI
    case Backend::MPI: {
      MPI_Barrier(MPI_COMM_WORLD);
      start_mpi_ = MPI_Wtime();
      break;
    }
#endif

#ifdef ENABLE_CUDA
    case Backend::CUDA: {
      if (start_event_ == nullptr) {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
      }
      cudaEventRecord(start_event_);
      break;
    }
#endif

    default:
      start_time_ = std::chrono::high_resolution_clock::now();
      break;
    }
  }

  // Stop timing
  void stop() {
    switch (backend_) {
    case Backend::SEQUENTIAL: {
      auto end = std::chrono::high_resolution_clock::now();
      elapsed_ = std::chrono::duration<double>(end - start_time_).count();
      break;
    }

#ifdef ENABLE_OPENMP
    case Backend::OPENMP: {
      elapsed_ = omp_get_wtime() - start_omp_;
      break;
    }
#endif

#ifdef ENABLE_MPI
    case Backend::MPI: {
      elapsed_ = MPI_Wtime() - start_mpi_;
      MPI_Barrier(MPI_COMM_WORLD);
      break;
    }
#endif

#ifdef ENABLE_CUDA
    case Backend::CUDA: {
      cudaEventRecord(stop_event_);
      cudaEventSynchronize(stop_event_);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start_event_, stop_event_);
      elapsed_ = ms / 1000.0; // Convert to seconds
      break;
    }
#endif

    default: {
      auto end = std::chrono::high_resolution_clock::now();
      elapsed_ = std::chrono::duration<double>(end - start_time_).count();
      break;
    }
    }
  }

  // Get elapsed time in seconds
  double elapsed() const { return elapsed_; }

  // Get elapsed time in milliseconds
  double elapsed_ms() const { return elapsed_ * 1000.0; }

  // Get elapsed time in microseconds
  double elapsed_us() const { return elapsed_ * 1000000.0; }

  // Reset timer
  void reset() {
    elapsed_ = 0.0;
    start();
  }

  // Report timing
  void report(std::ostream &os = std::cout) const {
    os << "[" << label_ << "] " << elapsed_ << " sec (" << elapsed_ms()
       << " ms)\n";
  }

  // Report with custom message
  void report(const std::string &msg, std::ostream &os = std::cout) const {
    os << "[" << msg << "] " << elapsed_ << " sec\n";
  }

  // Get backend
  Backend backend() const { return backend_; }

  // Get label
  const std::string &label() const { return label_; }

private:
  Backend backend_;
  std::string label_;
  double elapsed_;

  // Backend-specific timing data
  std::chrono::high_resolution_clock::time_point start_time_;

#ifdef ENABLE_OPENMP
  double start_omp_ = 0.0;
#endif

#ifdef ENABLE_MPI
  double start_mpi_ = 0.0;
#endif

#ifdef ENABLE_CUDA
  cudaEvent_t start_event_ = nullptr;
  cudaEvent_t stop_event_ = nullptr;
#endif
};

} // namespace hpc
