#pragma once

#include "./backends/backends.hh"
#include <chrono>
#include <iostream>
#include <string>

#ifdef HPC_ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef HPC_ENABLE_MPI
#include <mpi.h>
#endif

#ifdef HPC_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// Legacy macros (manual start/stop)
#define __TICK(backend)                                                        \
  static hpc::Timer __timer_##backend(hpc::Backend::backend, #backend);        \
  __timer_##backend.start();

#define __TOCK(backend)                                                        \
  __timer_##backend.stop();                                                    \
  __timer_##backend.report();

// Scoped timing macros (RAII - automatic start/stop)
#define __TIME_SCOPE(backend, name)                                            \
  hpc::ScopedTimer __scoped_timer_##__LINE__(hpc::Backend::backend, name)

#define __TIME_SEQUENTIAL(name) __TIME_SCOPE(SEQUENTIAL, name)

#ifdef HPC_ENABLE_OPENMP
#define __TIME_OPENMP(name) __TIME_SCOPE(OPENMP, name)
#else
#define __TIME_OPENMP(name) __TIME_SCOPE(SEQUENTIAL, name)
#endif

#ifdef HPC_ENABLE_MPI
#define __TIME_MPI(name) __TIME_SCOPE(MPI, name)
#else
#define __TIME_MPI(name) __TIME_SCOPE(SEQUENTIAL, name)
#endif

#ifdef HPC_ENABLE_SIMD
#define __TIME_SIMD(name) __TIME_SCOPE(SIMD, name)
#else
#define __TIME_SIMD(name) __TIME_SCOPE(SEQUENTIAL, name)
#endif

#ifdef HPC_ENABLE_CUDA
#define __TIME_CUDA(name) __TIME_SCOPE(CUDA, name)
#else
#define __TIME_CUDA(name) __TIME_SCOPE(SEQUENTIAL, name)
#endif

namespace hpc {

// ============================================================================
// Main Timer Class
// ============================================================================

class Timer {
public:
  explicit Timer(Backend mode = Backend::SEQUENTIAL,
                 const std::string &name = "Timer")
      : backend_(mode), label_(name), elapsed_(0.0) {
#ifdef HPC_ENABLE_MPI
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
#ifdef HPC_ENABLE_CUDA
    if (backend_ == Backend::CUDA && start_event_ != nullptr) {
      cudaEventDestroy(start_event_);
      cudaEventDestroy(stop_event_);
    }
#endif
  }

  // Disable copy (because of CUDA events)
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;

  // Allow move
  Timer(Timer &&other) noexcept
      : backend_(other.backend_), label_(std::move(other.label_)),
        elapsed_(other.elapsed_), start_time_(other.start_time_)
#ifdef HPC_ENABLE_OPENMP
        ,
        start_omp_(other.start_omp_)
#endif
#ifdef HPC_ENABLE_MPI
        ,
        start_mpi_(other.start_mpi_)
#endif
#ifdef HPC_ENABLE_CUDA
        ,
        start_event_(other.start_event_), stop_event_(other.stop_event_)
#endif
  {
#ifdef HPC_ENABLE_CUDA
    other.start_event_ = nullptr;
    other.stop_event_ = nullptr;
#endif
  }

  // Start timing
  void start() {
    switch (backend_) {
    case Backend::SEQUENTIAL: {
      start_time_ = std::chrono::high_resolution_clock::now();
      break;
    }

#ifdef HPC_ENABLE_OPENMP
    case Backend::OPENMP: {
      start_omp_ = omp_get_wtime();
      break;
    }
#endif

#ifdef HPC_ENABLE_MPI
    case Backend::MPI: {
      MPI_Barrier(MPI_COMM_WORLD);
      start_mpi_ = MPI_Wtime();
      break;
    }
#endif

#ifdef HPC_ENABLE_CUDA
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

#ifdef HPC_ENABLE_OPENMP
    case Backend::OPENMP: {
      elapsed_ = omp_get_wtime() - start_omp_;
      break;
    }
#endif

#ifdef HPC_ENABLE_MPI
    case Backend::MPI: {
      elapsed_ = MPI_Wtime() - start_mpi_;
      MPI_Barrier(MPI_COMM_WORLD);
      break;
    }
#endif

#ifdef HPC_ENABLE_CUDA
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

#ifdef HPC_ENABLE_OPENMP
  double start_omp_ = 0.0;
#endif

#ifdef HPC_ENABLE_MPI
  double start_mpi_ = 0.0;
#endif

#ifdef HPC_ENABLE_CUDA
  cudaEvent_t start_event_ = nullptr;
  cudaEvent_t stop_event_ = nullptr;
#endif
};

// Scoped Timer (RAII - automatic timing)
class ScopedTimer {
public:
  explicit ScopedTimer(Backend mode = Backend::SEQUENTIAL,
                       const std::string &name = "ScopedTimer",
                       bool auto_report = true)
      : timer_(mode, name), auto_report_(auto_report) {
    timer_.start();
  }

  ~ScopedTimer() {
    timer_.stop();
    if (auto_report_) {
      timer_.report();
    }
  }

  // Disable copy and move (scoped resource)
  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;
  ScopedTimer(ScopedTimer &&) = delete;
  ScopedTimer &operator=(ScopedTimer &&) = delete;

  // Get the underlying timer
  const Timer &timer() const { return timer_; }

  // Manual report (if auto_report is false)
  void report(std::ostream &os = std::cout) const { timer_.report(os); }

  // Get elapsed time so far (doesn't stop the timer)
  double elapsed() const {
    // We need to compute current elapsed without stopping
    // This is an approximation for sequential mode
    if (timer_.backend() == Backend::SEQUENTIAL) {
      auto now = std::chrono::high_resolution_clock::now();
      return std::chrono::duration<double>(now - start_snapshot_).count();
    }
    return timer_.elapsed();
  }

private:
  Timer timer_;
  bool auto_report_;
  std::chrono::high_resolution_clock::time_point start_snapshot_ =
      std::chrono::high_resolution_clock::now();
};
} // namespace hpc
