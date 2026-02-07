#pragma once

#ifdef HPC_ENABLE_MPI
#include <iostream>
#include <mpi.h>
#endif

#ifdef HPC_ENABLE_MPI
namespace hpc::mpi {
class MPIEnv {
public:
  MPIEnv(int argc, char **argv) : _thread_support_level(MPI_THREAD_SINGLE) {
    init(argc, argv, MPI_THREAD_SERIALIZED);
  }
  MPIEnv(int argc, char **argv, int required_thread_support_level)
      : _thread_support_level(MPI_THREAD_SINGLE) {
    init(argc, argv, required_thread_support_level);
  }

  ~MPIEnv() {
    if (!_finalized && _initialized) {
      MPI_Finalize();
      _finalized = true;
    }
  }

  MPIEnv(const MPIEnv &) = delete;
  MPIEnv &operator=(const MPIEnv &) = delete;

  void init(int argc, char **argv, int required_thread_support_level) {
    if (_initialized) {
      return;
    }

    int provided;
    if (required_thread_support_level == MPI_THREAD_SINGLE) {
      MPI_Init(&argc, &argv);
      _thread_support_level = MPI_THREAD_SINGLE;
    } else {
      MPI_Init_thread(&argc, &argv, required_thread_support_level, &provided);
      _thread_support_level = provided;

      if (provided < required_thread_support_level) {
        std::cerr << "Warning: MPI thread support level " << provided
                  << " is lower than requested "
                  << required_thread_support_level << std::endl;
      }
    }
    _initialized = true;
    _finalized = false;
  }

  [[nodiscard]] static bool initialized() noexcept { return _initialized; }
  [[nodiscard]] static bool finalized() noexcept { return _finalized; }
  [[nodiscard]] int thread_support_level() const noexcept {
    return _thread_support_level;
  }

private:
  static bool _initialized;
  static bool _finalized;
  int _thread_support_level;
};

bool MPIEnv::_initialized = false;
bool MPIEnv::_finalized = false;

} // namespace hpc::mpi
#endif
