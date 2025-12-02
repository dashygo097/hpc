#pragma once

#ifdef ENABLE_MPI
#include <iostream>
#include <mpi.h>
#endif

#ifdef ENABLE_MPI
namespace hpc::mpi {
class MPICommunicator {
public:
  MPICommunicator(MPI_Comm communicator = MPI_COMM_WORLD,
                  const std::string &comm_name = "DEFAULT_WORLD")
      : _comm(communicator), _name(comm_name), _comm_owned(false) {
    init();
  }

  MPICommunicator(const MPICommunicator &other)
      : _name(other._name), _comm_owned(true) {
    MPI_Comm_dup(other._comm, &_comm);
    init();
  }

  ~MPICommunicator() {
    if (_comm_owned && (_comm != MPI_COMM_NULL)) {
      MPI_Comm_free(&_comm);
    }
  }

  MPICommunicator &operator=(const MPICommunicator &) = delete;

  void init() {
    if (_comm == MPI_COMM_NULL) {
      throw std::runtime_error("MPI Communicator is MPI_COMM_NULL");
    }

    int result = MPI_Comm_rank(_comm, &_rank);
    if (result != MPI_SUCCESS) {
      throw std::runtime_error("Failed to get MPI rank");
    }

    result = MPI_Comm_size(_comm, &_size);
    if (result != MPI_SUCCESS) {
      throw std::runtime_error("Failed to get MPI size");
    }
  }

  [[nodiscard]] MPI_Comm comm() const noexcept { return _comm; }
  [[nodiscard]] int rank() const noexcept { return _rank; }
  [[nodiscard]] int size() const noexcept { return _size; }
  [[nodiscard]] const std::string &name() const noexcept { return _name; }

  [[nodiscard]] bool is_null() const noexcept { return _comm == MPI_COMM_NULL; }
  [[nodiscard]] bool is_root() const noexcept { return _rank == 0; }

  void barrier() const { MPI_Barrier(_comm); }
  void abort(int errorcode) const { MPI_Abort(_comm, errorcode); }
  void info() const {
    if (is_root()) {
      std::cout << "[INFO] MPI Communicator: " << _name << std::endl;
      std::cout << "  Size: " << _size << std::endl;
      std::cout << "  Rank: " << _rank << std::endl;
    }
  }

private:
  MPI_Comm _comm;
  int _rank;
  int _size;
  std::string _name;
  bool _comm_owned;
};
} // namespace hpc::mpi
#endif
