# Build configuration options

# Backend toggles
set(ENABLE_SIMD ON)
set(ENABLE_OPENMP ON)
set(ENABLE_MPI ON)
set(ENABLE_CUDA OFF)

# BLAS/LAPACK options
set(ENABLE_ACCELERATE OFF)

# Python options
set(ENABLE_PYBIND11 OFF)
set(ENABLE_PYTORCH OFF)

# Compilation options
set(USE_CCACHE ON)
set(USE_HIGH_LEVEL_OPTIMIZATIONS ON)

# Application options
set(ENABLE_BENCHMARKING ON)
