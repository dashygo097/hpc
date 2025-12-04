# Find and configure all external dependencies

# --- 3rdparty libraries (always included) ---
include_directories(3rdparty)
if(ENABLE_BUILD_BENCHMARKS)
  add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(3rdparty/benchmark EXCLUDE_FROM_ALL)
endif()

# --- CUDA ---
if(ENABLE_CUDA AND NOT APPLE)
  include(CheckLanguage)
  check_language(CUDA)
  
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(HAS_CUDA TRUE CACHE INTERNAL "CUDA is available")
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
    
    message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
  else()
    set(HAS_CUDA FALSE CACHE INTERNAL "CUDA is not available")
    message(WARNING "CUDA requested but compiler not found; CUDA support disabled.")
  endif()
else()
  set(HAS_CUDA FALSE CACHE INTERNAL "CUDA is not available")
  if(APPLE AND ENABLE_CUDA)
    message(STATUS "CUDA not supported on macOS; CUDA support disabled.")
  endif()
endif()

# --- OpenMP ---
if(ENABLE_OPENMP)
  if(APPLE)
    # macOS-specific OpenMP setup (Homebrew libomp)
    set(OPENMP_PATH "/opt/homebrew/opt/libomp" CACHE PATH "Path to OpenMP installation")
    if(EXISTS "${OPENMP_PATH}")
      set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OPENMP_PATH}/include")
      set(OpenMP_CXX_FLAGS "${OpenMP_C_FLAGS}")
      set(OpenMP_C_LIB_NAMES "omp")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${OPENMP_PATH}/lib/libomp.dylib")
    else()
      message(WARNING "OpenMP path ${OPENMP_PATH} not found.  Install with: brew install libomp")
    endif()
  endif()
  
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    set(HAS_OPENMP TRUE CACHE INTERNAL "OpenMP is available")
    message(STATUS "OpenMP C version: ${OpenMP_C_VERSION}")
    message(STATUS "OpenMP CXX version: ${OpenMP_CXX_VERSION}")
  else()
    set(HAS_OPENMP FALSE CACHE INTERNAL "OpenMP is not available")
    message(WARNING "OpenMP requested but not found")
  endif()
else()
  set(HAS_OPENMP FALSE CACHE INTERNAL "OpenMP is not available")
endif()

# --- SIMD Detection ---
if(ENABLE_SIMD)
  include(CheckCXXSourceCompiles)

  # Detect platform SIMD capabilities
  if(APPLE)
    # Apple SIMD
    check_cxx_source_compiles("
      #include <simd/simd.h>
      int main() {
        simd_float4 v = {1.0f, 2.0f, 3.0f, 4.0f};
        return 0;
      }
    " HAS_APPLE_SIMD)
      
    if(HAS_APPLE_SIMD)
      set(HAS_SIMD TRUE CACHE INTERNAL "SIMD is available")
      set(SIMD_TYPE "Apple" CACHE INTERNAL "SIMD implementation type")
      message(STATUS "SIMD: Apple Silicon SIMD detected (M-series chip)")
    else()
      set(HAS_SIMD FALSE CACHE INTERNAL "SIMD is not available")
      message(WARNING "SIMD requested but Apple SIMD headers not found")
    endif()
    
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    # ARM NEON
    check_cxx_source_compiles("
      #include <arm_neon.h>
      int main() {
        float32x4_t v = vdupq_n_f32(1.0f);
        return 0;
      }
    " HAS_ARM_NEON)
    
    if(HAS_ARM_NEON)
      set(HAS_SIMD TRUE CACHE INTERNAL "SIMD is available")
      set(SIMD_TYPE "NEON" CACHE INTERNAL "SIMD implementation type")
      message(STATUS "SIMD: ARM NEON detected")
      add_compile_options(-mfpu=neon)
    else()
      set(HAS_SIMD FALSE CACHE INTERNAL "SIMD is not available")
      message(WARNING "SIMD requested but ARM NEON not available")
    endif()
    
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # x86 SSE/AVX
    check_cxx_source_compiles("
      #include <immintrin.h>
      int main() {
        __m256 v = _mm256_set1_ps(1.0f);
        return 0;
      }
    " HAS_AVX)
    
    check_cxx_source_compiles("
      #include <xmmintrin.h>
      int main() {
        __m128 v = _mm_set1_ps(1.0f);
        return 0;
      }
    " HAS_SSE)
    
    if(HAS_AVX)
      set(HAS_SIMD TRUE CACHE INTERNAL "SIMD is available")
      set(SIMD_TYPE "AVX" CACHE INTERNAL "SIMD implementation type")
      add_compile_options(-mavx -mfma)
      message(STATUS "SIMD: AVX detected")
    elseif(HAS_SSE)
      set(HAS_SIMD TRUE CACHE INTERNAL "SIMD is available")
      set(SIMD_TYPE "SSE" CACHE INTERNAL "SIMD implementation type")
      add_compile_options(-msse4.2)
      message(STATUS "SIMD: SSE4.2 detected")
    else()
      set(HAS_SIMD FALSE CACHE INTERNAL "SIMD is not available")
      message(WARNING "SIMD requested but no x86 SIMD extensions found")
    endif()
  else()
    set(HAS_SIMD FALSE CACHE INTERNAL "SIMD is not available")
    message(WARNING "SIMD requested but platform ${CMAKE_SYSTEM_PROCESSOR} not recognized")
  endif()
else()
  set(HAS_SIMD FALSE CACHE INTERNAL "SIMD is not available")
endif()
 

# --- MPI ---
if(ENABLE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    set(HAS_MPI TRUE CACHE INTERNAL "MPI is available")
    message(STATUS "MPI C version: ${MPI_C_VERSION}")
    message(STATUS "MPI CXX version: ${MPI_CXX_VERSION}")
  else()
    set(HAS_MPI FALSE CACHE INTERNAL "MPI is not available")
    message(WARNING "MPI requested but not found")
  endif()
else()
  set(HAS_MPI FALSE CACHE INTERNAL "MPI is not available")
endif()

# --- Pybind11 ---
if(ENABLE_PYBIND11) 
  add_subdirectory(3rdparty/pybind11)

  set(HAS_PYBIND11 TRUE CACHE INTERNAL "pybind11 is available")
  message(STATUS "pybind11 found: ${pybind11_VERSION}")
else()
  set(HAS_PYBIND11 FALSE CACHE INTERNAL "pybind11 is not available")
endif()

# --- PyTorch ---
if(ENABLE_PYTORCH)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch. utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PYTHON_EXIT_CODE
  )
  
  if(NOT TORCH_PYTHON_EXIT_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to get PyTorch path.  Ensure PyTorch is installed: pip install torch")
  endif()
  
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
  
  set(CMAKE_FIND_LIBRARY_SUFFIXES_BACKUP ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(APPLE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ". dylib" ". a" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
  
  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_BACKUP})
  
  set(HAS_PYTORCH TRUE CACHE INTERNAL "PyTorch is available")
  message(STATUS "PyTorch found at: ${TORCH_CMAKE_PATH}")
  message(STATUS "PyTorch version: ${Torch_VERSION}")
else()
  set(HAS_PYTORCH FALSE CACHE INTERNAL "PyTorch is not available")
endif()

# --- Apple Accelerate ---
if(APPLE AND ENABLE_ACCELERATE)
  find_library(ACCELERATE_LIB Accelerate REQUIRED)
  set(HAS_ACCELERATE TRUE CACHE INTERNAL "Accelerate is available")
  message(STATUS "Apple Accelerate framework found")
else()
  set(HAS_ACCELERATE FALSE CACHE INTERNAL "Accelerate is not available")
endif()
