#ifdef ENABLE_SIMD

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <simd/simd.h>

#else
#include <immintrin.h>
#endif

#endif
