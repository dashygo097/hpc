#pragma once

#ifdef SIMD_APPLE
#include <simd/simd.h>
#endif

#ifdef SIMD_NEON
#include <arm_neon.h>
#endif

#ifdef SIMD_AVX
#include <immintrin.h>
#endif

#ifdef SIMD_SSE
#include <xmmintrin.h>
#endif
