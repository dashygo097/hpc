#pragma once

#include "./impls/mmul_impl.hh"
#include "./impls/vec_omp_impl.hh"

#ifdef ENABLE_SIMD
#include "./impls/simd_impl.hh"
#include "./impls/vec_omp_simd_impl.hh"
#endif
