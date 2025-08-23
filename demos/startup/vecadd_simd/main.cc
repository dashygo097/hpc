#include <assert.h>
#include <simd/simd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Scalar version for comparison
void vecadd_scalar(const float *a, const float *b, float *result,
                   size_t count) {
  for (size_t i = 0; i < count; i++) {
    result[i] = a[i] + b[i];
  }
}

// SIMD version using simd_float4 (processes 4 floats at once)
void vecadd_simd4(const float *a, const float *b, float *result, size_t count) {
  size_t simd_count = count - (count % 4);
  size_t i;

  // Process 4 elements at a time
  for (i = 0; i < simd_count; i += 4) {
    simd_float4 va = simd_make_float4(a[i], a[i + 1], a[i + 2], a[i + 3]);
    simd_float4 vb = simd_make_float4(b[i], b[i + 1], b[i + 2], b[i + 3]);
    simd_float4 vr = va + vb;

    result[i] = vr.x;
    result[i + 1] = vr.y;
    result[i + 2] = vr.z;
    result[i + 3] = vr.w;
  }

  // Handle remaining elements
  for (; i < count; i++) {
    result[i] = a[i] + b[i];
  }
}

// More efficient SIMD version using direct memory loading
void vecadd_simd_efficient(const float *a, const float *b, float *result,
                           size_t count) {
  size_t simd_count = count - (count % 4);

  for (size_t i = 0; i < simd_count; i += 4) {
    // Load 4 floats directly from memory
    simd_float4 va = *((simd_float4 *)(a + i));
    simd_float4 vb = *((simd_float4 *)(b + i));
    simd_float4 vr = va + vb;

    // Store result directly to memory
    *((simd_float4 *)(result + i)) = vr;
  }

  // Handle remaining elements
  for (size_t i = simd_count; i < count; i++) {
    result[i] = a[i] + b[i];
  }
}

// Utility function to initialize arrays with random values
void init_array(float *arr, size_t count, float min_val, float max_val) {
  for (size_t i = 0; i < count; i++) {
    arr[i] = min_val + (float)rand() / RAND_MAX * (max_val - min_val);
  }
}

// Utility function to verify results
int verify_results(const float *a, const float *b, size_t count,
                   float tolerance) {
  for (size_t i = 0; i < count; i++) {
    if (fabs(a[i] - b[i]) > tolerance) {
      printf("Mismatch at index %zu: %f vs %f\n", i, a[i], b[i]);
      return 0;
    }
  }
  return 1;
}

// Benchmark function
double benchmark(void (*func)(const float *, const float *, float *, size_t),
                 const float *a, const float *b, float *result, size_t count,
                 int iterations) {
  clock_t start = clock();

  for (int i = 0; i < iterations; i++) {
    func(a, b, result, count);
  }

  clock_t end = clock();
  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int main() {
  printf("SIMD Vector Addition Example on macOS\n");
  printf("====================================\n\n");

  srand((unsigned int)time(NULL));

  const size_t count = 1000000; // 1M elements
  const int iterations = 1000;  // For benchmarking

  // Allocate aligned memory for better SIMD performance
  float *a = (float *)aligned_alloc(32, count * sizeof(float));
  float *b = (float *)aligned_alloc(32, count * sizeof(float));
  float *result_scalar = (float *)aligned_alloc(32, count * sizeof(float));
  float *result_simd4 = (float *)aligned_alloc(32, count * sizeof(float));
  float *result_efficient = (float *)aligned_alloc(32, count * sizeof(float));

  if (!a || !b || !result_scalar || !result_simd4 || !result_efficient) {
    printf("Memory allocation failed!\n");
    return 1;
  }

  // Initialize arrays
  init_array(a, count, -100.0f, 100.0f);
  init_array(b, count, -100.0f, 100.0f);

  printf("Processing %zu elements...\n\n", count);

  // Test correctness first
  vecadd_scalar(a, b, result_scalar, count);
  vecadd_simd4(a, b, result_simd4, count);
  vecadd_simd_efficient(a, b, result_efficient, count);

  // Verify results
  printf("Correctness verification:\n");
  printf("SIMD4 vs Scalar:    %s\n",
         verify_results(result_scalar, result_simd4, count, 1e-6f) ? "PASS"
                                                                   : "FAIL");
  printf("Efficient vs Scalar: %s\n",
         verify_results(result_scalar, result_efficient, count, 1e-6f)
             ? "PASS"
             : "FAIL");

  printf("\nBenchmarking (%d iterations):\n", iterations);

  // Benchmark performance
  double time_scalar =
      benchmark(vecadd_scalar, a, b, result_scalar, count, iterations);
  double time_simd4 =
      benchmark(vecadd_simd4, a, b, result_simd4, count, iterations);

  double time_efficient = benchmark(vecadd_simd_efficient, a, b,
                                    result_efficient, count, iterations);

  printf("Scalar version:      %.4f seconds\n", time_scalar);
  printf("SIMD4 version:       %.4f seconds (%.2fx speedup)\n", time_simd4,
         time_scalar / time_simd4);
  printf("Efficient SIMD:      %.4f seconds (%.2fx speedup)\n", time_efficient,
         time_scalar / time_efficient);

  // Show some sample results
  printf("\nSample results (first 10 elements):\n");
  printf("Index |    A     |    B     |  Result\n");
  printf("------|----------|----------|----------\n");
  for (int i = 0; i < 10; i++) {
    printf("  %2d  | %8.3f | %8.3f | %8.3f\n", i, a[i], b[i], result_scalar[i]);
  }

  // Clean up
  free(a);
  free(b);
  free(result_scalar);
  free(result_simd4);
  free(result_efficient);

  return 0;
}
