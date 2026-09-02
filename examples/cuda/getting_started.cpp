/*

  Simple example of the 1D type-1 transform. To compile, run

       nvcc -o getting_started getting_started.cpp -lcufinufft

  followed by

       ./getting_started

  with the necessary paths set if the library is not installed in the standard
  directories. If the library has been compiled in the standard way, this means

       export CPATH="${CPATH:+${CPATH}:}../../include"
       export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}../../build"
       export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}../../build"

 */

// sphinx tag (don't remove): @cuex_getting_started_includes_start
#include <complex.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufinufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// sphinx tag (don't remove): @cuex_getting_started_includes_end

static const double PI = 3.141592653589793238462643383279502884;

int main() {
  // sphinx tag (don't remove): @cuex_getting_started_params_start
  // Problem size: number of nonuniform points (M) and grid size (N).
  const int M = 100000, N = 10000;

  // Size of the grid as an array.
  int64_t modes[1] = {N};

  // Host pointers: frequencies (x), coefficients (c), and output (f).
  float *x;
  float _Complex *c;
  float _Complex *f;
  // sphinx tag (don't remove): @cuex_getting_started_params_end

  // sphinx tag (don't remove): @cuex_getting_started_device_ptrs_start
  // Device pointers.
  float *d_x;
  cuFloatComplex *d_c, *d_f;

  // Store cufinufft plan.
  cufinufftf_plan plan;
  // sphinx tag (don't remove): @cuex_getting_started_device_ptrs_end

  // sphinx tag (don't remove): @cuex_getting_started_manual_vars_start
  // Manual calculation at a single point idx.
  int idx;
  float _Complex f0;
  // sphinx tag (don't remove): @cuex_getting_started_manual_vars_end

  // sphinx tag (don't remove): @cuex_getting_started_fill_host_start
  // Allocate the host arrays.
  x = (float *)malloc(M * sizeof(float));
  c = (float _Complex *)malloc(M * sizeof(float _Complex));
  f = (float _Complex *)malloc(N * sizeof(float _Complex));

  // Fill with random numbers. Frequencies must be in the interval [-pi, pi)
  // while strengths can be any value.
  srand(0);

  for (int j = 0; j < M; ++j) {
    x[j] = 2 * PI * (((float)rand()) / RAND_MAX - 1);
    c[j] =
        (2 * ((float)rand()) / RAND_MAX - 1) + I * (2 * ((float)rand()) / RAND_MAX - 1);
  }
  // sphinx tag (don't remove): @cuex_getting_started_fill_host_end

  // sphinx tag (don't remove): @cuex_getting_started_alloc_copy_start
  // Allocate the device arrays and copy the x and c arrays.
  cudaMalloc(&d_x, M * sizeof(float));
  cudaMalloc(&d_c, M * sizeof(float _Complex));
  cudaMalloc(&d_f, N * sizeof(float _Complex));

  cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, M * sizeof(float _Complex), cudaMemcpyHostToDevice);
  // sphinx tag (don't remove): @cuex_getting_started_alloc_copy_end

  // sphinx tag (don't remove): @cuex_getting_started_makeplan_start
  // Make the cufinufft plan for a 1D type-1 transform with six digits of
  // tolerance. Any ier above 1 is an error; 1 is a warning and the result is
  // still usable.
  int ier = cufinufftf_makeplan(1, 1, modes, 1, 1, 1e-6, &plan, NULL);
  if (ier > 0) return ier;
  // sphinx tag (don't remove): @cuex_getting_started_makeplan_end

  // sphinx tag (don't remove): @cuex_getting_started_setpts_execute_start
  // Set the frequencies of the nonuniform points.
  ier = cufinufftf_setpts(plan, M, d_x, NULL, NULL, 0, NULL, NULL, NULL);
  if (ier > 0) return ier;

  // Actually execute the plan on the given coefficients and store the result
  // in the d_f array.
  ier = cufinufftf_execute(plan, d_c, d_f);
  if (ier > 0) return ier;
  // sphinx tag (don't remove): @cuex_getting_started_setpts_execute_end

  // sphinx tag (don't remove): @cuex_getting_started_copyback_destroy_start
  // Copy the result back onto the host.
  cudaMemcpy(f, d_f, N * sizeof(float _Complex), cudaMemcpyDeviceToHost);

  // Destroy the plan and free the device arrays after we're done.
  cufinufftf_destroy(plan);

  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_f);
  // sphinx tag (don't remove): @cuex_getting_started_copyback_destroy_end

  // sphinx tag (don't remove): @cuex_getting_started_print_start
  // Pick an index to check the result of the calculation.
  idx = 4 * N / 7;

  printf("f[%d] = %lf + %lfi\n", idx, crealf(f[idx]), cimagf(f[idx]));
  // sphinx tag (don't remove): @cuex_getting_started_print_end

  // sphinx tag (don't remove): @cuex_getting_started_manual_nudft_start
  // Calculate the result manually using the formula for the type-1
  // transform.
  f0 = 0;

  for (int j = 0; j < M; ++j) {
    f0 += c[j] * cexp(I * x[j] * (idx - N / 2));
  }

  printf("f0[%d] = %lf + %lfi\n", idx, crealf(f0), cimagf(f0));
  // sphinx tag (don't remove): @cuex_getting_started_manual_nudft_end

  // sphinx tag (don't remove): @cuex_getting_started_free_start
  // Finally free the host arrays.
  free(x);
  free(c);
  free(f);
  // sphinx tag (don't remove): @cuex_getting_started_free_end

  return 0;
}
