// this is all you must include to access cufinufft from C...
#include <cufinufft.h>

// also needed for this example...
#include <assert.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.141592653589793238462643383279502884;

int main()
/* Simple example of calling the cuFINUFFT library from C, using the simple
   (one-shot) interface and the C complex type, with a math test.
   Double-precision. C99 style. Needs a GPU. Usage: ./simple1d1c
*/
{
  int M             = 1e6;  // number of nonuniform points
  int N             = 1e6;  // number of modes
  double tol        = 1e-9; // desired accuracy

  // generate some random nonuniform points (x) and complex strengths (c):
  double *x         = (double *)malloc(sizeof(double) * M);
  double complex *c = (double complex *)malloc(sizeof(double complex) * M);
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * ((double)rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }
  // allocate complex output array for the Fourier modes
  double complex *F = (double complex *)malloc(sizeof(double complex) * N);

  // the transform reads and writes device memory, so copy the inputs over...
  double *d_x;
  cuDoubleComplex *d_c, *d_F;
  cudaMalloc((void **)&d_x, sizeof(double) * M);
  cudaMalloc((void **)&d_c, sizeof(cuDoubleComplex) * M);
  cudaMalloc((void **)&d_F, sizeof(cuDoubleComplex) * N);
  cudaMemcpy(d_x, x, sizeof(double) * M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, sizeof(cuDoubleComplex) * M, cudaMemcpyHostToDevice);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(cuerr));
    return 1;
  }

  cufinufft_opts opts;           // opts struct (not ptr)
  cufinufft_default_opts(&opts); // set default opts (must do this)
  opts.debug = 2;                // show how to override a default

  // call the NUFFT (with iflag=+1), passing device pointers...
  int ier    = cufinufft1d1(M, d_x, d_c, +1, tol, N, d_F, &opts);

  cudaMemcpy(F, d_F, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_F);

  int k = 142519;                       // check the answer just for this mode...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  double complex Ftest = 0.0 + 0.0 * I; // defined in complex.h (I too)
  for (int j = 0; j < M; ++j) Ftest += c[j] * cexp(I * (double)k * x[j]);
  double Fmax = 0.0;                    // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = cabs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = cabs(F[kout] - Ftest) / Fmax;
  printf("1D type 1 cuNUFFT done. ier=%d, err in F[%d] rel to max(F) is %.3g\n", ier, k,
         err);

  free(x);
  free(c);
  free(F);
  if (ier) return ier;
  return err < 1e-7 ? 0 : 1; // the ctest verdict is the error, not just ier
}
