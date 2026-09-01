#include <cufinufft.h>

#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failed(cudaError_t err, const char *what) {
  if (err == cudaSuccess) return 0;
  std::printf("%s: %s\n", what, cudaGetErrorString(err));
  return 1;
}

int main() {
  const int M = 100000, N = 1000;
  const double tol = 1e-9, pi = 3.14159265358979323846;
  const std::complex<double> I(0.0, 1.0);

  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M), F(N);
  for (int j = 0; j < M; ++j) {
    x[j] = pi * (2 * (double)rand() / RAND_MAX - 1);
    c[j] = {2 * (double)rand() / RAND_MAX - 1, 2 * (double)rand() / RAND_MAX - 1};
  }

  double *d_x          = nullptr;
  cuDoubleComplex *d_c = nullptr;
  cuDoubleComplex *d_F = nullptr;
  if (failed(cudaMalloc(&d_x, M * sizeof(double)), "cudaMalloc") ||
      failed(cudaMalloc(&d_c, M * sizeof(cuDoubleComplex)), "cudaMalloc") ||
      failed(cudaMalloc(&d_F, N * sizeof(cuDoubleComplex)), "cudaMalloc") ||
      failed(cudaMemcpy(d_x, x.data(), M * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy") ||
      failed(
          cudaMemcpy(d_c, c.data(), M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice),
          "cudaMemcpy"))
    return 1;

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  const int ier = cufinufft1d1(M, d_x, d_c, +1, tol, N, d_F, &opts);
  if (ier > 1) {
    std::printf("cufinufft1d1 failed, ier=%d\n", ier);
    return 1;
  }
  if (failed(
          cudaMemcpy(F.data(), d_F, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost),
          "cudaMemcpy"))
    return 1;
  cudaFree(d_x);
  cudaFree(d_c);
  cudaFree(d_F);

  const int k = N / 3;
  std::complex<double> Fk(0.0, 0.0);
  for (int j = 0; j < M; ++j) Fk += c[j] * std::exp(I * (double)k * x[j]);
  double Fmax = 0.0;
  for (int m = 0; m < N; ++m) Fmax = std::max(Fmax, std::abs(F[m]));

  const double err = std::abs(F[k + N / 2] - Fk) / Fmax;
  std::printf("cufinufft1d1: ier=%d, rel err in mode %d is %.3g\n", ier, k, err);
  return err < 1e-7 ? 0 : 1;
}
