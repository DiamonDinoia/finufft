#include <finufft.h>

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const int M = 100000, N = 1000;
  const double tol = 1e-9, pi = 3.14159265358979323846;

  double *x         = (double *)malloc(sizeof(double) * M);
  double complex *c = (double complex *)malloc(sizeof(double complex) * M);
  double complex *F = (double complex *)malloc(sizeof(double complex) * N);
  for (int j = 0; j < M; ++j) {
    x[j] = pi * (2 * (double)rand() / RAND_MAX - 1);
    c[j] = 2 * (double)rand() / RAND_MAX - 1 + I * (2 * (double)rand() / RAND_MAX - 1);
  }

  finufft_opts opts;
  finufft_default_opts(&opts);
  const int ier = finufft1d1(M, x, c, +1, tol, N, F, &opts);
  if (ier > 1) {
    printf("finufft1d1 failed, ier=%d\n", ier);
    return 1;
  }

  const int k       = N / 3;
  double complex Fk = 0.0 + 0.0 * I;
  for (int j = 0; j < M; ++j) Fk += c[j] * cexp(I * (double)k * x[j]);
  double Fmax = 0.0;
  for (int m = 0; m < N; ++m)
    if (cabs(F[m]) > Fmax) Fmax = cabs(F[m]);

  const double err = cabs(F[k + N / 2] - Fk) / Fmax;
  printf("finufft1d1: ier=%d, rel err in mode %d is %.3g\n", ier, k, err);
  free(x);
  free(c);
  free(F);
  return err < 1e-7 ? 0 : 1;
}
