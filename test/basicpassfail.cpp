#include "utils/test_defs.hpp"

// Basic pass-fail test of one routine in library w/ default opts.
// exit code 0 success, failure otherwise. This is useful for brew recipe.
// Works for either single/double: the body is templated on FLT and the
// FINUFFT_TEST_PREC macro (set per target) selects the instantiation.
// Simplified from Amit Moscovitz and example1d1. Barnett 11/1/18.
// Using vectors and default opts, 2/29/20; dual-prec lib 7/3/20.

template<typename FLT> int run() {
  using CPX  = std::complex<FLT>;
  using CAPI = finufft_capi<FLT>;
  BIGINT M = 1e3, N = 1e3;            // defaults: M = # srcs, N = # modes out
  // single precision rounding floor is ~1.4e-4 at N=1e3, so use a looser tol
  const double tol = std::is_same_v<FLT, float> ? 1e-3 : 1e-5;
  int isign        = +1;              // exponential sign for NUFFT
  const CPX I(0.0, 1.0);              // imaginary unit
  std::vector<CPX> F(N);              // alloc output mode coeffs

  // Make the input data....................................
  srand(42);                                             // seed (fixed)
  std::vector<FLT> x(M);                                 // NU pts locs
  std::vector<CPX> c(M);                                 // strengths
  for (BIGINT j = 0; j < M; ++j) {
    x[j] = PI * (2 * ((FLT)rand() / (FLT)RAND_MAX) - 1); // uniform random in
                                                         // [-pi,pi)
    c[j] = 2 * ((FLT)rand() / (FLT)RAND_MAX) - 1 +
           I * (2 * ((FLT)rand() / (FLT)RAND_MAX) - 1);
  }
  // Run it (nullptr = default opts) ....................................
  int ier = CAPI::f1d1(M, x.data(), c.data(), isign, FLT(tol), N, F.data(), nullptr);
  if (ier != 0) {
    printf("basicpassfail: finufft1d1 error (ier=%d)!", ier);
    exit(ier);
  }
  // Check correct math for a single mode...................
  BIGINT n  = (BIGINT)(0.37 * N); // choose some mode near the top (N/2)
  CPX Ftest = CPX(0.0, 0.0);      // crude exact answer & error check...
  for (BIGINT j = 0; j < M; ++j) Ftest += c[j] * exp((FLT)isign * I * (FLT)n * x[j]);
  BIGINT nout = n + N / 2;        // index in output array for freq mode n
  FLT Finfnrm = 0.0;              // compute inf norm of F...
  for (int m = 0; m < N; ++m) {
    FLT aF = abs(F[m]);           // note C++ abs complex type, not C fabs(f)
    if (aF > Finfnrm) Finfnrm = aF;
  }
  FLT relerr = abs(F[nout] - Ftest) / Finfnrm;
  // printf("requested tol %.3g: rel err for one mode %.3g\n",tol,relerr);
  return (std::isnan(relerr) || relerr > 10.0 * tol); // true reports failure
}

int main() { return run<FINUFFT_TEST_PREC>(); }
