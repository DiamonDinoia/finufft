/* Demonstrate the RAII C++ interface performing a stack of 1d type 1
   transforms in a single execute call. See guru1d1.cpp for other
   features demonstrated. Barnett 11/22/23
   To compile, see README.
   Usage: ./gurumany1d1           (exit code 0 indicates success)
*/

// this is all you must include for the finufft lib...
#include <finufft.hpp>

// specific to this demo...
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>

// only good for small projects...
using namespace std;

// allows 1i to be the imaginary unit... (C++14 onwards)
using namespace std::complex_literals;

int main() {
  int M      = 2e5;          // number of nonuniform points
  int N      = 1e5;          // number of modes
  double tol = 1e-9;         // desired accuracy
  int ntrans = 100;          // request a bunch of transforms in the execute
  int isign  = +1;           // sign of i in the transform math definition

  finufft::plan p(1, {N}, isign, tol, ntrans);

  // generate random nonuniform points and pass to FINUFFT
  vector<double> x(M);
  for (int j = 0; j < M; ++j)
    x[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1); // unif in [-pi,pi)
  p.setpts(x);

  // generate ntrans complex strength vectors each of length M (the slow bit!)
  vector<complex<double>> c(M * ntrans); // plain contiguous storage
  for (int j = 0; j < M * ntrans; ++j)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + 1i * (2 * ((double)rand() / RAND_MAX) - 1);

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<double>> F(N * ntrans);
  std::cout << "guru many 1D type-1 double-prec, tol=" << tol << ", executing " << ntrans
            << " transforms (vectorized), each size " << M << " NU pts to " << N
            << " modes...\n";
  p.execute(c, F); // spans carry the packed sizes; they are checked

  // could now change c, do another execute, do another setpts, execute, etc...

  // rest is math checking and reporting...
  int k     = 42519;                                // check one mode
  int trans = 71;                                   // ...in this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2); // ensure meaningful test
  assert(trans >= 0 && trans < ntrans);
  complex<double> Ftest = 0.0 + 0.0i;
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(1i * (double)k * x[j]); // c offset to trans
  double Fmax = 0.0; // compute inf norm of F for selected transform
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m + N * trans]);
    if (aF > Fmax) Fmax = aF;
  }
  int nout   = k + N / 2 + N * trans; // output index, freq mode k, transform #trans
  double err = abs(F[nout] - Ftest) / Fmax;
  std::cout << "rel err in F[" << k << "] (trans=" << trans << ") is " << err << '\n';
  return 0;
}
