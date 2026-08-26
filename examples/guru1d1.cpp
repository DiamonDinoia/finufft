// this is all you must include for the finufft lib...
#include <finufft.hpp>

// specific to this example...
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>

// only good for small projects...
using namespace std;

// allows 1i to be the imaginary unit... (C++14 onwards)
using namespace std::complex_literals;

int main()
/* Example calling the RAII C++ interface to the FINUFFT library, with
   STL vectors of double complex numbers and a math check.
   Barnett 2/27/20
   To compile see README. Also see ../docs/cex.rst
   Usage: ./guru1d1
*/
{
  int M      = 1e6;      // number of nonuniform points
  int N      = 1e6;      // number of modes
  double tol     = 1e-9;     // desired accuracy
  int ntransf = 1;       // we want to do a single transform at a time

  int changeopts = 0;    // do you want to try changing opts? 0 or 1
  auto opts      = finufft::default_opts<double>();
  if (changeopts) opts.debug = 1; // example options change

  // the tolerance literal picks double precision; errors throw finufft::error
  finufft::plan p(1, {N}, +1, tol, ntransf, opts);

  // generate some random nonuniform points
  vector<double> x(M);
  for (int j = 0; j < M; ++j)
    x[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1); // unif in [-pi,pi)
  p.setpts(x); // one span selects a 1d point set

  // generate some complex strengths
  vector<complex<double>> c(M);
  for (int j = 0; j < M; ++j)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + 1i * (2 * ((double)rand() / RAND_MAX) - 1);

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<double>> F(N);
  p.execute(c, F);

  // could now change c, do another execute, do another setpts, execute, etc...
  // the plan frees itself when it goes out of scope

  // rest is math checking and reporting...
  int k                 = 142519; // check the answer just for this mode frequency...
  complex<double> Ftest = 0.0 + 0.0i;
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(1i * (double)k * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = abs(F[kout] - Ftest) / Fmax;
  std::cout << "guru 1D type-1 double-prec NUFFT done. rel err in F[" << k << "] is "
            << err << '\n';
  return 0;
}
