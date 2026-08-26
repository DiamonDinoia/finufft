// this is all you must include for the finufft lib...
#include <finufft.hpp>

// also used in this example...
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>
using namespace std;

int main()
/* Example of calling the FINUFFT library from modern C++, using STL
   double complex vectors, with a math test.
   Double-precision version (see simple1d1f for single-precision).
   To compile, see README in this directory.
   Also see ../docs/cex.rst or online documentation.
   Usage: ./simple1d1
*/
{
  int M             = 1e6;                       // number of nonuniform points
  int N             = 1e6;                       // number of modes
  double acc        = 1e-9;                      // desired accuracy
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<double> x(M);
  vector<complex<double>> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1); // unif in [-pi,pi)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }
  // allocate output array for the Fourier modes...
  vector<complex<double>> F(N);

  // make the 1d type-1 plan, pass the points, transform (iflag=+1)...
  // the tolerance literal picks double precision; errors throw finufft::error
  finufft::plan p(1, {N}, +1, acc);
  p.setpts(x);
  p.execute(c, F);

  int k = 142519; // check the answer just for this mode frequency...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  complex<double> Ftest = complex<double>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * (double)k * x[j]);
  double Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2; // index in output array for freq mode k
  double err = abs(F[kout] - Ftest) / Fmax;
  std::cout << "1D type-1 double-prec NUFFT done. rel err in F[" << k << "] is " << err
            << '\n';
  return 0;
}
