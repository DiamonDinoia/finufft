// this is all you must include for the finufft lib...
#include <finufft.hpp>

#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>
using namespace std;

int main()
/* Example of calling the vectorized FINUFFT library from modern C++, using
   STL double complex vectors, with a math test.
   To compile, see README.  Usage: ./many1d1
*/
{
  int ntrans        = 3;    // how many stacked transforms to do
  int M             = 1e6;  // nonuniform points (same for all transforms)
  int N             = 1e6;  // number of modes (same for all transforms)
  double tol        = 1e-9; // desired accuracy
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<double> x(M);
  vector<complex<double>> c(M * ntrans);
  for (int j = 0; j < M; ++j)
    x[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1); // unif in [-pi,pi)
  for (int j = 0; j < M * ntrans; ++j) // fill all ntrans vectors...
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  // allocate output array for the Fourier modes...
  vector<complex<double>> F(N * ntrans);

  finufft::plan p(1, {N}, +1, tol, ntrans);
  p.setpts(x);
  p.execute(c, F);

  int k     = 142519;     // check the answer just for this mode...
  int trans = ntrans - 1; // ...in this transform
  assert(k >= -(double)N / 2 && k < (double)N / 2);

  complex<double> Ftest = complex<double>(0, 0);           // do the naive calc...
  for (int j = 0; j < M; ++j)
    Ftest += c[j + M * trans] * exp(I * (double)k * x[j]); // c from transform # trans
  double Fmax = 0.0; // compute inf norm of F for transform # trans
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m + N * trans]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout   = k + N / 2 + N * trans; // output index, freq mode k, transform # trans
  double err = abs(F[kout] - Ftest) / Fmax;
  std::cout << "1D type-1 double-prec NUFFT done. rel err in F[" << k << "] is " << err
            << '\n';
  return 0;
}
