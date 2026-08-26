// this is all you must include...
#include <finufft.hpp>

// also needed for this example...
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>
using namespace std;

int main()
/* Example of calling the FINUFFT library from modern C++, using STL
   single complex vectors, with a math test.
   (See simple1d1 for double-precision version.)
   To compile, see README. Usage: ./simple1d1f
*/
{
  int M            = 1e5;                      // number of nonuniform points
  int N            = 1e4;                      // number of modes
  float acc        = 1e-3;                     // desired accuracy
  complex<float> I = complex<float>(0.0, 1.0); // the imaginary unit

  // generate some random nonuniform points (x) and complex strengths (c)...
  vector<float> x(M);
  vector<complex<float>> c(M);
  for (int j = 0; j < M; ++j) {
    x[j] = numbers::pi_v<float> * (2 * ((float)rand() / (float)RAND_MAX) - 1); // unif in
                                                                               // [-pi,pi)
    c[j] = 2 * ((float)rand() / (float)RAND_MAX) - 1 +
           I * (2 * ((float)rand() / (float)RAND_MAX) - 1);
  }
  // allocate output array for the Fourier modes...
  vector<complex<float>> F(N);

  // the tolerance literal picks single precision; setpts deduces 1d
  finufft::plan p(1, {N}, +1, acc);
  p.setpts(x);
  p.execute(c, F);

  int k   = 1425; // check the answer just for this mode...
  assert(k >= -(double)N / 2 && k < (double)N / 2);
  complex<float> Ftest = complex<float>(0, 0);
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(I * (float)k * x[j]);
  float Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    float aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout  = k + N / 2; // index in output array for freq mode k
  float err = abs(F[kout] - Ftest) / Fmax;
  std::cout << "1D type-1 single-prec NUFFT done. rel err in F[" << k << "] is " << err
            << '\n';
  return 0;
}
