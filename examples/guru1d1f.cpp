// this is all you must include for the finufft lib...
#include <finufft.hpp>

// specific to this example...
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>

// only good for small projects...
using namespace std;

// allows 1i to be the imaginary unit... (C++14 onwards)
using namespace std::complex_literals;

int main()
/* Example calling the RAII C++ interface to the FINUFFT library,
   single-prec, with STL vectors of float complex numbers and a math check.
   Barnett 7/5/20
   To compile, see README.  Usage: ./guru1d1f
*/
{
  int M     = 1e5;                // number of nonuniform points
  int N     = 1e4;                // number of modes
  float tol = 1e-3;               // desired accuracy
  int ntransf    = 1;                  // single transform at a time

  int changeopts = 1;             // do you want to try changing opts? 0 or 1
  auto opts      = finufft::default_opts<float>();
  if (changeopts) opts.debug = 2; // example options change

  // the tolerance literal picks single precision; errors throw finufft::error
  finufft::plan p(1, {N}, +1, tol, ntransf, opts);

  // generate some random nonuniform points
  vector<float> x(M);
  for (int j = 0; j < M; ++j)
    x[j] = numbers::pi_v<float> * (2 * ((float)rand() / (float)RAND_MAX) - 1);
  p.setpts(x);

  // generate some complex strengths
  vector<complex<float>> c(M);
  for (int j = 0; j < M; ++j)
    c[j] = 2 * ((float)rand() / (float)RAND_MAX) - 1 +
           1if * (2 * ((float)rand() / (float)RAND_MAX) - 1);

  // alloc output array for the Fourier modes, then do the transform
  vector<complex<float>> F(N);
  p.execute(c, F);

  // rest is math checking and reporting...
  int k                = 1425; // check the answer just for this mode frequency...
  complex<float> Ftest = 0.0f + 0.0if;
  for (int j = 0; j < M; ++j) Ftest += c[j] * exp(1if * (float)k * x[j]);
  float Fmax = 0.0; // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    float aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  int kout  = k + N / 2; // index in output array for freq mode k
  float err = abs(F[kout] - Ftest) / Fmax;
  std::cout << "guru 1D type-1 single-prec NUFFT done. rel err in F[" << k << "] is "
            << err << '\n';
  return 0;
}
