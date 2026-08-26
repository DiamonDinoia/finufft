// this is all you must include for the finufft lib...
#include <finufft.hpp>

// also needed for this example...
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>
using namespace std;

int main() {

  /* Simple 2D type-1 example of calling the FINUFFT library from modern C++,
     using STL double complex vectors, with a math test.
     To compile, see README. Usage:  ./simple2d1
  */

  int M      = 1e6;            // number of nonuniform points
  int N      = 1e6;            // approximate total number of modes (N1*N2)
  double tol = 1e-6;           // desired accuracy
  complex<double> I(0.0, 1.0); // the imaginary unit

  // generate random non-uniform points on (x,y) and complex strengths (c):
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);

  for (int i = 0; i < M; i++) {
    x[i] = numbers::pi * (2 * (double)rand() / RAND_MAX - 1); // unif in [-pi, pi)
    y[i] = numbers::pi * (2 * (double)rand() / RAND_MAX - 1);

    // each component uniform random in [-1,1]
    c[i] =
        2 * ((double)rand() / RAND_MAX - 1) + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }

  // choose numbers of output Fourier coefficients in each dimension
  int N1 = round(2.0 * sqrt(N));
  int N2 = round(N / N1);

  // output array for the Fourier modes
  vector<complex<double>> F(N1 * N2);

  // 2d type-1 plan with a low-upsampling option (iflag=+1)
  auto opts      = finufft::default_opts<double>();
  opts.upsampfac = 1.25;
  finufft::plan p(1, {N1, N2}, +1, tol, 1, opts);
  p.setpts(x, y); // two spans select two dimensions
  p.execute(c, F);

  int k1 = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  int k2 = round(-0.35 * N2);

  complex<double> Ftest(0, 0);
  for (int j = 0; j < M; j++)
    Ftest += c[j] * exp(I * ((double)k1 * x[j] + (double)k2 * y[j]));

  // compute inf norm of F
  double Fmax = 0.0;
  for (int m = 0; m < N1 * N2; m++) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }

  // indices in output array for this frequency pair (k1,k2)
  int k1out    = k1 + N1 / 2;
  int k2out    = k2 + N2 / 2;
  int indexOut = k1out + k2out * (N1);

  // compute relative error
  double err = abs(F[indexOut] - Ftest) / Fmax;
  cout << "2D type-1 NUFFT done. err in F[" << indexOut << "] rel to max(F) is "
       << setprecision(2) << err << endl;
  return 0;
}
