// this is all you must include for the finufft lib...
#include <finufft.hpp>

#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>
using namespace std;

int main() {
  /* 2D demo of computing the *adjoint* of the planned transform.
     We plan a type 2, and then perform its adjoint (which is a type 1 with
     the opposite isign). Uses STL double complex vectors, with a math test.
     Computes an identical transform to guru2d1 except using the
     execute_adjoint feature. Barbone and Barnett, June 2025.
     To compile, see README.  Usage: ./guru2d1_adjoint
  */
  int M      = 1e6;  // number of nonuniform points
  int N      = 1e6;  // approximate total number of modes (N1*N2)
  double tol = 1e-6; // desired accuracy
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

  auto opts      = finufft::default_opts<double>();
  opts.upsampfac = 1.25;

  // step 1: make a plan... note we choose isign=-1 for this type 2 plan
  finufft::plan p(2, {N1, N2}, -1, tol, 1, opts);
  // step 2: send in M nonuniform points (just x, y in this case)...
  p.setpts(x, y);
  // step 3: do the adjoint of the planned transform. This maps
  // c strength data, to F output, and is identical to the type 1 with isign=+1.
  p.execute_adjoint(c, F);
  // ... you could now send in new points, and/or do transforms or their adjoints.

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
  int k1out    = k1 + (int)N1 / 2;
  int k2out    = k2 + (int)N2 / 2;
  int indexOut = k1out + k2out * (N1);

  // compute relative error
  double err = abs(F[indexOut] - Ftest) / Fmax;
  cout << "2D adjoint-of-type-2 NUFFT done. err in F[" << indexOut
       << "] rel to max(F) is " << setprecision(2) << err << endl;
  return 0;
}
