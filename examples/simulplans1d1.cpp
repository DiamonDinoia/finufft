/* Demo two simultaneous FINUFFT plans (A,B) being handled in C++ without
   interacting (or at least without crashing; note that FFTW initialization
   is the only global state of FINUFFT library).
   Using STL double complex vectors, with a math test.
   To compile, see README in this directory. Also see ../docs/cex.rst
   Edited from guru1d1, Barnett 2/15/22
   Usage: ./simulplans1d1
*/

// this is all you must include for the finufft lib...
#include <finufft.hpp>

// also used in this example...
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>
using namespace std;

void strengths(vector<complex<double>> &c) { // fill random complex array
  for (long unsigned int j = 0; j < c.size(); ++j)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + 1i * (2 * ((double)rand() / RAND_MAX) - 1);
}

double chk1d1(int n, vector<double> &x, vector<complex<double>> &c,
              vector<complex<double>> &F)
// return error in output array F, for n'th mode only, rel to ||F||_inf
{
  int N = F.size();
  if (n >= N / 2 || n < -N / 2) {
    std::cout << "n out of bounds!\n";
    return NAN;
  }
  complex<double> Ftest = complex<double>(0, 0);
  for (long unsigned int j = 0; j < x.size(); ++j)
    Ftest += c[j] * exp(1i * (double)n * x[j]);
  int nout    = n + N / 2; // index in output array for freq mode n
  double Fmax = 0.0;       // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  return abs(F[nout] - Ftest) / Fmax;
}

int main() {
  double tol = 1e-9;             // desired accuracy for both plans

  int MA = 3e6;              // number of nonuniform points    PLAN A
  int NA = 1e6;              // number of modes
  int MB = 2e6;              // number of nonuniform points    PLAN B, diff sizes
  int NB = 1e5;              // number of modes

  finufft::plan planA(1, {NA}, +1, tol);
  finufft::plan planB(1, {NB}, +1, tol);

  // generate some random nonuniform points
  vector<double> xA(MA), xB(MB);
  for (int j = 0; j < MA; ++j)
    xA[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1); // unif in [-pi,pi)
  for (int j = 0; j < MB; ++j)
    xB[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1);

  planA.setpts(xA);
  planB.setpts(xB);

  // generate some complex strengths
  vector<complex<double>> cA(MA), cB(MB);
  strengths(cA);
  strengths(cB);

  // allocate output arrays for the Fourier modes...
  vector<complex<double>> FA(NA), FB(NB);
  planA.execute(cA, FA);
  planB.execute(cB, FB);

  // change strengths and exec again for fun...
  strengths(cA);
  strengths(cB);
  planA.execute(cA, FA);
  planB.execute(cB, FB);
  // both plans free themselves at the end of the scope

  // math checking and reporting...
  int n       = 116354;
  double errA = chk1d1(n, xA, cA, FA);
  std::cout << "planA: 1D type-1 double-prec NUFFT done. rel err in F[" << n << "] is "
            << errA << '\n';
  n           = 27152;
  double errB = chk1d1(n, xB, cB, FB);
  std::cout << "planB: 1D type-1 double-prec NUFFT done. rel err in F[" << n << "] is "
            << errB << '\n';

  return 0;
}
