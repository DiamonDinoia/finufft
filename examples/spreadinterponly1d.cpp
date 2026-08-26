// this is all you must include for the finufft lib...
#include <finufft.hpp>

// also used in this example...
#include <chrono>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>
using namespace std;

using namespace std::chrono;

int main()
/* Example of double-prec spread/interp only tasks, with basic math tests.
   Complex I/O arrays, but recall the kernel is real.  Barnett 1/8/25.

   The math tests are:
   1) for spread, check sum of spread kernel masses is as expected from sum
   of strengths (ie testing the zero-frequency component in NUFFT).
   2) for interp, check each interp kernel mass is the same as from one.

   Without knowing the kernel, this is about all that can be done!
   (Better math tests would be, ironically, to wrap the spreader/interpolator
   into a NUFFT and test that :) But we already have that in FINUFFT.)

   To compile, see README. Usage: ./spreadinterponly1d
   See: spreadtestnd for usage of internal (non FINUFFT-API) spread/interp.
*/
{
  int M                 = 1e7;  // number of nonuniform points
  int N                 = 1e7;  // size of regular grid
  auto opts             = finufft::default_opts<double>();
  opts.spreadinterponly = 1;    // task: the following control kernel used...
  double tol            = 1e-9; // tolerance for (real) kernel shape design only
  opts.upsampfac        = 2.0;  // pretend upsampling factor (really no upsampling)

  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit
  vector<double> x(M);                           // input
  vector<complex<double>> c(M);                  // input
  vector<complex<double>> F(N);                  // output (spread to this array)

  finufft::plan<double> p1(1, {N}, +1, tol, 1, opts); // spread-only "1d1"
  finufft::plan<double> p2(2, {N}, +1, tol, 1, opts); // interp-only "1d2"

  // first spread a single unit-strength at the origin, to get the kernel mass...
  vector<double> xone{0.0};
  vector<complex<double>> cone{1.0};
  p1.setpts(xone);
  p1.execute(cone, F);            // warm-up: M=1 spread
  complex<double> kersum = 0.0;
  for (auto Fk : F) kersum += Fk; // kernel mass

  // Now generate random nonuniform points (x) and complex strengths (c)...
  for (int j = 0; j < M; ++j) {
    x[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1); // unif in [-pi,pi)
    c[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }

  auto t0 = steady_clock::now(); // now spread with all M pts... (dir=1)
  p1.setpts(x);
  p1.execute(c, F);
  double t             = (steady_clock::now() - t0) / 1.0s;
  complex<double> csum = 0.0; // tot input strength
  for (auto cj : c) csum += cj;
  complex<double> mass = 0.0; // tot output mass
  for (auto Fk : F) mass += Fk;
  double relerr = abs(mass - kersum * csum) / abs(mass);
  std::cout << "1D spread-only, double-prec, " << t << " s (" << M / t
            << " NU pt/sec), mass err " << relerr << '\n';

  for (auto &Fk : F) Fk = complex<double>{1.0, 0.0}; // unit grid input
  t0 = steady_clock::now(); // now interp to all M pts...  (dir=2)
  p2.setpts(x);
  p2.execute(c, F);         // type 2: F is the grid input, c the output at NU pts
  t    = (steady_clock::now() - t0) / 1.0s;
  csum = 0.0; // tot output
  for (auto cj : c) csum += cj;
  double maxerr = 0.0;
  for (auto cj : c) maxerr = max(maxerr, abs(cj - kersum));
  std::cout << "1D interp-only, double-prec, " << t << " s (" << M / t
            << " NU pt/sec), max err " << maxerr / abs(kersum) << '\n';
  return 0;
}
