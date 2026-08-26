// this is all you must include for the finufft lib...
#include <finufft.hpp>

// also used in this example...
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <omp.h>
#include <sstream>
#include <vector>
using namespace std;

int main()
/* Demo single-threaded FINUFFT calls from inside a OMP parallel block.
   Adapted from simple1d1.cpp: C++, STL double complex vectors, with math test.
   Barnett 4/19/21, eg for Goran Zauhar, issue #183. Also see: many1d1.cpp.
   To compile, see README.
   Usage: ./threadsafe1d1
   Expected output: multiple text lines (however many default threads), each
   reporting small error.
*/
{
  int M             = 1e5;                       // number of nonuniform points
  int N             = 1e5;                       // number of modes
  double acc        = 1e-9;                      // desired accuracy
  complex<double> I = complex<double>(0.0, 1.0); // the imaginary unit

  int overallstatus = 0;

  // Now have each thread do independent 1D type 1 on their own data:
#pragma omp parallel
  {
    // generate some random nonuniform points (x) and complex strengths (c)...
    // Note that these are local to the thread (if you have the *same* sets of
    // NU pts x for each thread, consider instead using one vectorized multithreaded
    // transform, which would be faster).
    vector<double> x(M);
    vector<complex<double>> c(M);
    for (int j = 0; j < M; ++j) {
      x[j] = numbers::pi * (2 * ((double)rand() / RAND_MAX) - 1); // unif in [-pi,pi)
      c[j] =
          2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
    }

    // allocate output array for the Fourier modes... local to the thread
    vector<complex<double>> F(N);

    // *crucial* nthreads=1: each plan stays single-threaded (else oversubscription)
    int ier = 0;
    try {
      auto opts     = finufft::default_opts<double>();
      opts.nthreads = 1;
      finufft::plan p(1, {N}, +1, acc, 1, opts);
      p.setpts(x);
      p.execute(c, F);
    } catch (const finufft::error &e) {
      std::ostringstream msg;
      msg << "[thread " << omp_get_thread_num() << "] " << e.what() << '\n';
      std::cout << msg.str();
      ier = 1;
    }
    if (ier > 0) overallstatus = 1;

    int k = 42519; // check the answer just for this mode frequency...
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

    std::ostringstream msg;
    msg << "[thread " << omp_get_thread_num()
        << "] 1D t-1 dbl-prec NUFFT done. rel err in F[" << k << "]: " << err << '\n';
    std::cout << msg.str();
  }

  return overallstatus;
}
