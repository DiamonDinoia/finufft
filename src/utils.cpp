#include <common/common.h>
#include <cmath>

#ifdef __CUDACC__
#include <cufinufft/types.h>
#endif

namespace finufft {
namespace common {

void gaussquad(int n, double *xgl, double *wgl) {
  double x = 0, dx = 0;
  int convcount = 0;

  xgl[n / 2] = 0;
  for (int i = 0; i < n / 2; i++) {
    convcount = 0;
    x = std::cos((2 * i + 1) * ::finufft::common::PI / (2 * n));
    while (true) {
      auto [p, dp] = leg_eval(n, x);
      dx = -p / dp;
      x += dx;
      if (std::abs(dx) < 1e-14) {
        convcount++;
      }
      if (convcount == 3) {
        break;
      }
    }
    xgl[i] = -x;
    xgl[n - i - 1] = x;
  }

  for (int i = 0; i < n / 2 + 1; i++) {
    auto [junk1, dp] = leg_eval(n, xgl[i]);
    auto [p, junk2] = leg_eval(n + 1, xgl[i]);
    wgl[i] = -2 / ((n + 1) * dp * p);
    wgl[n - i - 1] = wgl[i];
  }
}

std::tuple<double, double> leg_eval(int n, double x) {
  if (n == 0) {
    return {1.0, 0.0};
  }
  if (n == 1) {
    return {x, 1.0};
  }
  double p0 = 0.0, p1 = 1.0, p2 = x;
  for (int i = 1; i < n; i++) {
    p0 = p1;
    p1 = p2;
    p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1);
  }
  return {p2, n * (x * p2 - p1) / (x * x - 1)};
}

} // namespace common
} // namespace finufft

namespace cufinufft {
#ifndef __CUDACC__
using CUFINUFFT_BIGINT = int;
#endif
namespace utils {

CUFINUFFT_BIGINT next235beven(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT b)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth") and is a multiple of b (b is a number that the only prime
// factors are 2,3,5). Adapted from fortran in hellskitchen. Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
// added condition about b, Melody Shih 05/31/20
{
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;                // even
  CUFINUFFT_BIGINT nplus  = n - 2;       // to cancel out the +=2 at start of loop
  CUFINUFFT_BIGINT numdiv = 2;           // a dummy that is >1
  while ((numdiv > 1) || (nplus % b != 0)) {
    nplus += 2;                          // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0) numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0) numdiv /= 3;
    while (numdiv % 5 == 0) numdiv /= 5;
  }
  return nplus;
}

} // namespace utils
} // namespace cufinufft

