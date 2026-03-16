#ifndef MATH_PSWF_H
#define MATH_PSWF_H

namespace finufft::common {
/*
normalized zeroth-order pswf: psi_0^c(x) / psi_0^c(0), supported on [-1,1]
*/
double pswf(double c, double x);

} // namespace finufft::common
#endif // MATH_PSWF_H
