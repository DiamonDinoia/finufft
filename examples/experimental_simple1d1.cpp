// this is all you must include for the finufft lib...
#include <finufft.hpp>

int main(int argc, char *argv[])
/* Example of calling the FINUFFT library from C++, using STL
   double complex vectors, with a math test.
   Double-precision version (see simple1d1f for single-precision)

   Compile with (static library case):
   g++ simple1d1.cpp -I../include ../lib-static/libfinufft.a -o simple1d1 -lfftw3
   -lfftw3_omp or if you have built a single-core version: g++ simple1d1.cpp -I../include
   ../lib-static/libfinufft.a -o simple1d1 -lfftw3

   Usage: ./simple1d1

   Also see ../docs/cex.rst or online documentation.
*/
{
  int type  = 1;
  int iflag = 1;
  float eps = 1e-6;
  int M     = 1e6;                                      // number of nonuniform points
  int N     = 1e6;                                      // number of modes

  double acc = 1e-9;                                    // desired accuracy
  finufft::plan<float> nufft{type, {1024}, iflag, eps}; // create a default plan

  return 0;
}
