// ------------ FINUFFT data type definitions ----------------------------------

#if (!defined(DATATYPES_H) && !defined(SINGLE)) || (!defined(DATATYPESF_H) && defined(SINGLE))
// Make sure we only include once per precision (as in finufft_eitherprec.h).
#ifndef SINGLE
#define DATATYPES_H
#else
#define DATATYPESF_H
#endif

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday...
// Note: BIGINT is modified to have ``int'' data type for cufinufft.
typedef int BIGINT;

// decide which kind of complex numbers to use in interface...
#ifdef __cplusplus
#include <complex>          // C++ type
#define COMPLEXIFY(X) std::complex<X>
#else
#include <complex.h>        // C99 type
#define COMPLEXIFY(X) X complex
#endif

#undef CUFINUFFT_FLT
#undef CPX

// Precision-independent real and complex types for interfacing...
// (note these cannot be typedefs since we want dual-precision library)
#ifdef SINGLE
  #define CUFINUFFT_FLT float
#else
  #define CUFINUFFT_FLT double
#endif

#define CPX COMPLEXIFY(CUFINUFFT_FLT)

#endif  // DATATYPES_H or DATATYPESF_H
