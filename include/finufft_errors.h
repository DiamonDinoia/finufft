#ifndef FINUFFT_ERRORS_H
#define FINUFFT_ERRORS_H

#include <finufft_common/defines.h>

// ---------- Global error/warning output codes for the library ---------------
// docs/error.rst embeds this enum, so each code is described once, here.
// clang-format off
// sphinx tag (don't remove): @error_codes_start
enum {
  FINUFFT_WARN_EPS_TOO_SMALL FINUFFT_DEPRECATED_ENUM(               // [DEPRECATED]
      "use FINUFFT_ERR_EPS_TOO_SMALL instead") = 1,                 // now code 26
  FINUFFT_ERR_MAXNALLOC        = 2,  // internal array would exceed MAX_NF (plan.hpp)
  FINUFFT_ERR_SPREAD_BOX_SMALL = 3,  // fine grid too small for the kernel width
  FINUFFT_ERR_SPREAD_PTS_OUT_RANGE FINUFFT_DEPRECATED_ENUM(         // [DEPRECATED]
      "bounds checking was removed in v2.3.0") = 4,                 // never returned
  FINUFFT_ERR_SPREAD_ALLOC FINUFFT_DEPRECATED_ENUM(                 // [DEPRECATED]
      "unused; was never returned by the library") = 5,             // never returned
  FINUFFT_ERR_SPREAD_DIR          = 6,   // illegal spread direction (1 or 2 only)
  FINUFFT_ERR_UPSAMPFAC_TOO_SMALL = 7,   // upsampfac too small (must be > 1.0)
  FINUFFT_ERR_HORNER_WRONG_BETA   = 8,   // no Horner rule for this upsampfac
  FINUFFT_ERR_NTRANS_NOTVALID     = 9,   // ntrans invalid (must be >= 1)
  FINUFFT_ERR_TYPE_NOTVALID       = 10,  // transform type invalid
  FINUFFT_ERR_ALLOC               = 11,  // general internal allocation failure
  FINUFFT_ERR_DIM_NOTVALID        = 12,  // dimension invalid
  FINUFFT_ERR_SPREAD_THREAD_NOTVALID FINUFFT_DEPRECATED_ENUM(       // [DEPRECATED]
      "unused; opts.spread_thread was deprecated in v2.6.0") = 13,  // never returned
  FINUFFT_ERR_NDATA_NOTVALID      = 14,  // mode array invalid: > ~2^31, or 0 modes
  FINUFFT_ERR_CUDA_FAILURE        = 15,  // a CUDA call, kernel or malloc failed
  FINUFFT_ERR_PLAN_NOTVALID       = 16,  // destroy called on an uninitialized plan
  FINUFFT_ERR_METHOD_NOTVALID     = 17,  // spread/interp method invalid for this dim
  FINUFFT_ERR_BINSIZE_NOTVALID    = 18,  // subprob/blockgather bin size invalid
  FINUFFT_ERR_INSUFFICIENT_SHMEM  = 19,  // GPU shmem too small for these parameters
  FINUFFT_ERR_NUM_NU_PTS_INVALID  = 20,  // nj or nk negative, or > MAX_NU_PTS
  FINUFFT_ERR_INVALID_ARGUMENT    = 21,  // invalid input no other code covers
  FINUFFT_ERR_LOCK_FUNS_INVALID   = 22,  // invalid FFTW lock function
  FINUFFT_ERR_NTHREADS_NOTVALID   = 23,  // nthreads invalid
  FINUFFT_ERR_KERFORMULA_NOTVALID = 24,  // spread kernel formula type invalid
  FINUFFT_ERR_UNKNOWN_EXCEPTION   = 25,  // unknown exception caught
  FINUFFT_ERR_EPS_TOO_SMALL       = 26,  // tolerance below machine epsilon
  FINUFFT_ERR_PSWF_SETUP          = 27,  // the PSWF evaluator setup did not converge
};
// sphinx tag (don't remove): @error_codes_end
// clang-format on
#endif
