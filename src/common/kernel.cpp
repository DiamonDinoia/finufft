#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <finufft_common/common.h>
#include <finufft_common/kernel.h>
#include <finufft_common/spread_opts.h>
#include <limits>

namespace finufft::common {
double pswf(double c, double x);
}

// this module uses finufft_spread_opts but does not know about FINUFFT_PLAN class
// nor finufft_opts. This allows it to be used by CPU & GPU.

namespace finufft::kernel {

FINUFFT_EXPORT double kernel_definition(const finufft_spread_opts &spopts,
                                        const double z) {
  /* The spread/interp kernel phi_beta(z) function on standard interval z in [-1,1],
     This evaluation does not need to be fast; it is used *only* for polynomial
     interpolation via Horner coeffs (the interpolant is evaluated fast).
     It can thus always be double-precision. No analytic Fourier transform pair is
     needed, thanks to numerical quadrature in finufft_core:onedim*; playing with
     new kernels is thus very easy.
    Inputs:
    z      - real ordinate on standard interval [-1,1]. Handling of edge cases
            at or near +-1 is no longer crucial, because precompute_horner_coeffs
            (the only user of this function) has interpolation nodes in (-1,1).
    spopts - spread_opts struct containing fields:
      beta        - shape parameter for ES, KB, or other prolate kernels
                    (a.k.a. c parameter in PSWF).
      kerformula  - positive integer selecting among kernel function types; see
                    docs in the code below.
                    (More than one value may give the same type, to allow
                    kerformula also to select a parameter-choice method.)
                    Note: the default 0 (in opts.spread_kerformula) is invalid here;
                    selection of a >0 kernel type must already have happened.
    Output: phi(z), as in the notation of original 2019 paper ([FIN] in the docs).

    Notes: 1) no normalization of max value or integral is needed, since any
            overall factor is cancelled out in the deconvolve step. However,
            values as large as exp(beta) have caused floating-pt overflow; don't
            use them.
    Barnett rewritten 1/13/26 for double on [-1,1]; based on Barbone Dec 2025.
  */
  if (std::abs(z) > 1.0) return 0.0;           // restrict support to [-1,1]
  double beta = spopts.beta;                   // get shape param
  double arg  = beta * std::sqrt(1.0 - z * z); // common argument for exp, I0, etc
  int kf      = spopts.kerformula;

  if (kf == 1 || kf == 2)
    // ES ("exponential of semicircle" or "exp sqrt"), see [FIN] reference.
    // Used in FINUFFT 2017-2025 (up to v2.4.1). max is 1, as of v2.3.0.
    return std::exp(arg) / std::exp(beta);
  else if (kf == 3)
    // forwards Kaiser--Bessel (KB), normalized to max of 1.
    // std::cyl_bessel_i is from <cmath>, expects double. See src/common/utils.cpp
    return common::cyl_bessel_i(0, arg) / common::cyl_bessel_i(0, beta);
  else if (kf == 4)
    // continuous (deplinthed) KB, as in Barnett SIREV 2022, normalized to max nearly 1
    return (common::cyl_bessel_i(0, arg) - 1.0) / common::cyl_bessel_i(0, beta);
  else if (kf == 5)
    return std::cosh(arg) / std::cosh(beta); // normalized cosh-type of Rmk. 13 [FIN]
  else if (kf == 6)
    return (std::cosh(arg) - 1.0) / std::cosh(beta); // Potts-Tasche cont cosh-type
  else if (kf == 7)
    // Prolate spheroidal wave function (PSWF), normalized to max 1.
    return common::pswf(beta, z);
  else if (kf == 8)
    // PSWF with alternate beta-tuning (kf==8 uses same evaluation as 7).
    return common::pswf(beta, z);
  else {
    fprintf(stderr, "[%s] unknown spopts.kerformula=%d\n", __func__, spopts.kerformula);
    throw int(FINUFFT_ERR_KERFORMULA_NOTVALID);      // *** crashes matlab, not good
    return std::numeric_limits<double>::quiet_NaN(); // never gets here, non-signalling
  }
}

int theoretical_kernel_ns(double tol, int dim, int type, int debug,
                          const finufft_spread_opts &spopts) {
  // returns ideal preferred spread width (ns, a.k.a. w) using convergence rate,
  // in exact arithmetic, to achieve requested tolerance tol. Possibly uses
  // other parameters in spopts (upsampfac, kerformula,...). No clipping of ns
  // to valid range done here. Input upsampfac must be >1.0.
  int ns       = 0;
  double sigma = spopts.upsampfac;

  if (spopts.kerformula == 1) // ES legacy ns choice (v2.4.1, ie 2025, and before)
    if (sigma == 2.0)
      ns = (int)std::ceil(std::log10(10.0 / tol));
    else
      ns = (int)std::ceil(
          std::log(1.0 / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)));
  else { // generic formula for PSWF-like kernels.
    // tweak tolfac and nsoff for user tol matching (& tolsweep passing)...
    const double tolfac = (type == 3) ? 0.5 : 0.3; // only applies to outer of type 3
    const double nsoff  = 0.8; // width offset (helps balance err over sigma range)
    ns                  = (int)std::ceil(
        std::log(tolfac / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)) +
        nsoff);
  }
  return ns;
}

void set_kernel_shape_given_ns(finufft_spread_opts &spopts, int debug) {
  // Writes kernel shape parameter(s) (beta,...), into spopts, given previously-set
  // kernel info fields in spopts, principally: nspread, upsampfac, kerformula.
  // debug >0 causes stdout reporting.
  int ns       = spopts.nspread;
  double sigma = spopts.upsampfac;
  int kf       = spopts.kerformula;

  // these strings must match: kernel_definition(), the above, and the below
  const char *kernames[] = {"default",
                            "ES (legacy beta)",        // 1
                            "ES (Beatty beta)",        // 2
                            "KB (Beatty beta)",        // 3
                            "cont-KB (Beatty beta)",   // 4
                            "cosh (Beatty beta)",      // 5
                            "cont-cosh (Beatty beta)", // 6
                            "PSWF (Beatty beta)",      // 7
                            "PSWF (tuned beta)"};      // 8
  if (kf == 1) {
    // Exponential of Semicircle (ES), the legacy logic, from 2017, used to v2.4.1
    double betaoverns = 2.30;
    if (ns == 2)
      betaoverns = 2.20;
    else if (ns == 3)
      betaoverns = 2.26;
    else if (ns == 4)
      betaoverns = 2.38;

    if (sigma != 2.0) { // low-sigma option, introduced v1.0 (2018-2025)
      const double gamma = 0.97;
      betaoverns         = gamma * common::PI * (1.0 - 1.0 / (2.0 * sigma));
    }
    spopts.beta = betaoverns * (double)ns;
  } else if (kf == 8) {
    double t    = (double)ns * (1.0 - 1.0 / (2.0 * sigma));
    spopts.beta = ((-0.00149087 * t + 0.0218459) * t + 3.06269) * t - 0.0365245;
  } else if (kf == 3) {
    double t    = (double)ns * (1.0 - 1.0 / (2.0 * sigma));
    spopts.beta = -0.0149246 * t * t + 3.33163 * t - 0.664527;
  } else if (kf >= 2) {
    // Shape param formula (designed for K-B), from Beatty et al,
    // IEEE Trans Med Imaging, 2005 24(6):799-808. doi:10.1109/TMI.2005.848376
    // "Rapid gridding reconstruction with a minimal oversampling ratio".
    double t    = (double)ns * (1.0 - 1.0 / (2.0 * sigma));
    double c_beatty = (ns == 2) ? 0.5 : 0.8; // Beatty but tweak ns=2 for err fac 2 better
    spopts.beta = common::PI * std::sqrt(t * t - c_beatty); // just below std cutoff PI*t
    // in fact, in wsweepkerrcomp.m on KB we find beta=pi*t-0.17 is indistinguishable.
    // This is analogous to a safety factor of >0.99 around ns=10 (0.97 was too small)
  }

  // Plain shape param formula using std model for cutoff: (4.5) in [FIN], gamma=1:
  // spopts.beta = common::PI * (double)ns * (1.0 - 1.0 / (2.0 * sigma));
  // Expts show this formula with KB is 1/3-digit worse than Beatty, similar to ES.

  if (debug || spopts.debug)
    printf("[setup_spreadinterp]\tkerformula=%d: %s...\n", kf, kernames[kf]);
}

} // namespace finufft::kernel
