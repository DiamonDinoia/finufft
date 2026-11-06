#include <algorithm>
#include <cmath>
#include <cstdio>
#include <finufft_common/common.h>
#include <finufft_common/constants.h>
#include <finufft_common/kernel.h>
#include <finufft_common/pswf.h>
#include <finufft_common/safe_call.h>
#include <finufft_common/spread_opts.h>
#include <finufft_errors.h>
#include <functional>
#include <limits>

// this module uses finufft_spread_opts but does not know about FINUFFT_PLAN class
// nor finufft_opts. This allows it to be used by CPU & GPU.

namespace finufft::kernel {

std::function<double(double)> kernel_definition_lambda(double beta, int kf) {
  /* The spread/interp kernel phi_beta(z) function on standard interval z in [-1,1].
     This evaluation does not need to be fast; it is used *only* for polynomial
     interpolation via Horner coeffs (the interpolant is evaluated fast).
     It can thus always be double-precision. No analytic Fourier transform pair is
     needed, thanks to numerical quadrature in finufft_core:onedim*; playing with
     new kernels is thus very easy.
    Inputs:
    beta        - shape parameter for ES, KB, or other prolate kernels
                  (a.k.a. c parameter in PSWF).
    kerformula  - positive integer selecting among kernel function types; the
                  selection of a >0 kernel type must already have happened; see
                  kernel_beta_given_ns.
    Output: lambda function that can be called with z to return phi(z),
                    as in the notation of original 2019 paper ([FIN] in the docs).

    Notes: 1) no normalization of max value or integral is needed, since any
            overall factor is cancelled out in the deconvolve step. However,
            values as large as exp(beta) have caused floating-pt overflow; don't
            use them.
    Barnett rewritten 1/13/26 for double on [-1,1]; based on Barbone Dec 2025.
  */
  if (kf >= 7 && kf <= 9) {
    finufft::common::PSWF0 pswf(beta);
    return [pswf](double z) {
      if (std::abs(z) > 1.0) return 0.0; // restrict support to [-1,1]
      return pswf(z);
    }; // prolate (PSWF) Psi_0, normalized to 1 at z=0
  } else {
    fprintf(stderr, "[%s] unknown spopts.kerformula=%d\n", __func__, kf);
    throw finufft::exception(FINUFFT_ERR_KERFORMULA_NOTVALID);
  }
}

std::function<double(double)> kernel_definition_lambda(
    const finufft_spread_opts &spopts) {
  return kernel_definition_lambda(spopts.beta, spopts.kerformula);
}

static double kernel_tolfac(int dim, int type) {
  // Tolerance prefactor in the kernel aliasing law tol = tolfac*exp(-(ns-NSOFF)*pi*u).
  // PER_DIM^(dim-1) covers the measured error growth per dimension; TYPE3 covers the
  // type-3 outer spread. The two are separate fits that share the value 1.4. The width
  // law and its inverse both read it, so they cannot disagree.
  constexpr double TOLFAC_1D      = 0.18; // 1D type-1/2 base prefactor
  constexpr double TOLFAC_PER_DIM = 1.4;  // per-extra-dim worsening, ^(dim-1)
  constexpr double TOLFAC_TYPE3   = 1.4;  // type-3 outer-spread extra worsening
  return TOLFAC_1D * std::pow(TOLFAC_PER_DIM, dim - 1) * (type == 3 ? TOLFAC_TYPE3 : 1.0);
}

int theoretical_kernel_ns(double tol, int dim, int type, double sigma) {
  // Spread width ns that meets tol at upsampfac sigma > 1 in exact arithmetic, by the
  // aliasing law. One law for all PSWF kerformulas; no clip to the compiled range.
  const double tolfac = kernel_tolfac(dim, type);
  int ns              = (int)std::ceil(
      std::log(tolfac / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)) +
      common::NSOFF);
  return ns;
}

// Widest float ns that does not lose accuracy to catastrophic cancellation at low
// sigma: 8 plus a sigma- and dim-dependent step of 1/0.075 per unit over the knee.
static int float_max_ns_cc(double sigma, int dim) {
  constexpr double UPSAMPFAC_STEP = 1.0 / 0.075;
  return 8 + static_cast<int>(std::floor(
                 std::max(0.0, (sigma - 0.975 - 0.125 * dim) * UPSAMPFAC_STEP)));
}

int clamp_kernel_ns(int ns, double upsampfac, int dim, bool is_float) {
  using common::MAX_NSPREAD;
  const int max_ns = is_float ? MAX_NSPREAD<float> : MAX_NSPREAD<double>;
  ns               = std::max(common::MIN_NSPREAD, ns);
  ns               = std::min(ns, max_ns);
  // Float catastrophic-cancellation guard. Both sides are int, so a huge upsampfac
  // cannot overflow.
  if (is_float) ns = std::min(ns, float_max_ns_cc(upsampfac, dim));
  return ns;
}

double kernel_beta_given_ns(int ns, double sigma, int kf) {
  // Std shape param formula using ES model for cutoff, eg (4.5) in [FIN] with gamma=1.
  // For PSWF, aligns cut-off (start of aliasing) with freq (c) param.
  const double beta_cutoff = common::PI * (double)ns * (1.0 - 1.0 / (2.0 * sigma));

  if (kf == 7) {
    /* Shape param formula (designed for K-B), from Beatty et al,
      IEEE Trans Med Imaging, 2005 24(6):799-808. doi:10.1109/TMI.2005.848376
      "Rapid gridding reconstruction with a minimal oversampling ratio".
      This widens in real space, narrowing in k-space a little to exploit continued
      drop just after cutoff. We tweak Beatty's value 0.8 for ns=2 case to lower error.
    */
    double c_Beatty = (ns == 2) ? 0.5 : 0.8; // ns=2 case gives error fac 2 better for KB
    double pis      = common::PI * common::PI;
    return std::sqrt(beta_cutoff * beta_cutoff - c_Beatty / pis);

  } else if (kf == 8) {
    // Std shape param with const shift to exploit a little more tail decay,
    // in the style of Beatty (above) but without the sqrt; a const is better at high ns.
    // This is best for PSWF, within 0.1 digit.
    return beta_cutoff - 0.05; // Libin Lu 1/23/26

  } else if (kf == 9) {
    double t = beta_cutoff / common::PI;
    // Marco's LSQ fit using simple functions of t, 1/23/26.
    return ((-0.00149087 * t + 0.0218459) * t + 3.06269) * t - 0.0365245;
  }
  fprintf(stderr, "[%s] unknown spopts.kerformula=%d\n", __func__, kf);
  throw finufft::exception(FINUFFT_ERR_KERFORMULA_NOTVALID);
}

void set_kernel_shape_given_ns(finufft_spread_opts &spopts, int debug) {
  // Writes kernel shape parameter(s) (beta,...), into spopts, given previously-set
  // kernel info fields in spopts, principally: nspread, upsampfac, kerformula.
  // debug >0 causes stdout reporting.
  int kf      = spopts.kerformula;
  spopts.beta = kernel_beta_given_ns(spopts.nspread, spopts.upsampfac, kf);

  if (debug || spopts.debug) {
    const char *kname = [kf] {
      switch (kf) {
      case 7:
        return "PSWF (Beatty beta)";
      case 8:
        return "PSWF (beta shift)";
      case 9:
        return "PSWF (beta Marco)";
      default:
        return "unknown";
      }
    }();
    printf("[setup_spreadinterp]\tkerformula=%d: %s...\n", kf, kname);
  }
}

double min_sigma_for_aliasing_error(double tol, int dim, int type, int ns) {
  const double u =
      std::log(kernel_tolfac(dim, type) / tol) / ((ns - common::NSOFF) * common::PI);
  if (!(u < 1.0)) return std::numeric_limits<double>::infinity();
  if (u <= 0.0) return 1.0;
  return 1.0 / (1.0 - u * u);
}

double min_sigma_for_deconv_error(double tol, int dim, int ns, double eps_mach) {
  const double v = 2.0 * std::log(tol / (dim * eps_mach)) / (common::PI * ns);
  if (!(v > 0.0)) return std::numeric_limits<double>::infinity();
  const double q = std::min(std::sqrt(v), 1.0);
  return 1.0 / (q * (2.0 - q));
}

bool perdim_estimates_ok(double tol, int dim, int type, bool is_float,
                         const std::array<double, 3> &n,
                         const std::array<double, 3> &sigma_d,
                         const std::array<int, 3> &ns, std::array<double, 3> *sums) {
  // Per-dim terms of the uniform laws at (N_d, ns_d, sigma_d); the refusal sums them.
  constexpr double eps_float  = std::numeric_limits<float>::epsilon();
  constexpr double eps_double = std::numeric_limits<double>::epsilon();
  const double eps_mach       = is_float ? eps_float : eps_double;
  const double tolfac         = kernel_tolfac(dim, type) / dim; // per-dim prefactor share
  std::array<double, 3> s{0.0, 0.0, 0.0};
  for (int idim = 0; idim < dim; ++idim) {
    const double u = std::sqrt(1.0 - 1.0 / sigma_d[idim]); // aliasing decay rate
    s[0] += common::rounding_error(eps_mach, n[idim]);
    s[1] += tolfac * std::exp(-(ns[idim] - common::NSOFF) * common::PI * u);
    const double w = 1.0 - u; // phihat band-edge drop, deconvolution amplifies it
    s[2] += eps_mach * std::exp(0.5 * common::PI * ns[idim] * w * w);
  }
  if (sums) *sums = s;
  return s[0] <= tol && s[1] <= tol && s[2] <= tol;
}

} // namespace finufft::kernel

namespace finufft::common {

BIGINT fine_grid_len(double sigma, double n_modes, int ns) {
  return next235(std::max(BIGINT(std::ceil(sigma * n_modes)), BIGINT(2 * ns)), 2);
}

std::tuple<BIGINT, double, double> nhg_type3(double sigma, double X, double S, int ns,
                                             BIGINT max_nf) {
  const int nss = ns + 1; // since ns may be odd
  double Xs = X, Ss = S;  // tweak so X*S>=1, handling X=0 and/or S=0
  if (Xs == 0.0) {
    if (Ss == 0.0) {
      Xs = Ss = 1.0;
    } else
      Xs = 1.0 / Ss;
  } else
    Ss = std::max(Ss, 1.0 / Xs);
  double nfd = 2.0 * sigma * Ss * Xs / PI + nss;
  if (!std::isfinite(nfd)) nfd = 0.0;
  BIGINT nf = std::max((BIGINT)nfd, (BIGINT)(2 * ns)); // catch too-small / nan / +-inf
  if (nf < max_nf) nf = next235(nf, 2);                // else too big; spread will error
  return {nf, 2.0 * PI / (double)nf, (double)nf / (2.0 * sigma * Ss)};
}

double rounding_error(double eps_mach, double mode_diagonal) {
  return ROUND_FAC * eps_mach * mode_diagonal;
}

} // namespace finufft::common
