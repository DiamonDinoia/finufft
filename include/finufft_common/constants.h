#pragma once

#include <type_traits>

namespace finufft {
namespace common {

// constants needed within common
// Smallest spread width ns.
inline constexpr int MIN_NSPREAD = 2;
// Per-precision instantiation cap on ns. Above sigma 1.7 the float error is flat for
// ns 7-16, so float ns > 12 gains nothing.
template<typename T>
inline constexpr int MAX_NSPREAD = std::is_same_v<T, float> ? 12 : 16;
// max number of positive quadr nodes
inline constexpr int MAX_NQUAD = 100;
// Fraction growth cut-off in utils:arraywidcen, sets when translate in type-3
inline constexpr double ARRAYWIDCEN_GROWFRAC = 0.1;
inline constexpr double PI                   = 3.141592653589793238462643383279502884;
// 1 / (2 * PI)
inline constexpr double INV_2PI = 0.159154943091895335768883763372514362;

// polynomial degree bounds for all kernel approximation
inline constexpr int MIN_NC = 4;
inline constexpr int MAX_NC = 19;

// Upsampfac (sigma) bounds.
// best_sigma_for_tol searches [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] for all types.
// Below 1.25 it lands on the aliasing-law boundary with no accuracy margin; above 2.0 a
// narrower kernel can still pay off for dense, spread-dominated transforms.
inline constexpr double MIN_AUTO_UPSAMPFAC       = 1.25;
inline constexpr double MAX_AUTO_UPSAMPFAC       = 2.5;
// makeplan warns on a user upsampfac outside this range, which rarely pays off.
inline constexpr double MIN_USEFUL_UPSAMPFAC     = 1.15;
inline constexpr double MAX_USEFUL_UPSAMPFAC     = 3.0;

// The per-dimension kernel retune runs only when the realized ratios nf_d / N_d
// differ between dimensions by more than this factor (max over min). Below it the
// grid rounding moved every dimension by the same factor, the uniform kernel width
// already fits each dimension, and the plan stays the uniform one. Measured cells
// set the value: the win class (17x177, 177x17 double) realizes a 1.08-1.11 spread
// where the retune narrows ns 10 to 9 and wins 4-27%; the flat or losing cells
// (4000x250 double 1.054, 400x100x25 float 1.067, every square cell 1.00) stay at or
// below 1.067, so 1.07 keeps them on the uniform kernel.
inline constexpr double PERDIM_RETUNE_MIN_RATIO  = 1.07;

// The dispatched width tuples cover per-dim widths differing by at most this many
// grid points (17x177 retunes 10 to 9, a difference of 1). A retune whose widths
// spread wider falls back to the equal-width kernel instance.
inline constexpr int PERDIM_MAX_NS_DIFF          = 1;
// Packed (dy, dz) shift key span: key = SPAN*(dy+1) + (dz+1), dy,dz in [-DIFF, DIFF].
inline constexpr int PERDIM_SHIFT_SPAN           = 2 * PERDIM_MAX_NS_DIFF + 1;

// Width offset in the kernel aliasing law tol =
// tolfac*exp(-pi*(ns-NSOFF)*sqrt(1-1/sigma)), read by theoretical_kernel_ns and its
// inverse min_sigma_for_aliasing_error.
inline constexpr double NSOFF                    = 1.0;

// Fitted prefactor of rounding_error (finufft_common/kernel.h).
inline constexpr double ROUND_FAC                = 0.35;

} // namespace common
} // namespace finufft
