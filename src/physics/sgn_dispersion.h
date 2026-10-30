#ifndef SGN_DISPERSION_H
#define SGN_DISPERSION_H

// =============================================================================
// sgn_dispersion.h -- analytic linear dispersion relations
// =============================================================================
//
// WHAT THIS FILE IS
//   Closed-form phase speeds of small-amplitude waves of wavenumber k over a
//   flat bottom of depth H, for (a) exact linear water-wave theory (Airy),
//   (b) the dispersive model the relaxation approximates (alpha -> infinity),
//   and (c) the two branches of the hyperbolic relaxed system at finite alpha.
//   Escalante, Dumbser & Castro, JCP 394 (2019), eqs. (14)-(15).
//
//   Host-only helper: it is NOT used inside the time-stepping kernels. The
//   drivers use it to generate eigenmode initial conditions (slowCp2 gives the
//   wavenumber of a wave of prescribed period, e.g. apps/beji_battjes.cpp) and
//   the tests use it as the analytic reference (test/test_dispersion.cpp).
//
// CONVENTIONS
//   kH     dimensionless wavenumber k*H
//   gamma  dispersion parameter (3/2 SGN, 2 Sainte-Marie)
//   alpha  relaxation parameter, c = alpha sqrt(g H)
//   Every function returns the SQUARED, NORMALISED phase speed Cp^2 / (g H),
//   so the long-wave (shallow-water) limit is 1.
// =============================================================================

#include <cmath>

namespace sgn {

// Exact linear water-wave theory (Airy / Stokes first order):
//   Cp^2 / gH = tanh(kH) / (kH).   The kH = 0 limit is 1.
inline double airyCp2(double kH) {
  if (kH == 0.0) return 1.0;
  return std::tanh(kH) / kH;
}

// Dispersive model without relaxation (alpha -> infinity):
//   Cp^2 / gH = 1 / (1 + (kH)^2 / (2 gamma)).
// For gamma = 3/2 this is the SGN relation, accurate to O((kH)^4).
inline double sgnLimitCp2(double kH, double gamma) {
  return 1.0 / (1.0 + kH * kH / (2.0 * gamma));
}

// Relaxed hyperbolic system, SLOW branch: the physical water wave.
// Eq. (15) with the +sqrt root:
//   Cp^2 / gH = [ (J+mu)/2 + sqrt( ((J+mu)/2)^2 - mu ) ]^-1
//   J  = 1 + (kH)^2 / (2 gamma)       dispersive factor of the parent model
//   mu = (kH / alpha)^2 / (2 gamma)   relaxation factor; mu -> 0 as alpha -> inf,
//                                     and the slow branch tends to sgnLimitCp2.
inline double slowCp2(double kH, double gamma, double alpha) {
  const double J = 1.0 + kH * kH / (2.0 * gamma);
  const double mu = (kH / alpha) * (kH / alpha) / (2.0 * gamma);
  const double half = 0.5 * (J + mu);
  const double disc = half * half - mu;   // >= 0 for the strongly hyperbolic system
  // disc is clamped at zero only to guard against round-off at the branch point
  return 1.0 / (half + std::sqrt(disc > 0.0 ? disc : 0.0));
}

// Relaxed hyperbolic system, RAPID branch: the artificial acoustic wave of
// speed ~c introduced by the relaxation (the -sqrt root of the same relation).
// It carries no physics; it is what limits the explicit time step.
inline double rapidCp2(double kH, double gamma, double alpha) {
  const double J = 1.0 + kH * kH / (2.0 * gamma);
  const double mu = (kH / alpha) * (kH / alpha) / (2.0 * gamma);
  const double half = 0.5 * (J + mu);
  const double disc = half * half - mu;
  return 1.0 / (half - std::sqrt(disc > 0.0 ? disc : 0.0));
}

} // namespace sgn

#endif // SGN_DISPERSION_H
