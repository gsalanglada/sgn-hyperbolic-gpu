#ifndef SGN_FLUX_H
#define SGN_FLUX_H

// =============================================================================
// sgn_flux.h -- pointwise physics of the 1D hyperbolic SGN system
// =============================================================================
//
// WHAT THIS FILE IS
//   The single source of truth for every physical formula of the one-dimensional
//   model: the conservative flux, the characteristic (signal) speed, the
//   algebraic source terms, the non-conservative product and the Rusanov
//   interface flux. Nothing in here knows about meshes, elements, polynomials or
//   parallelism: every function takes the state at ONE point and returns values
//   at that point.
//
//   All functions are tagged SGN_HD (see hd.h), so the same text is compiled as
//   ordinary C++ for the CPU solver (src/solver/sgn1d.cpp) and as CUDA device
//   code for the GPU kernels (src/cuda/sgn1d_cuda.cu). That is what guarantees
//   the CPU and GPU solvers evaluate identical arithmetic.
//
// THE MODEL
//   Escalante, Dumbser & Castro, J. Comput. Phys. 394 (2019) 385-416, system (4).
//   Derivation: paper Appendix A and Escalante et al. (2019). In conservation form
//
//     d_t U + d_x F(U;H) + B(U) d_x U = S(U;H,Hx)
//
//     d_t h     + d_x( hu )                          = 0
//     d_t(hu)   + d_x( hu^2 + g(h^2-H^2)/2 + hp )    = (g(h-H) + gamma p) Hx - friction
//     d_t(hw)   + d_x( hu w )                        = gamma p
//     d_t(hp)   + d_x( hu (p + c^2) )                = c^2 u d_x h - 2 c^2 (w + u Hx)
//
// VARIABLES (conserved state U, stored "variable-major" in a field: all h, then
// all hu, ...)
//   U[H]  = h   water depth                                      [m]
//   U[HU] = hu  depth-integrated horizontal momentum (discharge) [m^2/s]
//   U[HW] = hw  depth-integrated vertical velocity               [m^2/s]
//   U[HP] = hp  depth-integrated non-hydrostatic pressure        [m^3/s^2]
//   Primitive values are recovered by dividing by h: u = hu/h, w = hw/h, p = hp/h.
//
//   H(x)  still-water depth (bathymetry), prescribed and never evolved; the bed
//         lies at z = -H and the free surface at eta = h - H.
//   Hx    d_x H at the node, precomputed once by the solver.
//
// WELL-BALANCED SPLIT
//   The hydrostatic part of the momentum flux is written as g(h^2 - H^2)/2 and
//   the bathymetry term as g(h - H) Hx. This is algebraically identical to the
//   textbook form g h^2/2 with source g h Hx, but at rest (h = H, u = w = p = 0)
//   every nodal flux and source value is EXACTLY 0.0 in floating point, so a lake
//   at rest over any smooth H stays at rest to machine precision without any
//   hydrostatic reconstruction.
// =============================================================================

#include "hd.h"
#include <cmath>

namespace sgn {

// Number of conserved variables in 1D, and their index in a state array.
static constexpr int NVAR = 4;
enum { H = 0, HU = 1, HW = 2, HP = 3 };

// -----------------------------------------------------------------------------
// Params -- physical constants of a run
// -----------------------------------------------------------------------------
struct Params {
  double g = 9.81;       // gravitational acceleration [m/s^2]
  double c = 0.0;        // artificial relaxation celerity c = alpha sqrt(g H0) [m/s].
                         // c -> infinity recovers the dispersive (SGN / Sainte-Marie)
                         // model; c = 0 with w = p = 0 is plain shallow water.
  double gamma = 1.5;    // dispersion parameter: 3/2 = Serre-Green-Naghdi,
                         // 2 = Sainte-Marie (wider range of dispersion accuracy)
  double nManning = 0.0; // Manning bottom-friction coefficient [s m^-1/3]; 0 = off
};

// -----------------------------------------------------------------------------
// Primitive recovery: conserved state -> physical quantities at one point.
// All divide by h, so they assume a wet point (h > 0).
// -----------------------------------------------------------------------------
// water depth h
SGN_HD inline double depth(const double U[NVAR]) { return U[H]; }
// horizontal velocity u = hu / h
SGN_HD inline double velU(const double U[NVAR]) { return U[HU] / U[H]; }
// vertical velocity w = hw / h
SGN_HD inline double velW(const double U[NVAR]) { return U[HW] / U[H]; }
// non-hydrostatic pressure p = hp / h
SGN_HD inline double pres(const double U[NVAR]) { return U[HP] / U[H]; }

// -----------------------------------------------------------------------------
// fromPrimitive -- the inverse: (h, u, w, p) -> conserved state U.
// Used to build initial conditions and relaxation-zone targets.
// -----------------------------------------------------------------------------
SGN_HD inline void fromPrimitive(double h, double u, double w, double p,
                                 double U[NVAR]) {
  U[H] = h;             // depth
  U[HU] = h * u;        // horizontal momentum
  U[HW] = h * w;        // depth-integrated vertical velocity
  U[HP] = h * p;        // depth-integrated non-hydrostatic pressure
}

// -----------------------------------------------------------------------------
// signalSpeed -- Ce = sqrt(g h + p + c^2).
// The eigenvalues of the flux Jacobian are u, u, u - Ce, u + Ce: information
// travels at speed Ce relative to the flow. The c^2 term is the fast artificial
// acoustic branch introduced by the relaxation, and it is what sets the explicit
// time-step limit.
// -----------------------------------------------------------------------------
SGN_HD inline double signalSpeed(const double U[NVAR], const Params& par) {
  const double h = U[H];
  const double p = U[HP] / h;

  // radicand of the eigenvalue
  const double arg = par.g * h + p + par.c * par.c;

  // A strongly negative transient pressure could make the radicand negative;
  // clamp at zero so sqrt never returns NaN.
  return std::sqrt(arg > 0.0 ? arg : 0.0);
}

// -----------------------------------------------------------------------------
// maxWaveSpeed -- spectral radius |u| + Ce of the flux Jacobian at one point.
// Used for the Rusanov dissipation coefficient and for the CFL time step.
// -----------------------------------------------------------------------------
SGN_HD inline double maxWaveSpeed(const double U[NVAR], const Params& par) {
  return std::fabs(U[HU] / U[H]) + signalSpeed(U, par);
}

// -----------------------------------------------------------------------------
// fluxX -- conservative flux F(U; H) at one point.
//   U    conserved state at the point
//   Hloc still-water depth at the same point (needed by the well-balanced split)
//   F    output flux, one entry per conserved variable
// -----------------------------------------------------------------------------
SGN_HD inline void fluxX(const double U[NVAR], double Hloc, const Params& par,
                         double F[NVAR]) {
  // Step 1: primitive variables
  const double h = U[H];
  const double hu = U[HU];
  const double u = hu / h;
  const double w = U[HW] / h;
  const double p = U[HP] / h;
  const double c2 = par.c * par.c;

  // Step 2: flux of each conserved variable
  F[H]  = hu;                   // mass flux

  // momentum flux: advection + pre-balanced hydrostatic pressure + non-hydrostatic
  // pressure. The -H^2 term makes this exactly zero at rest (see file header).
  F[HU] = hu * u + 0.5 * par.g * (h * h - Hloc * Hloc) + h * p;

  F[HW] = hu * w;               // advection of hw

  // advection of hp plus the c^2 u h piece of the relaxation term; together with
  // the non-conservative product (ncpX) this produces h c^2 d_x u.
  F[HP] = hu * (p + c2);
}

// -----------------------------------------------------------------------------
// source -- algebraic (pointwise) source S(U; H, Hx).
//   Hloc still-water depth at the point, Hx its gradient d_x H.
// -----------------------------------------------------------------------------
SGN_HD inline void source(const double U[NVAR], double Hloc, double Hx,
                          const Params& par, double S[NVAR]) {
  // Step 1: primitive variables
  const double h = U[H];
  const double u = U[HU] / h;
  const double w = U[HW] / h;
  const double p = U[HP] / h;

  // Step 2: sources of each equation
  S[H]  = 0.0;                                       // mass is conserved

  // bathymetry forcing on the momentum, in pre-balanced form (zero at rest)
  S[HU] = (par.g * (h - Hloc) + par.gamma * p) * Hx;

  S[HW] = par.gamma * p;                             // pressure drives w

  // algebraic part of the relaxation term: couples hp to w and to the
  // bed-following vertical velocity u Hx
  S[HP] = -2.0 * par.c * par.c * (w + u * Hx);

  // Step 3: optional Manning bottom friction, tau_b = n^2 g u|u| / h^{1/3}.
  // The branch is skipped entirely when n = 0, so the frictionless path is
  // bitwise identical to a build without friction.
  if (par.nManning != 0.0) {
    const double n2 = par.nManning * par.nManning;
    S[HU] -= n2 * par.g * u * std::fabs(u) / std::cbrt(h);
  }
}

// -----------------------------------------------------------------------------
// ncpX -- non-conservative product B(U) d_x U at one point.
//   hx  the depth gradient d_x h at the point, computed by the solver with the
//       FR corrected derivative and the CENTRAL interface value h* = (h- + h+)/2.
//   B   output; the solver SUBTRACTS it from the right-hand side.
// Only the hp equation carries a non-conservative term, -c^2 u d_x h. Combined
// with the c^2 u h in F[HP]:  d_x(c^2 h u) - c^2 u d_x h = c^2 h d_x u.
// -----------------------------------------------------------------------------
SGN_HD inline void ncpX(const double U[NVAR], double hx, const Params& par,
                        double B[NVAR]) {
  const double u = U[HU] / U[H];
  B[H]  = 0.0;
  B[HU] = 0.0;
  B[HW] = 0.0;
  B[HP] = -par.c * par.c * u * hx;
}

// -----------------------------------------------------------------------------
// rusanov -- numerical flux through an element interface (local Lax-Friedrichs).
//   UL, UR  states on the left and right side of the face
//   Hface   still-water depth at the face; with Gauss-Lobatto nodes the face is
//           a node of both elements, so this value is single-valued
//   Fn      output interface flux
//
//   Fn = (F(UL) + F(UR))/2 - lam/2 (UR - UL),  lam = max(|u|+Ce) over both sides
// The first term is the central average; the second is upwind dissipation
// proportional to the jump, which stabilises the scheme.
// -----------------------------------------------------------------------------
SGN_HD inline void rusanov(const double UL[NVAR], const double UR[NVAR],
                           double Hface, const Params& par, double Fn[NVAR]) {
  double FL[NVAR], FR[NVAR];

  // Step 1: physical flux evaluated with each side's state
  fluxX(UL, Hface, par, FL);
  fluxX(UR, Hface, par, FR);

  // Step 2: largest characteristic speed on either side
  const double lam =
      std::fmax(maxWaveSpeed(UL, par), maxWaveSpeed(UR, par));

  // Step 3: average minus jump dissipation, variable by variable
  for (int k = 0; k < NVAR; ++k)
    Fn[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * lam * (UR[k] - UL[k]);
}

// -----------------------------------------------------------------------------
// energyDensity -- mechanical energy per unit length at one point, used only as
// a diagnostic (the solver integrates it over the domain to monitor drift).
//   E = h/2 ( u^2 + w^2/gamma + g (h - 2H) )
// The last term is the column's potential energy g(eta^2 - H^2)/2 written with
// eta = h - H; it differs from the usual g eta^2/2 only by a constant in time.
// -----------------------------------------------------------------------------
SGN_HD inline double energyDensity(const double U[NVAR], double Hloc,
                                   const Params& par) {
  const double h = U[H];
  const double u = U[HU] / h;
  const double w = U[HW] / h;

  return 0.5 * h *
         (u * u + (1.0 / par.gamma) * w * w + par.g * (h - 2.0 * Hloc));
}

} // namespace sgn

#endif // SGN_FLUX_H
