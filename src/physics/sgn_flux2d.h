#ifndef SGN_FLUX2D_H
#define SGN_FLUX2D_H

// =============================================================================
// sgn_flux2d.h -- pointwise physics of the 2D hyperbolic SGN system
// =============================================================================
//
// WHAT THIS FILE IS
//   The two-dimensional counterpart of sgn_flux.h: x and y fluxes, signal speed,
//   algebraic sources, non-conservative products and the Rusanov flux, all
//   evaluated at ONE point. Tagged SGN_HD (see hd.h), so the 2D CPU solver
//   (src/solver/sgn2d.cpp), the single-GPU kernels (src/cuda/sgn2d_cuda.cu) and
//   the multi-GPU kernels (src/cuda/sgn2d_cuda_mpi.cu) all call exactly these
//   functions. Every production run of the paper goes through this file.
//
// THE MODEL
//   Escalante, Dumbser & Castro, J. Comput. Phys. 394 (2019), system (16), plus
//   optional Manning friction. Derivation: paper Appendix A.
//
//   d_t U + d_x F(U) + d_y G(U) + Bx(U) d_x U + By(U) d_y U = S(U):
//
//   d_t h    + d_x(hu)                 + d_y(hv)                 = 0
//   d_t(hu)  + d_x(hu²+½g(h²-H²)+hp)   + d_y(huv)                = (g(h-H)+γp)Hx
//   d_t(hv)  + d_x(huv)                + d_y(hv²+½g(h²-H²)+hp)   = (g(h-H)+γp)Hy
//   d_t(hw)  + d_x(uhw)                + d_y(vhw)                = γp
//   d_t(hp)  + d_x(hu(p+c²))           + d_y(hv(p+c²))           = c²(u h_x + v h_y)
//                                                                  - 2c²(w+uHx+vHy)
//   The c²(u h_x + v h_y) piece is the non-conservative product, returned by
//   ncpX/ncpY and SUBTRACTED by the solver; the rest of the last line is the
//   algebraic source.
//
// VARIABLES (conserved state, variable-major in a field)
//   U[H]  = h    water depth                                   [m]
//   U[HU] = hu   x-momentum (x-discharge)                      [m^2/s]
//   U[HV] = hv   y-momentum (y-discharge)                      [m^2/s]
//   U[HW] = hw   depth-integrated vertical velocity            [m^2/s]
//   U[HP] = hp   depth-integrated non-hydrostatic pressure     [m^3/s^2]
//   H(x,y) still-water depth (prescribed); eta = h - H surface elevation;
//   Hx, Hy its gradients, precomputed by the solver.
//
//   This is the direct tensor generalisation of the 1D file: the well-balanced
//   pre-split g(h²-H²)/2 appears in the diagonal momentum flux of each direction
//   and the bathymetry gradients enter as sources, so a 2D lake at rest also has
//   an exactly zero right-hand side.
// =============================================================================

#include "hd.h"
#include <cmath>

namespace sgn2d {

// Number of conserved variables in 2D, and their index in a state array.
static constexpr int NVAR = 5;
enum { H = 0, HU = 1, HV = 2, HW = 3, HP = 4 };

// Physical constants of a run (same meaning as sgn::Params in 1D).
struct Params {
  double g = 9.81;       // gravitational acceleration [m/s^2]
  double c = 0.0;        // relaxation celerity c = alpha sqrt(g H0) [m/s]; 0 = shallow water
  double gamma = 1.5;    // dispersion parameter: 3/2 SGN, 2 Sainte-Marie
  double nManning = 0.0; // Manning bottom-friction coefficient (0 = off)
};

// (h, u, v, w, p) -> conserved state. Used for initial conditions and targets.
SGN_HD inline void fromPrimitive(double h, double u, double v, double w,
                                 double p, double U[NVAR]) {
  U[H] = h; U[HU] = h * u; U[HV] = h * v; U[HW] = h * w; U[HP] = h * p;
}

// Signal speed Ce = sqrt(g h + p + c^2). Isotropic: the same in x and y. Along a
// direction with normal velocity u_n the characteristic speeds are u_n (repeated)
// and u_n -/+ Ce. Radicand clamped at zero against transient negative p.
SGN_HD inline double signalSpeed(const double U[NVAR], const Params& par) {
  const double h = U[H];
  const double p = U[HP] / h;
  const double arg = par.g * h + p + par.c * par.c;
  return std::sqrt(arg > 0.0 ? arg : 0.0);
}

// Spectral radius along one direction: |normal velocity| + Ce.
//   dir = 0 -> x (uses u = hu/h),  dir = 1 -> y (uses v = hv/h)
SGN_HD inline double maxWaveSpeed(const double U[NVAR], int dir,
                                  const Params& par) {
  const double vn = (dir == 0 ? U[HU] : U[HV]) / U[H];
  return std::fabs(vn) + signalSpeed(U, par);
}

// x-direction conservative flux F(U; H) at one point.
SGN_HD inline void fluxX(const double U[NVAR], double Hloc, const Params& par,
                         double F[NVAR]) {
  // primitive variables
  const double h = U[H];
  const double hu = U[HU];
  const double u = hu / h, v = U[HV] / h, w = U[HW] / h, p = U[HP] / h;
  const double c2 = par.c * par.c;
  F[H]  = hu;                                                   // mass
  F[HU] = hu * u + 0.5 * par.g * (h * h - Hloc * Hloc) + h * p; // x-momentum, pre-balanced
  F[HV] = hu * v;                                               // y-momentum advected in x
  F[HW] = hu * w;                                               // hw advected in x
  F[HP] = hu * (p + c2);                                        // hp advected + c^2 relaxation piece
}

// y-direction conservative flux G(U; H) at one point (mirror of fluxX).
SGN_HD inline void fluxY(const double U[NVAR], double Hloc, const Params& par,
                         double G[NVAR]) {
  // primitive variables
  const double h = U[H];
  const double hv = U[HV];
  const double u = U[HU] / h, v = hv / h, w = U[HW] / h, p = U[HP] / h;
  const double c2 = par.c * par.c;
  G[H]  = hv;                                                   // mass
  G[HU] = hv * u;                                               // x-momentum advected in y
  G[HV] = hv * v + 0.5 * par.g * (h * h - Hloc * Hloc) + h * p; // y-momentum, pre-balanced
  G[HW] = hv * w;                                               // hw advected in y
  G[HP] = hv * (p + c2);                                        // hp advected + c^2 relaxation piece
}

// Algebraic source S(U; H, Hx, Hy) at one point: bathymetry forcing in both
// directions, the gamma p drive of hw, the algebraic relaxation term of hp and
// optional Manning friction. The friction branch is skipped when n = 0, so the
// frictionless path stays bitwise identical.
SGN_HD inline void source(const double U[NVAR], double Hloc, double Hx,
                          double Hy, const Params& par, double S[NVAR]) {
  // primitive variables
  const double h = U[H];
  const double u = U[HU] / h, v = U[HV] / h, w = U[HW] / h, p = U[HP] / h;
  const double c2 = par.c * par.c;
  // common factor of the bathymetry forcing, zero at rest (h = H, p = 0)
  const double bath = par.g * (h - Hloc) + par.gamma * p;
  S[H]  = 0.0;                              // mass is conserved
  S[HU] = bath * Hx;                        // bathymetry forcing on x-momentum
  S[HV] = bath * Hy;                        // bathymetry forcing on y-momentum
  S[HW] = par.gamma * p;                    // pressure drives w
  S[HP] = -2.0 * c2 * (w + u * Hx + v * Hy); // algebraic relaxation term
  if (par.nManning != 0.0) {
    // tau_b = n^2 g |vel| vel / h^{1/3}, applied along the velocity vector
    const double n2 = par.nManning * par.nManning;
    const double vmag = std::sqrt(u * u + v * v);
    const double cf = n2 * par.g * vmag / std::cbrt(h);
    S[HU] -= cf * u;
    S[HV] -= cf * v;
  }
}

// Non-conservative products, to be SUBTRACTED by the solver. Only the hp
// equation carries them:
//   Bx d_xU = (0,0,0,0, -c^2 u h_x),   By d_yU = (0,0,0,0, -c^2 v h_y).
//   hx / hy are the depth gradients computed by the solver with the central
//   interface state h* = (h- + h+)/2 (path-consistent treatment).
SGN_HD inline void ncpX(const double U[NVAR], double hx, const Params& par,
                        double B[NVAR]) {
  const double u = U[HU] / U[H];
  B[H] = 0.0; B[HU] = 0.0; B[HV] = 0.0; B[HW] = 0.0;
  B[HP] = -par.c * par.c * u * hx;
}
SGN_HD inline void ncpY(const double U[NVAR], double hy, const Params& par,
                        double B[NVAR]) {
  const double v = U[HV] / U[H];
  B[H] = 0.0; B[HU] = 0.0; B[HV] = 0.0; B[HW] = 0.0;
  B[HP] = -par.c * par.c * v * hy;
}

// Rusanov (local Lax-Friedrichs) flux through a face.
//   UL, UR  states on the two sides (L = lower index side: left for x faces,
//           bottom for y faces)
//   dir     0 = x face (uses fluxX), 1 = y face (uses fluxY)
//   Hface   still depth at the face, single-valued because GLL faces are nodes
//   Fn      output: (F(UL)+F(UR))/2 - lam/2 (UR-UL), lam = max spectral radius
SGN_HD inline void rusanov(const double UL[NVAR], const double UR[NVAR],
                           int dir, double Hface, const Params& par,
                           double Fn[NVAR]) {
  double FL[NVAR], FR[NVAR];
  if (dir == 0) { fluxX(UL, Hface, par, FL); fluxX(UR, Hface, par, FR); }
  else          { fluxY(UL, Hface, par, FL); fluxY(UR, Hface, par, FR); }
  const double lam = std::fmax(maxWaveSpeed(UL, dir, par),
                               maxWaveSpeed(UR, dir, par));
  for (int k = 0; k < NVAR; ++k)
    Fn[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * lam * (UR[k] - UL[k]);
}

// Mechanical energy density, used as a run diagnostic:
//   E = h/2 ( u²+v² + w²/γ + g(h-2H) )
// The last term is the potential energy g(eta²-H²)/2 with eta = h - H, which
// differs from g eta²/2 only by a constant in time.
SGN_HD inline double energyDensity(const double U[NVAR], double Hloc,
                                   const Params& par) {
  const double h = U[H];
  const double u = U[HU] / h, v = U[HV] / h, w = U[HW] / h;
  return 0.5 * h * (u * u + v * v + (1.0 / par.gamma) * w * w +
                    par.g * (h - 2.0 * Hloc));
}

} // namespace sgn2d

#endif // SGN_FLUX2D_H
