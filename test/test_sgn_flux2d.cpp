// =============================================================================
// test_sgn_flux2d.cpp -- unit tests of the 2D pointwise physics (src/physics/sgn_flux2d.h)
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Tests for the 2D hyperbolic-SGN physics (src/physics/sgn_flux2d.h).
//   1) x-flux and y-flux vs hand-computed values (pre-balanced form).
//   2) Rotational consistency: rotating (u,v)->(v,u) and swapping x/y flux
//      components reproduces the same physics (fluxY == fluxX of the swapped
//      state). Catches x/y asymmetry bugs.
//   3) Signal speed + directional max wave speed.
//   4) Rusanov single-state consistency in x and y.
//   5) Source with bathymetry (both gradients) and NCPs.
//   6) Well-balanced rest state: h=H, u=v=w=p=0 -> both fluxes & source EXACT 0.
//   7) 1D embedding: v=0, Hy=0, d_y(.)=0 reduces the x-operators to the 1D ones
//      (checked by matching sgn_flux.h values on the shared variables).
//
// Build: part of `make run-test`.

#include "../src/physics/sgn_flux2d.h"
#include "../src/physics/sgn_flux.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// Test harness: check() prints "ok" or "FAIL", the description and the
// measured value, and counts failures; main() returns non-zero if any failed.
static int g_failures = 0;
static void check(bool ok, const char* what, double v) {
  std::printf("  %s: %-52s  %.3e\n", ok ? "ok  " : "FAIL", what, v);
  if (!ok) ++g_failures;
}

int main() {
  std::printf("=== hyperbolic-SGN 2D physics ===\n\n");

  sgn2d::Params par;
  par.g = 9.81; par.gamma = 1.5;
  par.c = 3.0 * std::sqrt(par.g * 1.0);
  const double c2 = par.c * par.c;

  const double h = 1.3, u = 0.4, v = -0.25, w = -0.2, p = 0.15;
  const double Hb = 0.9, Hx = 0.7, Hy = -0.3;
  double U[sgn2d::NVAR];
  sgn2d::fromPrimitive(h, u, v, w, p, U);

  using namespace sgn2d;

  // 1) Flux components.
  {
    double F[NVAR], G[NVAR];
    fluxX(U, Hb, par, F);
    fluxY(U, Hb, par, G);
    const double hyd = 0.5 * par.g * (h * h - Hb * Hb);
    double err = 0.0;
    // x-flux
    err = std::max(err, std::fabs(F[H]  - h * u));
    err = std::max(err, std::fabs(F[HU] - (h * u * u + hyd + h * p)));
    err = std::max(err, std::fabs(F[HV] - (h * u * v)));
    err = std::max(err, std::fabs(F[HW] - (h * u * w)));
    err = std::max(err, std::fabs(F[HP] - (h * u * (p + c2))));
    // y-flux
    err = std::max(err, std::fabs(G[H]  - h * v));
    err = std::max(err, std::fabs(G[HU] - (h * v * u)));
    err = std::max(err, std::fabs(G[HV] - (h * v * v + hyd + h * p)));
    err = std::max(err, std::fabs(G[HW] - (h * v * w)));
    err = std::max(err, std::fabs(G[HP] - (h * v * (p + c2))));
    check(err < 1e-13, "x- and y-flux match hand-computed values", err);
  }

  // 2) Rotational consistency: G(U) == swap( F(swap(U)) ).
  {
    double Uswap[NVAR];
    fromPrimitive(h, v, u, w, p, Uswap); // swap u<->v
    double Fs[NVAR], G[NVAR];
    fluxX(Uswap, Hb, par, Fs);
    fluxY(U, Hb, par, G);
    // After computing F of the (v,u) state, swapping its HU<->HV momentum
    // components must reproduce G of the original.
    double err = 0.0;
    err = std::max(err, std::fabs(G[H]  - Fs[H]));
    err = std::max(err, std::fabs(G[HU] - Fs[HV]));
    err = std::max(err, std::fabs(G[HV] - Fs[HU]));
    err = std::max(err, std::fabs(G[HW] - Fs[HW]));
    err = std::max(err, std::fabs(G[HP] - Fs[HP]));
    check(err < 1e-13, "rotational consistency G == swap(F(swap))", err);
  }

  // 3) Signal speed and directional wave speeds.
  {
    const double Ce = std::sqrt(par.g * h + p + c2);
    double err = 0.0;
    err = std::max(err, std::fabs(signalSpeed(U, par) - Ce));
    err = std::max(err, std::fabs(maxWaveSpeed(U, 0, par) - (std::fabs(u) + Ce)));
    err = std::max(err, std::fabs(maxWaveSpeed(U, 1, par) - (std::fabs(v) + Ce)));
    check(err < 1e-13, "Ce and directional |vel_n|+Ce", err);
  }

  // 4) Rusanov single-state consistency (x and y).
  {
    double F[NVAR], G[NVAR], Fn[NVAR], Gn[NVAR];
    fluxX(U, Hb, par, F); rusanov(U, U, 0, Hb, par, Fn);
    fluxY(U, Hb, par, G); rusanov(U, U, 1, Hb, par, Gn);
    double err = 0.0;
    for (int k = 0; k < NVAR; ++k) {
      err = std::max(err, std::fabs(Fn[k] - F[k]));
      err = std::max(err, std::fabs(Gn[k] - G[k]));
    }
    check(err < 1e-13, "rusanov(U,U) == flux both directions", err);
  }

  // 5) Source + NCPs.
  {
    double S[NVAR], Bx[NVAR], By[NVAR];
    source(U, Hb, Hx, Hy, par, S);
    ncpX(U, /*hx=*/0.6, par, Bx);
    ncpY(U, /*hy=*/-0.4, par, By);
    const double bath = par.g * (h - Hb) + par.gamma * p;
    double err = 0.0;
    err = std::max(err, std::fabs(S[HU] - bath * Hx));
    err = std::max(err, std::fabs(S[HV] - bath * Hy));
    err = std::max(err, std::fabs(S[HW] - par.gamma * p));
    err = std::max(err, std::fabs(S[HP] - (-2.0 * c2 * (w + u * Hx + v * Hy))));
    err = std::max(err, std::fabs(Bx[HP] - (-c2 * u * 0.6)));
    err = std::max(err, std::fabs(By[HP] - (-c2 * v * (-0.4))));
    check(err < 1e-13, "source (both Hx,Hy) and NCPs correct", err);
  }

  // 6) Well-balanced rest state.
  {
    double Ur[NVAR];
    fromPrimitive(Hb, 0.0, 0.0, 0.0, 0.0, Ur);
    double F[NVAR], G[NVAR], S[NVAR];
    fluxX(Ur, Hb, par, F);
    fluxY(Ur, Hb, par, G);
    source(Ur, Hb, Hx, Hy, par, S);
    double err = 0.0;
    for (int k = 0; k < NVAR; ++k) {
      err = std::max(err, std::fabs(F[k]));
      err = std::max(err, std::fabs(G[k]));
      err = std::max(err, std::fabs(S[k]));
    }
    check(err == 0.0, "rest (h=H): fluxes and source EXACTLY zero", err);
  }

  // 7) 1D embedding: v=0 -> x-operators equal the 1D sgn_flux.h ones on the
  //    shared variables (h, hu, hw, hp).
  {
    double U1[sgn::NVAR];
    sgn::fromPrimitive(h, u, w, p, U1);
    double U2[NVAR];
    fromPrimitive(h, u, 0.0, w, p, U2);
    sgn::Params par1; par1.g = par.g; par1.gamma = par.gamma; par1.c = par.c;

    double F1[sgn::NVAR], F2[NVAR];
    sgn::fluxX(U1, Hb, par1, F1);
    fluxX(U2, Hb, par, F2);
    double err = 0.0;
    err = std::max(err, std::fabs(F2[H]  - F1[sgn::H]));
    err = std::max(err, std::fabs(F2[HU] - F1[sgn::HU]));
    err = std::max(err, std::fabs(F2[HW] - F1[sgn::HW]));
    err = std::max(err, std::fabs(F2[HP] - F1[sgn::HP]));
    err = std::max(err, std::fabs(F2[HV])); // v=0 -> no y-momentum flux in x
    check(err < 1e-13, "1D embedding: x-flux matches sgn_flux.h", err);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
