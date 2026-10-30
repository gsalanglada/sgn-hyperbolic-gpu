// =============================================================================
// test_sgn_flux.cpp -- unit tests of the 1D pointwise physics (src/physics/sgn_flux.h)
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Tests for the hyperbolic-SGN physics (src/physics/sgn_flux.h).
//   1) Flux components vs hand-computed values (pre-balanced form, H != 0).
//   2) Signal speed Ce = sqrt(g h + p + c^2) and maxWaveSpeed = |u| + Ce.
//   3) Shallow-water limit: c=0, w=p=0, H=0 -> SWE flux, zero source/NCP.
//   4) Rusanov single-state consistency: rusanov(U,U,H) == fluxX(U,H).
//   5) Algebraic source with bathymetry: S = ( 0, (g(h-H)+gamma p) Hx,
//      gamma p, -2 c^2 (w + u Hx) ); flat (Hx=0) reduces to the old form.
//   6) Non-conservative product B d_xU = (0, 0, 0, -c^2 u h_x).
//   7) Well-balanced rest state: h=H, u=w=p=0 -> flux and source EXACTLY zero.
//
// Build: part of `make run-test`.

#include "../src/physics/sgn_flux.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// Test harness: check() prints "ok" or "FAIL", the description and the
// measured value, and counts failures; main() returns non-zero if any failed.
static int g_failures = 0;
static void check(bool ok, const char* what, double v) {
  std::printf("  %s: %-50s  %.3e\n", ok ? "ok  " : "FAIL", what, v);
  if (!ok) ++g_failures;
}

int main() {
  std::printf("=== hyperbolic-SGN physics ===\n\n");

  sgn::Params par;
  par.g = 9.81;
  par.gamma = 1.5;
  par.c = 3.0 * std::sqrt(par.g * 1.0); // alpha=3, H0=1

  // Generic state and bathymetry values.
  const double h = 1.3, u = 0.4, w = -0.2, p = 0.15;
  const double Hb = 0.9, Hx = 0.7;
  double U[sgn::NVAR];
  sgn::fromPrimitive(h, u, w, p, U);

  // 1) Conservative flux vs hand values (pre-balanced hydrostatic term).
  {
    const double c2 = par.c * par.c;
    double F[sgn::NVAR];
    sgn::fluxX(U, Hb, par, F);
    double err = 0.0;
    err = std::max(err, std::fabs(F[sgn::H]  - h * u));
    err = std::max(err, std::fabs(F[sgn::HU] -
        (h * u * u + 0.5 * par.g * (h * h - Hb * Hb) + h * p)));
    err = std::max(err, std::fabs(F[sgn::HW] - (h * u * w)));
    err = std::max(err, std::fabs(F[sgn::HP] - (h * u * (p + c2))));
    check(err < 1e-13, "F(U;H) matches hand-computed flux", err);
  }

  // 2) Signal speed and max wave speed.
  {
    const double Ce = std::sqrt(par.g * h + p + par.c * par.c);
    double err = 0.0;
    err = std::max(err, std::fabs(sgn::signalSpeed(U, par) - Ce));
    err = std::max(err, std::fabs(sgn::maxWaveSpeed(U, par) - (std::fabs(u) + Ce)));
    check(err < 1e-13, "Ce = sqrt(g h + p + c^2), lam = |u| + Ce", err);
  }

  // 3) Shallow-water limit: c=0, w=p=0, H=0 -> SWE flux, zero source & NCP.
  {
    sgn::Params sw;
    sw.g = par.g; sw.gamma = par.gamma; sw.c = 0.0;
    double Usw[sgn::NVAR];
    sgn::fromPrimitive(h, u, 0.0, 0.0, Usw);
    double F[sgn::NVAR], S[sgn::NVAR], B[sgn::NVAR];
    sgn::fluxX(Usw, 0.0, sw, F);
    sgn::source(Usw, 0.0, 0.0, sw, S);
    sgn::ncpX(Usw, /*hx=*/0.7, sw, B);
    double err = 0.0;
    err = std::max(err, std::fabs(F[sgn::H]  - h * u));
    err = std::max(err, std::fabs(F[sgn::HU] - (h * u * u + 0.5 * sw.g * h * h)));
    err = std::max(err, std::fabs(F[sgn::HW]));
    err = std::max(err, std::fabs(F[sgn::HP]));
    for (int k = 0; k < sgn::NVAR; ++k) {
      err = std::max(err, std::fabs(S[k]));
      err = std::max(err, std::fabs(B[k]));
    }
    check(err < 1e-13, "SWE limit (c=0,w=p=0,H=0): SWE flux, no src", err);
  }

  // 4) Rusanov single-state consistency.
  {
    double F[sgn::NVAR], Fn[sgn::NVAR];
    sgn::fluxX(U, Hb, par, F);
    sgn::rusanov(U, U, Hb, par, Fn);
    double err = 0.0;
    for (int k = 0; k < sgn::NVAR; ++k) err = std::max(err, std::fabs(Fn[k] - F[k]));
    check(err < 1e-13, "rusanov(U,U,H) == fluxX(U,H) (consistency)", err);
  }

  // 5) Algebraic source, with and without bathymetry gradient.
  {
    double S[sgn::NVAR];
    sgn::source(U, Hb, Hx, par, S);
    const double c2 = par.c * par.c;
    double err = 0.0;
    err = std::max(err, std::fabs(S[sgn::H]));
    err = std::max(err, std::fabs(S[sgn::HU] -
        (par.g * (h - Hb) + par.gamma * p) * Hx));
    err = std::max(err, std::fabs(S[sgn::HW] - par.gamma * p));
    err = std::max(err, std::fabs(S[sgn::HP] - (-2.0 * c2 * (w + u * Hx))));
    check(err < 1e-13, "S with bathymetry matches hand values", err);

    sgn::source(U, Hb, 0.0, par, S);
    err = 0.0;
    err = std::max(err, std::fabs(S[sgn::HU]));
    err = std::max(err, std::fabs(S[sgn::HW] - par.gamma * p));
    err = std::max(err, std::fabs(S[sgn::HP] - (-2.0 * c2 * w)));
    check(err < 1e-13, "S flat (Hx=0) = (0, 0, gamma*p, -2 c^2 w)", err);
  }

  // 6) Non-conservative product.
  {
    const double hx = 0.7;
    double B[sgn::NVAR];
    sgn::ncpX(U, hx, par, B);
    const double c2 = par.c * par.c;
    double err = 0.0;
    err = std::max(err, std::fabs(B[sgn::H]));
    err = std::max(err, std::fabs(B[sgn::HU]));
    err = std::max(err, std::fabs(B[sgn::HW]));
    err = std::max(err, std::fabs(B[sgn::HP] - (-c2 * u * hx)));
    check(err < 1e-13, "B d_xU = (0, 0, 0, -c^2 u h_x)", err);
  }

  // 7) Well-balanced rest state: EXACT zeros (not just small).
  {
    double Ur[sgn::NVAR];
    sgn::fromPrimitive(Hb, 0.0, 0.0, 0.0, Ur); // h = H, at rest
    double F[sgn::NVAR], S[sgn::NVAR];
    sgn::fluxX(Ur, Hb, par, F);
    sgn::source(Ur, Hb, Hx, par, S); // arbitrary Hx: must not matter at rest
    double err = 0.0;
    for (int k = 0; k < sgn::NVAR; ++k) {
      err = std::max(err, std::fabs(F[k]));
      err = std::max(err, std::fabs(S[k]));
    }
    check(err == 0.0, "rest (h=H): flux and source EXACTLY zero", err);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
