// =============================================================================
// test_sgn1d.cpp -- 1D CPU solver (src/solver/sgn1d.*)
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Tests for the 1D hyperbolic-SGN FR driver (src/solver/sgn1d.*).
//   1) Lake at rest: h=const, u=w=p=0 -> RHS == 0 to machine precision.
//   2) Freestream: h=const, u=const, w=p=0 -> RHS == 0 to machine precision
//      (checks the c^2-flux / NCP split does not break constant states).
//   3) RHS consistency: smooth periodic state, check the hp equation RHS
//      against the analytic  -(d_x(hup) + h c^2 (u_x + 2w/h))  at P high enough
//      that interpolation error ~ round-off is NOT expected -> convergence
//      under element refinement at order ~P+1 instead.
//   4) Conservation: integrate a smooth wave with adaptive RK; mass and
//      momentum conserved to machine precision (periodic, flat bottom).
//   5) Robustness: the run is NaN-free and the adaptive controller finishes.
//
// Build: part of `make run-test` (links sgn1d, rk, fr_operators).

#include "../src/solver/sgn1d.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Test harness: check() prints "ok" or "FAIL", the description and the
// measured value, and counts failures; main() returns non-zero if any failed.
static int g_failures = 0;
static void check(bool ok, const char* what, double v) {
  std::printf("  %s: %-56s  %.3e\n", ok ? "ok  " : "FAIL", what, v);
  if (!ok) ++g_failures;
}

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  std::printf("=== SGN 1D FR driver (flat bottom, periodic) ===\n\n");

  sgn::Params par;
  par.g = 9.81;
  par.gamma = 1.5;
  const double H0 = 1.0;
  par.c = 3.0 * std::sqrt(par.g * H0); // alpha = 3

  const double L = 10.0;

  // 1) Lake at rest.
  {
    sgn::SGN1D sys(4, 8, L, par);
    const int N = sys.nDof();
    std::vector<double> h(N, H0), u(N, 0.0), w(N, 0.0), p(N, 0.0);
    auto U = sys.makeState(h, u, w, p);
    sgn::State R;
    sys.rhs(U, R);
    double err = 0.0;
    for (double r : R) err = std::fmax(err, std::fabs(r));
    check(err < 1e-12, "lake at rest: max|RHS| == 0", err);
  }

  // 2) Freestream preservation.
  {
    sgn::SGN1D sys(4, 8, L, par);
    const int N = sys.nDof();
    std::vector<double> h(N, H0), u(N, 0.7), w(N, 0.0), p(N, 0.0);
    auto U = sys.makeState(h, u, w, p);
    sgn::State R;
    sys.rhs(U, R);
    double err = 0.0;
    for (double r : R) err = std::fmax(err, std::fabs(r));
    check(err < 1e-11, "freestream (u=0.7): max|RHS| == 0", err);
  }

  // 3) Spatial convergence of the RHS on a smooth manufactured state.
  //    h = H0 + 0.1 sin(kx), u = 0.2 cos(kx), w = 0.05 sin(kx),
  //    p = 0.03 cos(kx), k = 2*pi/L. Compare rhs() against the analytic
  //    RHS of the PDE evaluated at the nodes; error -> 0 at order ~P+1.
  {
    const double k = 2.0 * kPi / L;
    const double ah = 0.1, au = 0.2, aw = 0.05, ap = 0.03;
    const int P = 3;
    double errPrev = 0.0;
    double order = 0.0;
    for (int r = 0; r < 2; ++r) {
      const int ne = 16 << r;
      sgn::SGN1D sys(P, ne, L, par);
      const int N = sys.nDof();
      const auto x = sys.xCoords();
      std::vector<double> h(N), u(N), w(N), p(N);
      for (int i = 0; i < N; ++i) {
        h[i] = H0 + ah * std::sin(k * x[i]);
        u[i] = au * std::cos(k * x[i]);
        w[i] = aw * std::sin(k * x[i]);
        p[i] = ap * std::cos(k * x[i]);
      }
      auto U = sys.makeState(h, u, w, p);
      sgn::State R;
      sys.rhs(U, R);

      // Analytic RHS at each node (flat bottom):
      //   R_h  = -d_x(hu)
      //   R_hu = -d_x(hu^2 + g h^2/2 + hp)
      //   R_hw = -d_x(huw) + gamma p
      //   R_hp = -d_x(hup) - h c^2 (u_x + 2 w / h)
      double err = 0.0;
      const double c2 = par.c * par.c;
      for (int i = 0; i < N; ++i) {
        const double s = std::sin(k * x[i]), cx = std::cos(k * x[i]);
        const double hh = h[i], uu = u[i], ww = w[i], pp = p[i];
        const double hx = ah * k * cx;
        const double ux = -au * k * s;
        const double wx = aw * k * cx;
        const double px = -ap * k * s;

        const double Rh  = -(hx * uu + hh * ux);
        const double Rhu = -(hx * uu * uu + 2.0 * hh * uu * ux +
                             par.g * hh * hx + hx * pp + hh * px);
        const double Rhw = -(hx * uu * ww + hh * ux * ww + hh * uu * wx) +
                           par.gamma * pp;
        const double Rhp = -(hx * uu * pp + hh * ux * pp + hh * uu * px) -
                           hh * c2 * (ux + 2.0 * ww / hh);

        err = std::fmax(err, std::fabs(R[0 * N + i] - Rh));
        err = std::fmax(err, std::fabs(R[1 * N + i] - Rhu));
        err = std::fmax(err, std::fabs(R[2 * N + i] - Rhw));
        err = std::fmax(err, std::fabs(R[3 * N + i] - Rhp));
      }
      if (r > 0) order = std::log2(errPrev / err);
      errPrev = err;
    }
    // P=3 -> expect ~P+1 = 4 (allow margin: interface terms can shave a bit).
    check(order > 3.0, "manufactured RHS converges (P=3, order>3)", order);
  }

  // 4+5) Conservation + robustness on an evolving smooth wave.
  {
    const int P = 4, ne = 24;
    sgn::SGN1D sys(P, ne, L, par);
    const int N = sys.nDof();
    const auto x = sys.xCoords();
    const double k = 2.0 * kPi / L;
    std::vector<double> h(N), u(N), w(N), p(N, 0.0);
    for (int i = 0; i < N; ++i) {
      h[i] = H0 + 0.05 * std::sin(k * x[i]);
      u[i] = 0.0;
      w[i] = 0.0;
    }
    auto U = sys.makeState(h, u, w, p);

    const double M0 = sys.mass(U);
    const double Mom0 = sys.momentum(U);

    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-9; opt.rtol = 1e-7;
    const double tEnd = 0.5; // a few wave crossings of the fast branch
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, tEnd,
                                           sys.cflTimeStep(U, 0.5), opt);

    bool finite = true;
    for (double v : U) if (!std::isfinite(v)) { finite = false; break; }

    const double dM = std::fabs(sys.mass(U) - M0) / std::fabs(M0);
    const double dMom = std::fabs(sys.momentum(U) - Mom0); // Mom0 == 0

    check(!st.aborted && st.tReached >= tEnd - 1e-12,
          "smooth wave: adaptive run finishes", st.tReached);
    check(finite, "smooth wave: state stays finite", finite ? 0.0 : 1.0);
    check(dM < 1e-13, "mass conserved to machine precision", dM);
    check(dMom < 1e-11, "momentum conserved (starts at 0)", dMom);
    std::printf("       [info] steps accepted=%ld rejected=%ld dtLast=%.3e\n",
                st.accepted, st.rejected, st.dtLast);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
