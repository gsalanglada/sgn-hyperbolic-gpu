// =============================================================================
// test_sgn2d.cpp -- 2D CPU solver (src/solver/sgn2d.*)
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Tests for the 2D tensor-product SGN FR driver (src/solver/sgn2d.*).
//   1) Lake at rest over a 2D bump: max|RHS| == 0 (well-balancing).
//   2) Freestream (h,u,v const; w=p=0): max|RHS| == 0.
//   3) x-embedding: a state constant in y with v=0 reproduces the validated 1D
//      driver's RHS BITWISE on (h,hu,hw,hp); the hv RHS is ~0.
//   4) y-embedding: constant in x with u=0 reproduces the 1D driver along y.
//   5) Mass conservation over a short 2D run (bump wave), state finite.
//
// The embedding gates tie the whole 2D machinery back to the bitwise-validated
// 1D solver, so a bug in the tensor coupling shows up immediately.
//
// Build: part of `make run-test`.

#include "../src/solver/sgn2d.h"
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
  std::printf("=== SGN 2D tensor-product FR driver ===\n\n");

  const double g = 9.81, H0 = 1.0, gamma = 1.5, alpha = 3.0;
  sgn2d::Params par; par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);
  sgn::Params par1; par1.g = g; par1.gamma = gamma; par1.c = par.c;

  const double Lx = 10.0, Ly = 8.0;
  const int P = 4, Nx = 8, Ny = 6;

  auto bump2d = [Lx, Ly](double x, double y) {
    const double dx = x - 0.5 * Lx, dy = y - 0.5 * Ly;
    return 1.0 - 0.4 * std::exp(-(dx * dx + dy * dy) / 6.0);
  };

  // 1) Lake at rest over 2D bump.
  {
    sgn::SGN2D sys(P, Nx, Ny, Lx, Ly, par, bump2d);
    const int Nd = sys.nDof();
    std::vector<double> x, y; sys.nodeCoords(x, y);
    std::vector<double> h(Nd), z(Nd, 0.0);
    for (int i = 0; i < Nd; ++i) h[i] = bump2d(x[i], y[i]);
    auto U = sys.makeState(h, z, z, z, z);
    sgn::State R; sys.rhs(U, R);
    double err = 0.0; for (double r : R) err = std::fmax(err, std::fabs(r));
    check(err < 1e-12, "lake at rest over 2D bump: max|RHS| == 0", err);
  }

  // 2) Freestream.
  {
    sgn::SGN2D sys(P, Nx, Ny, Lx, Ly, par);
    const int Nd = sys.nDof();
    std::vector<double> h(Nd, H0), uu(Nd, 0.5), vv(Nd, -0.3), z(Nd, 0.0);
    auto U = sys.makeState(h, uu, vv, z, z);
    sgn::State R; sys.rhs(U, R);
    double err = 0.0; for (double r : R) err = std::fmax(err, std::fabs(r));
    check(err < 1e-11, "freestream (u,v const): max|RHS| == 0", err);
  }

  // 3) x-embedding vs 1D driver (flat bottom).
  {
    sgn::SGN2D s2(P, Nx, Ny, Lx, Ly, par);          // flat
    sgn::SGN1D s1(P, Nx, Lx, par1);                 // flat, same x-mesh
    const int Nd2 = s2.nDof(), Nd1 = s1.nDof(), np = P + 1;
    std::vector<double> x2, y2; s2.nodeCoords(x2, y2);
    const double k = 2.0 * kPi / Lx;
    auto fh = [&](double x){ return H0 + 0.08 * std::sin(k * x); };
    auto fu = [&](double x){ return 0.15 * std::cos(k * x); };
    auto fw = [&](double x){ return 0.04 * std::sin(k * x); };
    auto fp = [&](double x){ return 0.02 * std::cos(k * x); };

    std::vector<double> h2(Nd2), u2(Nd2), v2(Nd2, 0.0), w2(Nd2), p2(Nd2);
    for (int i = 0; i < Nd2; ++i) { h2[i]=fh(x2[i]); u2[i]=fu(x2[i]); w2[i]=fw(x2[i]); p2[i]=fp(x2[i]); }
    auto U2 = s2.makeState(h2, u2, v2, w2, p2);

    const auto x1 = s1.xCoords();
    std::vector<double> h1(Nd1), u1(Nd1), w1(Nd1), p1(Nd1);
    for (int i = 0; i < Nd1; ++i) { h1[i]=fh(x1[i]); u1[i]=fu(x1[i]); w1[i]=fw(x1[i]); p1[i]=fp(x1[i]); }
    auto U1 = s1.makeState(h1, u1, w1, p1);

    sgn::State R2, R1; s2.rhs(U2, R2); s1.rhs(U1, R1);

    // Compare at ey=0, b=0 line; 2D var (H,HU,HW,HP)=(0,1,3,4) <-> 1D (0,1,2,3).
    const int map2[4] = {0, 1, 3, 4}, map1[4] = {0, 1, 2, 3};
    double err = 0.0, hvMax = 0.0;
    for (int ex = 0; ex < Nx; ++ex)
      for (int a = 0; a < np; ++a) {
        const int e2 = ex; // ey=0
        const int g2 = e2 * np * np + a + np * 0;
        const int g1 = ex * np + a;
        for (int m = 0; m < 4; ++m)
          err = std::fmax(err, std::fabs(R2[(size_t)map2[m]*Nd2 + g2] -
                                         R1[(size_t)map1[m]*Nd1 + g1]));
        hvMax = std::fmax(hvMax, std::fabs(R2[(size_t)sgn2d::HV*Nd2 + g2]));
      }
    check(err < 1e-12, "x-embedding: 2D RHS == 1D RHS (bitwise-ish)", err);
    check(hvMax < 1e-12, "x-embedding: hv RHS == 0", hvMax);
  }

  // 4) y-embedding vs 1D driver along y.
  {
    sgn::SGN2D s2(P, Nx, Ny, Lx, Ly, par);
    sgn::SGN1D s1(P, Ny, Ly, par1);                 // 1D along y-mesh
    const int Nd2 = s2.nDof(), Nd1 = s1.nDof(), np = P + 1;
    std::vector<double> x2, y2; s2.nodeCoords(x2, y2);
    const double k = 2.0 * kPi / Ly;
    auto fh = [&](double y){ return H0 + 0.08 * std::sin(k * y); };
    auto fv = [&](double y){ return 0.15 * std::cos(k * y); };
    auto fw = [&](double y){ return 0.04 * std::sin(k * y); };
    auto fp = [&](double y){ return 0.02 * std::cos(k * y); };

    std::vector<double> h2(Nd2), u2(Nd2, 0.0), v2(Nd2), w2(Nd2), p2(Nd2);
    for (int i = 0; i < Nd2; ++i) { h2[i]=fh(y2[i]); v2[i]=fv(y2[i]); w2[i]=fw(y2[i]); p2[i]=fp(y2[i]); }
    auto U2 = s2.makeState(h2, u2, v2, w2, p2);

    const auto y1 = s1.xCoords();
    std::vector<double> h1(Nd1), u1(Nd1), w1(Nd1), p1(Nd1);
    for (int i = 0; i < Nd1; ++i) { h1[i]=fh(y1[i]); u1[i]=fv(y1[i]); w1[i]=fw(y1[i]); p1[i]=fp(y1[i]); }
    auto U1 = s1.makeState(h1, u1, w1, p1);

    sgn::State R2, R1; s2.rhs(U2, R2); s1.rhs(U1, R1);

    // 2D var (H,HV,HW,HP)=(0,2,3,4) <-> 1D (0,1,2,3); compare at ex=0, a=0.
    const int map2[4] = {0, 2, 3, 4}, map1[4] = {0, 1, 2, 3};
    double err = 0.0, huMax = 0.0;
    for (int ey = 0; ey < Ny; ++ey)
      for (int b = 0; b < np; ++b) {
        const int e2 = 0 + Nx * ey; // ex=0
        const int g2 = e2 * np * np + 0 + np * b;
        const int g1 = ey * np + b;
        for (int m = 0; m < 4; ++m)
          err = std::fmax(err, std::fabs(R2[(size_t)map2[m]*Nd2 + g2] -
                                         R1[(size_t)map1[m]*Nd1 + g1]));
        huMax = std::fmax(huMax, std::fabs(R2[(size_t)sgn2d::HU*Nd2 + g2]));
      }
    check(err < 1e-12, "y-embedding: 2D RHS == 1D RHS (bitwise-ish)", err);
    check(huMax < 1e-12, "y-embedding: hu RHS == 0", huMax);
  }

  // 5) Mass conservation over a short 2D run.
  {
    sgn::SGN2D sys(P, Nx, Ny, Lx, Ly, par, bump2d);
    const int Nd = sys.nDof();
    std::vector<double> x, y; sys.nodeCoords(x, y);
    const double kx = 2.0 * kPi / Lx, ky = 2.0 * kPi / Ly;
    std::vector<double> h(Nd), z(Nd, 0.0);
    for (int i = 0; i < Nd; ++i)
      h[i] = bump2d(x[i], y[i]) + 0.03 * std::sin(kx * x[i]) * std::cos(ky * y[i]);
    auto U = sys.makeState(h, z, z, z, z);
    const double M0 = sys.mass(U);
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-9; opt.rtol = 1e-7;
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, 0.4,
                                           sys.cflTimeStep(U, 0.5), opt);
    bool finite = true; for (double vv : U) if (!std::isfinite(vv)) { finite=false; break; }
    const double dM = std::fabs(sys.mass(U) - M0) / std::fabs(M0);
    check(!st.aborted && finite, "2D run: clean finite", st.tReached);
    check(dM < 1e-13, "2D run: mass conserved (machine)", dM);
    std::printf("       [info] steps=%ld  E0=%.6e  E=%.6e\n",
                st.accepted, sys.energy(sys.makeState(h,z,z,z,z)), sys.energy(U));
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
