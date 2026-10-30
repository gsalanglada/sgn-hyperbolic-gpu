// =============================================================================
// test_bathymetry.cpp -- validation TEST 6: well-balancing over variable bathymetry
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// TEST 6 of the validation ladder: well-balancing over variable bathymetry,
// plus consistency gates for the bathymetry code path.
//
//   1) Lake at rest over a submerged Gaussian bump: max|RHS| EXACTLY ~0 (the
//      pre-balanced flux/source vanish nodally in floating point).
//   2) Lake at rest INTEGRATED to t=1: surface stays flat to round-off.
//   3) Flat-bottom consistency: H(x)=const through the bathymetry path gives
//      the same RHS as the legacy H==0 path up to constant-offset round-off
//      (the constant H^2/2 in the flux is annihilated by the FR derivative).
//   4) Wave over the bump: clean adaptive run, mass conserved to machine
//      precision (bathymetry sources must not create/destroy mass).
//
// Build: part of `make run-test`.

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
  std::printf("=== TEST 6: well-balancing over bathymetry ===\n\n");

  const double g = 9.81, H0 = 1.0, gamma = 1.5, alpha = 3.0;
  const double L = 20.0;
  const int P = 4, ne = 32;

  sgn::Params par;
  par.g = g;
  par.gamma = gamma;
  par.c = alpha * std::sqrt(g * H0);

  // Submerged Gaussian bump: still depth 1.0 far away, 0.6 over the crest.
  auto bump = [L](double x) {
    const double dx = x - 0.5 * L;
    return 1.0 - 0.4 * std::exp(-dx * dx / 8.0);
  };

  // 1) Lake at rest: h = H(x), u = w = p = 0 -> RHS == 0.
  {
    sgn::SGN1D sys(P, ne, L, par, bump);
    const int N = sys.nDof();
    const auto x = sys.xCoords();
    std::vector<double> h(N), u(N, 0.0), w(N, 0.0), p(N, 0.0);
    for (int i = 0; i < N; ++i) h[i] = bump(x[i]);
    auto U = sys.makeState(h, u, w, p);
    sgn::State R;
    sys.rhs(U, R);
    double err = 0.0;
    for (double r : R) err = std::fmax(err, std::fabs(r));
    check(err < 1e-13, "lake at rest over bump: max|RHS| == 0", err);
  }

  // 2) Lake at rest integrated: surface stays flat.
  {
    sgn::SGN1D sys(P, ne, L, par, bump);
    const int N = sys.nDof();
    const auto x = sys.xCoords();
    std::vector<double> h(N), u(N, 0.0), w(N, 0.0), p(N, 0.0);
    for (int i = 0; i < N; ++i) h[i] = bump(x[i]);
    auto U = sys.makeState(h, u, w, p);

    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-10; opt.rtol = 1e-8;
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, 1.0,
                                           sys.cflTimeStep(U, 0.5), opt);
    double etaMax = 0.0;
    for (int i = 0; i < N; ++i)
      etaMax = std::fmax(etaMax, std::fabs(U[i] - bump(x[i])));
    check(!st.aborted, "rest integration finishes", st.tReached);
    check(etaMax < 1e-12, "surface stays flat after t=1 (max|eta|)", etaMax);
  }

  // 3) Flat-bottom consistency: bathymetry path with H=const vs legacy H==0.
  {
    sgn::SGN1D sysLegacy(P, ne, L, par);
    sgn::SGN1D sysBathy(P, ne, L, par, [](double) { return 1.0; });
    const int N = sysLegacy.nDof();
    const auto x = sysLegacy.xCoords();
    const double k = 2.0 * kPi / L;
    std::vector<double> h(N), u(N), w(N), p(N);
    for (int i = 0; i < N; ++i) {
      h[i] = H0 + 0.1 * std::sin(k * x[i]);
      u[i] = 0.2 * std::cos(k * x[i]);
      w[i] = 0.05 * std::sin(k * x[i]);
      p[i] = 0.03 * std::cos(k * x[i]);
    }
    auto U = sysLegacy.makeState(h, u, w, p);
    sgn::State Ra, Rb;
    sysLegacy.rhs(U, Ra);
    sysBathy.rhs(U, Rb);
    double err = 0.0, scale = 0.0;
    for (size_t i = 0; i < Ra.size(); ++i) {
      err = std::fmax(err, std::fabs(Ra[i] - Rb[i]));
      scale = std::fmax(scale, std::fabs(Ra[i]));
    }
    check(err / scale < 1e-12, "H=const == legacy flat (rel RHS diff)", err / scale);
  }

  // 4) Small wave over the bump: clean run + mass conservation.
  {
    sgn::SGN1D sys(P, ne, L, par, bump);
    const int N = sys.nDof();
    const auto x = sys.xCoords();
    const double k = 2.0 * kPi / L;
    std::vector<double> h(N), u(N, 0.0), w(N, 0.0), p(N, 0.0);
    for (int i = 0; i < N; ++i)
      h[i] = bump(x[i]) + 0.05 * std::sin(k * x[i]); // eta = 0.05 sin(kx)
    auto U = sys.makeState(h, u, w, p);
    const double M0 = sys.mass(U);
    const double E0 = sys.energy(U);

    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-9; opt.rtol = 1e-7;
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, 2.0,
                                           sys.cflTimeStep(U, 0.5), opt);
    bool finite = true;
    for (double v : U) if (!std::isfinite(v)) { finite = false; break; }
    const double dM = std::fabs(sys.mass(U) - M0) / std::fabs(M0);

    check(!st.aborted && finite, "wave over bump: clean finite run", st.tReached);
    check(dM < 1e-13, "wave over bump: mass conserved (machine)", dM);
    std::printf("       [info] steps=%ld  E0=%.6e  E(t=2)=%.6e\n",
                st.accepted, E0, sys.energy(U));
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
