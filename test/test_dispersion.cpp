// =============================================================================
// test_dispersion.cpp -- validation TEST 1: linear dispersion of the discrete solver
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// TEST 1 of the validation ladder: linear dispersion relation vs Airy.
//
// Initialise the SLOW-branch eigenmode of the linearised hyperbolic-SGN system
// (paper eqs. 11-13) at small amplitude and measure the numerical phase speed
// by tracking the phase of the k-mode of (h - H) over one period.
//
// Linearised eigenmode, phase phi = k x - omega t, Cp = omega/k (slow branch):
//   eta = A cos(phi)
//   u   = (Cp/H) A cos(phi)
//   p   = (Cp^2/H - g) A cos(phi)
//   w   = -gamma * A_p / (omega H) * sin(phi)      [quadrature phase]
// Rows 1-3 of the linear system fix the amplitudes; row 4 is then satisfied
// IFF the dispersion relation (eq. 14) holds -> asserted here as a residual.
//
// Gates:
//   a) eigenmode residual of row 4 ~ 0            (analytic self-consistency)
//   b) slow branch -> SGN limit as alpha -> inf   (helper sanity)
//   c) c_num == Cp_slow(k) to ~1e-6 relative      (the discretization
//      reproduces the model's dispersion; model-vs-Airy is a MODEL error,
//      printed as a table, not gated)
//
// Build: part of `make run-test`.

#include "../src/physics/sgn_dispersion.h"
#include "../src/solver/sgn1d.h"

#include <cmath>
#include <complex>
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

// Complex amplitude of the k-mode of (h - H): a = sum qw*J * (h-H) e^{-i k x}.
static std::complex<double> kMode(const sgn::SGN1D& sys, const sgn::State& U,
                                  double H, double k,
                                  const std::vector<double>& x,
                                  const std::vector<double>& qwJ) {
  const int N = sys.nDof();
  std::complex<double> a(0.0, 0.0);
  for (int i = 0; i < N; ++i) {
    const double eta = U[i] - H; // var 0 = h
    a += qwJ[i] * eta * std::exp(std::complex<double>(0.0, -k * x[i]));
  }
  return a;
}

int main() {
  std::printf("=== TEST 1: linear dispersion vs Airy ===\n\n");

  const double g = 9.81;
  const double gamma = 1.5;
  const double alpha = 3.0;

  // b) helper sanity: alpha -> inf recovers the SGN limit.
  {
    double err = 0.0;
    for (double kH : {0.3, 1.0, 2.5}) {
      err = std::fmax(err, std::fabs(sgn::slowCp2(kH, gamma, 1e7) -
                                     sgn::sgnLimitCp2(kH, gamma)));
    }
    check(err < 1e-9, "slow branch -> SGN limit as alpha->inf", err);
  }

  // Numerical phase-speed measurement for several kH.
  const double L = 10.0;          // one wavelength per (periodic) domain
  const double k = 2.0 * kPi / L;
  const int P = 4, ne = 16;

  for (double kH : {0.5, 1.0, 2.0}) {
    const double H = kH / k;
    sgn::Params par;
    par.g = g;
    par.gamma = gamma;
    par.c = alpha * std::sqrt(g * H);

    const double Cp2 = sgn::slowCp2(kH, gamma, alpha) * g * H;
    const double Cp = std::sqrt(Cp2);
    const double omega = k * Cp;

    // Eigenmode amplitudes.
    const double A = 1e-4 * H; // linear regime
    const double Au = Cp / H * A;
    const double Ap = (Cp2 / H - g) * A;
    const double Aw = -gamma * Ap / (omega * H);

    // a) row-4 residual: H*omega*Ap - H*c^2*k*Au + 2*c^2*Aw == 0.
    {
      const double c2 = par.c * par.c;
      const double res = H * omega * Ap - H * c2 * k * Au + 2.0 * c2 * Aw;
      const double scale = std::fabs(H * omega * Ap) +
                           std::fabs(H * c2 * k * Au) +
                           std::fabs(2.0 * c2 * Aw);
      char buf[64];
      std::snprintf(buf, sizeof buf, "kH=%.1f: eigenmode row-4 residual", kH);
      check(std::fabs(res) / scale < 1e-12, buf, std::fabs(res) / scale);
    }

    sgn::SGN1D sys(P, ne, L, par);
    const int N = sys.nDof();
    const auto x = sys.xCoords();

    std::vector<double> h(N), u(N), w(N), p(N);
    for (int i = 0; i < N; ++i) {
      const double cph = std::cos(k * x[i]);
      const double sph = std::sin(k * x[i]);
      h[i] = H + A * cph;
      u[i] = Au * cph;
      p[i] = Ap * cph;
      w[i] = Aw * sph;
    }
    auto U = sys.makeState(h, u, w, p);

    // Quadrature weights * J at every node (for the k-mode projection).
    std::vector<double> qwJ(N);
    {
      // rebuild from the operator weights: qw per node within element, * J
      sgn::FROperators1D op(P);
      const double J = (L / ne) / 2.0;
      for (int e = 0; e < ne; ++e)
        for (int j = 0; j <= P; ++j) qwJ[e * (P + 1) + j] = op.weights()[j] * J;
    }

    // Track the k-mode phase over one period in 5 sub-intervals
    // (each phase step 0.4*pi < pi -> unambiguous unwrapping).
    const double T = 2.0 * kPi / omega;
    const int nSamp = 5;
    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-14;  // fields are O(1e-5): atol must sit well below
    opt.rtol = 1e-10;

    double phasePrev = std::arg(kMode(sys, U, H, k, x, qwJ));
    double phaseTot = 0.0;
    bool aborted = false;
    for (int s = 0; s < nSamp; ++s) {
      const auto st = sgn::integrateAdaptive(sys.rhsFn(), U,
                                             s * T / nSamp, (s + 1) * T / nSamp,
                                             sys.cflTimeStep(U, 0.5), opt);
      if (st.aborted) { aborted = true; break; }
      const double ph = std::arg(kMode(sys, U, H, k, x, qwJ));
      double dph = ph - phasePrev;
      while (dph > kPi) dph -= 2.0 * kPi;   // unwrap
      while (dph < -kPi) dph += 2.0 * kPi;
      phaseTot += dph;
      phasePrev = ph;
    }
    // For cos(kx - wt), the k-mode ~ e^{-i w t}: phase decreases at rate omega.
    const double omegaNum = -phaseTot / T;
    const double cNum = omegaNum / k;

    const double relErr = std::fabs(cNum - Cp) / Cp;
    char buf[64];
    std::snprintf(buf, sizeof buf, "kH=%.1f: c_num == slow branch (rel)", kH);
    check(!aborted && relErr < 1e-6, buf, relErr);

    const double cAiry = std::sqrt(sgn::airyCp2(kH) * g * H);
    const double cLimit = std::sqrt(sgn::sgnLimitCp2(kH, gamma) * g * H);
    std::printf(
        "       [kH=%.1f] c_num=%.8f  slow=%.8f  SGNlim=%.8f  Airy=%.8f\n"
        "                model-vs-Airy: %+.3f%%   discr-vs-slow: %+.2e%%\n",
        kH, cNum, Cp, cLimit, cAiry,
        100.0 * (Cp - cAiry) / cAiry, 100.0 * (cNum - Cp) / Cp);
  }

  // Analytic model-error table (no runs): reproduces the content of paper
  // Figs. 1-2 — relative slow-branch error vs Airy for alpha and gamma choices.
  std::printf("\n  [model error vs Airy, percent]  (analytic, eq. 15)\n");
  std::printf("   kH  |  g=3/2,a=3   g=3/2,a=5   g=3/2,a=10 |  g=2,a=3     g=2,a=10\n");
  for (double kH : {0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}) {
    auto pct = [&](double gam, double al) {
      return 100.0 * (std::sqrt(sgn::slowCp2(kH, gam, al) / sgn::airyCp2(kH)) - 1.0);
    };
    std::printf("   %.2f | %+10.3f %+10.3f %+10.3f | %+10.3f %+10.3f\n", kH,
                pct(1.5, 3.0), pct(1.5, 5.0), pct(1.5, 10.0),
                pct(2.0, 3.0), pct(2.0, 10.0));
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
