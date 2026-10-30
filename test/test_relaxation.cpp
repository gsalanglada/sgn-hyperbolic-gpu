// =============================================================================
// test_relaxation.cpp -- Manning friction and wave generation/absorption zones
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Gates for Manning friction and wave generation/absorption relaxation zones
// (the pieces that enable Beji-Battjes and the random-sea statistics cases).
//
//   1) Manning friction, uniform flow over flat bottom: h stays constant and
//      u(t) solves u' = -cf u^2 exactly -> u(t) = u0/(1 + u0 cf t), with
//      cf = n^2 g / h^{4/3}. Tight quantitative gate.
//   2) n=0 regression: friction off -> RHS bitwise identical to a driver
//      without the parameter (guarded branch).
//   3) Wave GENERATION: quiescent flume; a generation zone forced with the
//      slow-branch eigenmode produces a wave train whose interior amplitude
//      matches the target within a few % after the front passes.
//   4) Wave ABSORPTION: a solitary wave enters an absorbing zone (target =
//      rest); the residual/reflected surface elevation outside the zone is
//      < 1% of the incident amplitude.
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

int main() {
  std::printf("=== Manning friction + relaxation zones ===\n\n");

  const double g = 9.81, H0 = 1.0, gamma = 1.5, alpha = 3.0;

  // 1) Friction decay of a uniform flow.
  {
    sgn::Params par;
    par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);
    par.nManning = 0.05;
    sgn::SGN1D sys(4, 8, 10.0, par);
    const int N = sys.nDof();
    const double u0 = 0.8;
    std::vector<double> h(N, H0), u(N, u0), w(N, 0.0), p(N, 0.0);
    auto U = sys.makeState(h, u, w, p);

    const double tEnd = 2.0;
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-12; opt.rtol = 1e-10;
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, tEnd,
                                           sys.cflTimeStep(U, 0.5), opt);
    const double cf = par.nManning * par.nManning * g / std::pow(H0, 4.0 / 3.0);
    const double uExact = u0 / (1.0 + u0 * cf * tEnd);
    double uMaxErr = 0.0, hErr = 0.0;
    for (int i = 0; i < N; ++i) {
      uMaxErr = std::fmax(uMaxErr, std::fabs(U[1 * N + i] / U[i] - uExact));
      hErr = std::fmax(hErr, std::fabs(U[i] - H0));
    }
    check(!st.aborted, "friction: run finishes", st.tReached);
    check(hErr < 1e-12, "friction: h stays constant", hErr);
    check(uMaxErr / uExact < 1e-8, "friction: u(t) == u0/(1+u0 cf t) (rel)",
          uMaxErr / uExact);
  }

  // 2) n=0 regression: bitwise-identical RHS.
  {
    sgn::Params parA, parB;
    parA.g = parB.g = g; parA.gamma = parB.gamma = gamma;
    parA.c = parB.c = alpha * std::sqrt(g * H0);
    parA.nManning = 0.0; // explicit off
    sgn::SGN1D sysA(4, 12, 10.0, parA), sysB(4, 12, 10.0, parB);
    const int N = sysA.nDof();
    const auto x = sysA.xCoords();
    const double k = 2.0 * kPi / 10.0;
    std::vector<double> h(N), u(N), w(N), p(N);
    for (int i = 0; i < N; ++i) {
      h[i] = H0 + 0.1 * std::sin(k * x[i]);
      u[i] = 0.2 * std::cos(k * x[i]);
      w[i] = 0.05 * std::sin(k * x[i]);
      p[i] = 0.03 * std::cos(k * x[i]);
    }
    auto U = sysA.makeState(h, u, w, p);
    sgn::State Ra, Rb;
    sysA.rhs(U, Ra);
    sysB.rhs(U, Rb);
    double diff = 0.0;
    for (size_t i = 0; i < Ra.size(); ++i)
      diff = std::fmax(diff, std::fabs(Ra[i] - Rb[i]));
    check(diff == 0.0, "n=0: RHS bitwise identical (guarded branch)", diff);
  }

  // 3) Wave generation: slow-branch eigenmode target, amplitude check.
  {
    const double L = 60.0;
    const double kH = 0.8, k = kH / H0;      // wavelength ~ 7.85
    const double lam = 2.0 * kPi / k;
    sgn::Params par;
    par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);

    const double Cp = std::sqrt(sgn::slowCp2(kH, gamma, alpha) * g * H0);
    const double omega = k * Cp;
    const double A = 0.01 * H0;              // linear regime
    const double Au = Cp / H0 * A;
    const double Ap = (Cp * Cp / H0 - g) * A;
    const double Aw = -gamma * Ap / (omega * H0);

    auto waveTarget = [=](double x, double t, double prim[4]) {
      const double ph = k * x - omega * t;
      prim[0] = H0 + A * std::cos(ph);
      prim[1] = Au * std::cos(ph);
      prim[2] = Aw * std::sin(ph);
      prim[3] = Ap * std::cos(ph);
    };
    auto restTarget = [=](double, double, double prim[4]) {
      prim[0] = H0; prim[1] = prim[2] = prim[3] = 0.0;
    };

    sgn::SGN1D sys(4, 120, L, par);
    sys.addRelaxationZone(0.0, 1.2 * lam, 4.0 * omega, waveTarget);
    sys.addRelaxationZone(L - 2.0 * lam, L, 4.0 * omega, restTarget);

    const int N = sys.nDof();
    const auto x = sys.xCoords();
    std::vector<double> h(N, H0), z(N, 0.0);
    auto U = sys.makeState(h, z, z, z);

    // Run long enough for the front (group speed < Cp) to cross the interior.
    const double T = 2.0 * kPi / omega;
    const double tEnd = 12.0 * T;
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-12; opt.rtol = 1e-9;
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, tEnd,
                                           sys.cflTimeStep(U, 0.5), opt);

    // Amplitude measurement: FIRST-HARMONIC time-DFT at several interior
    // points, averaged. Unlike an envelope max, this is insensitive to the
    // small residual reflection r from the absorbing zone (error ~ r^2/2,
    // not ~ r): the standing modulation averages out across points.
    (void)x;
    double ampMean = 0.0;
    {
      const int nS = 64;                       // samples over one period
      const double xs[6] = {20.0, 22.8, 25.6, 28.4, 31.2, 34.0};
      std::complex<double> c[6];
      for (int q = 0; q < 6; ++q) c[q] = {0.0, 0.0};
      for (int s = 0; s < nS; ++s) {
        const double ts = tEnd + s * T / nS;
        sgn::integrateAdaptive(sys.rhsFn(), U, ts, ts + T / nS,
                               sys.cflTimeStep(U, 0.5), opt);
        for (int q = 0; q < 6; ++q) {
          double prim[4];
          sys.samplePrimitive(U, xs[q], prim);
          c[q] += (prim[0] - H0) *
                  std::exp(std::complex<double>(0.0, -omega * (ts + T / nS)));
        }
      }
      for (int q = 0; q < 6; ++q) ampMean += 2.0 * std::abs(c[q]) / nS;
      ampMean /= 6.0;
    }
    const double relAmp = ampMean / A;
    check(!st.aborted, "generation: run finishes", st.tReached);
    check(std::fabs(relAmp - 1.0) < 0.03,
          "generation: 1st-harmonic amplitude == target (3%)", relAmp);
    std::printf("       [info] A_target=%.4e  A_measured=%.4e (ratio %.4f)\n",
                A, ampMean, relAmp);
  }

  // 4) Absorption of a solitary wave.
  {
    const double L = 40.0, h0 = 1.0, a = 0.15;
    sgn::Params par;
    par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * h0);
    const double cs = std::sqrt(g * (h0 + a));
    const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a)));
    const double x0 = 10.0;

    auto restTarget = [=](double, double, double prim[4]) {
      prim[0] = h0; prim[1] = prim[2] = prim[3] = 0.0;
    };

    sgn::SGN1D sys(4, 80, L, par);
    sys.addRelaxationZone(22.0, 40.0, 3.0, restTarget);

    const int N = sys.nDof();
    const auto x = sys.xCoords();
    std::vector<double> h(N), u(N), w(N), p(N);
    for (int i = 0; i < N; ++i) {
      const double t = std::tanh(kap * (x[i] - x0));
      const double s = 1.0 - t * t;
      const double hh = h0 + a * s;
      const double hx = -2.0 * kap * a * s * t;
      const double hxx = 4.0 * kap * kap * a * s * t * t -
                         2.0 * kap * kap * a * s * s;
      h[i] = hh;
      u[i] = cs * (1.0 - h0 / hh);
      w[i] = -cs * h0 * hx / (2.0 * hh);
      p[i] = -(cs * h0 / gamma) *
             (-0.5 * cs * h0 * (hxx * hh - hx * hx) / (hh * hh));
    }
    auto U = sys.makeState(h, u, w, p);

    // Soliton reaches the zone at t ~ (22-10)/cs ~ 3.5; run well past that.
    const double tEnd = 9.0;
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-11; opt.rtol = 1e-8;
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, tEnd,
                                           sys.cflTimeStep(U, 0.5), opt);

    double etaResidual = 0.0;
    for (int i = 0; i < N; ++i)
      if (x[i] < 22.0)
        etaResidual = std::fmax(etaResidual, std::fabs(U[i] - h0));
    check(!st.aborted, "absorption: run finishes", st.tReached);
    check(etaResidual / a < 0.01,
          "absorption: reflected/residual eta < 1% of a", etaResidual / a);
    std::printf("       [info] residual=%.3e (%.3f%% of a)\n", etaResidual,
                100.0 * etaResidual / a);
  }

  // 5) Gauge sampler: exact for the element polynomial.
  {
    sgn::Params par;
    par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);
    sgn::SGN1D sys(4, 10, 10.0, par);
    const int N = sys.nDof();
    const auto x = sys.xCoords();
    // The sampler interpolates the CONSERVED variables, so exactness needs
    // h (deg<=P) AND h*u (deg<=P): take a cubic h and a CONSTANT u (h*u then
    // stays cubic). A non-constant u would make h*u degree>P -> legitimate
    // aliasing, not a sampler bug.
    auto fh = [&](double xx){ return H0 + 0.01 * xx + 0.002 * xx * xx - 1e-4 * xx * xx * xx; };
    const double u0 = 0.05;
    std::vector<double> h(N), u(N, u0), w(N, 0.0), p(N, 0.0);
    for (int i = 0; i < N; ++i) h[i] = fh(x[i]);
    auto U = sys.makeState(h, u, w, p);
    double err = 0.0;
    for (double xq : {0.37, 2.941, 5.0, 7.123, 9.99}) {
      double prim[4];
      sys.samplePrimitive(U, xq, prim);
      err = std::fmax(err, std::fabs(prim[0] - fh(xq)));
      err = std::fmax(err, std::fabs(prim[1] - u0));
    }
    check(err < 1e-12, "gauge sampler: exact for polynomial fields", err);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
