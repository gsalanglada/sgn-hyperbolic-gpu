// =============================================================================
// test_rk.cpp -- RK integrators on ODEs with known solutions (src/solver/rk.*)
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Tests for the RK integrators (src/solver/rk.h).
//   1) rk4Step: u' = -u converges at order 4.
//   2) integrateAdaptive: u' = -u hits exp(-1) within tolerance; clean finish.
//   3) integrateAdaptive: harmonic oscillator (2 vars), energy error bounded,
//      controller actually adapts (accepted > 0, no abort).
//
// Build (host, standalone):
//   g++ -std=c++17 -O2 -I. test/test_rk.cpp src/solver/rk.cpp -o test_rk && ./test_rk

#include "../src/solver/rk.h"

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
  std::printf("=== RK integrators ===\n\n");

  const sgn::RhsFn decay = [](double, const sgn::State& u, sgn::State& du) {
    du[0] = -u[0];
  };

  // 1) RK4 order: halve dt, error drops ~16x.
  {
    double errs[2];
    for (int r = 0; r < 2; ++r) {
      const int nSteps = 32 << r;
      const double dt = 1.0 / nSteps;
      sgn::State u{1.0};
      for (int s = 0; s < nSteps; ++s) sgn::rk4Step(decay, s * dt, dt, u);
      errs[r] = std::fabs(u[0] - std::exp(-1.0));
    }
    const double order = std::log2(errs[0] / errs[1]);
    check(order > 3.7, "rk4Step order ~4 on u'=-u", order);
  }

  // 2) Adaptive integrator accuracy + clean finish.
  {
    sgn::State u{1.0};
    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-10; opt.rtol = 1e-8;
    const auto st = sgn::integrateAdaptive(decay, u, 0.0, 1.0, 1e-3, opt);
    const double err = std::fabs(u[0] - std::exp(-1.0));
    check(!st.aborted && st.accepted > 0, "adaptive: clean finish", (double)st.accepted);
    check(err < 1e-6, "adaptive: |u(1) - e^-1| small", err);
    check(std::fabs(st.tReached - 1.0) < 1e-12, "adaptive: tReached == tEnd", st.tReached);
  }

  // 3) Harmonic oscillator: u'' = -u as a 2-var system; energy stays bounded.
  {
    const sgn::RhsFn osc = [](double, const sgn::State& u, sgn::State& du) {
      du[0] = u[1];
      du[1] = -u[0];
    };
    sgn::State u{1.0, 0.0};
    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-10; opt.rtol = 1e-8;
    const auto st = sgn::integrateAdaptive(osc, u, 0.0, 20.0, 1e-3, opt);
    const double E = 0.5 * (u[0] * u[0] + u[1] * u[1]);
    check(!st.aborted, "oscillator: no abort", (double)st.aborted);
    check(std::fabs(E - 0.5) < 1e-5, "oscillator: energy drift small (t=20)", std::fabs(E - 0.5));
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
