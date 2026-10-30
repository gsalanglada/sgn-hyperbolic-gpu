// =============================================================================
// test_fr_operators.cpp -- exactness of the 1D FR operators (src/core/fr_operators.*)
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Exactness tests for the 1D FR core (src/core/fr_operators.*).
//   1) Differentiation matrix is exact for polynomials of degree <= P.
//   2) Face interpolation rows lL, lR reproduce f(-1), f(+1) exactly.
//   3) frDerivative with continuous common values == exact derivative (deg<=P).
//   4) Quadrature weights sum to 2 and integrate polynomials exactly (GLL: 2P-1).
//
// Build (host, standalone):
//   g++ -std=c++17 -O2 test/test_fr_operators.cpp src/core/fr_operators.cpp -o t && ./t

#include "../src/core/fr_operators.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Test harness: check() prints "ok" or "FAIL", the description and the
// measured value, and counts failures; main() returns non-zero if any failed.
static int g_failures = 0;
static void check(bool ok, const char* what, double v) {
  std::printf("  %s: %-52s  %.3e\n", ok ? "ok  " : "FAIL", what, v);
  if (!ok) ++g_failures;
}

// Evaluate sum_m a[m] x^m and its derivative.
static double polyval(const std::vector<double>& a, double x) {
  double s = 0.0, xp = 1.0;
  for (double c : a) { s += c * xp; xp *= x; }
  return s;
}
static double polyderiv(const std::vector<double>& a, double x) {
  double s = 0.0, xp = 1.0;
  for (size_t m = 1; m < a.size(); ++m) { s += m * a[m] * xp; xp *= x; }
  return s;
}

int main() {
  std::printf("=== FR 1D operators (exactness) ===\n\n");

  for (int P = 1; P <= 6; ++P) {
    sgn::FROperators1D op(P, sgn::PointSet::GaussLobatto);
    const auto& x = op.nodes();
    const int n = P + 1;

    // A generic degree-P polynomial (coefficients 1, 0.5, ... arbitrary).
    std::vector<double> a(P + 1);
    for (int m = 0; m <= P; ++m) a[m] = std::sin(0.7 * m + 1.0);

    std::vector<double> f(n);
    for (int j = 0; j < n; ++j) f[j] = polyval(a, x[j]);

    // 1) D differentiates exactly.
    {
      auto df = op.differentiate(f);
      double err = 0.0;
      for (int i = 0; i < n; ++i)
        err = std::max(err, std::fabs(df[i] - polyderiv(a, x[i])));
      char buf[64]; std::snprintf(buf, sizeof buf, "P=%d: D f == f' (deg<=P)", P);
      check(err < 1e-11, buf, err);
    }

    // 2) Face interpolation exact.
    {
      const auto& lL = op.lL();
      const auto& lR = op.lR();
      double fL = 0.0, fR = 0.0;
      for (int j = 0; j < n; ++j) { fL += lL[j] * f[j]; fR += lR[j] * f[j]; }
      double err = std::max(std::fabs(fL - polyval(a, -1.0)),
                            std::fabs(fR - polyval(a, +1.0)));
      char buf[64]; std::snprintf(buf, sizeof buf, "P=%d: lL,lR == f(-1),f(+1)", P);
      check(err < 1e-12, buf, err);
    }

    // 3) frDerivative with continuous common faces == exact derivative.
    {
      const double fLm = polyval(a, -1.0);
      const double fRp = polyval(a, +1.0);
      auto df = op.frDerivative(f, fLm, fRp);
      double err = 0.0;
      for (int i = 0; i < n; ++i)
        err = std::max(err, std::fabs(df[i] - polyderiv(a, x[i])));
      char buf[64]; std::snprintf(buf, sizeof buf, "P=%d: frDerivative == f' (continuous)", P);
      check(err < 1e-11, buf, err);
    }

    // 4) Quadrature: weights sum to 2; integrate x^q exactly for q <= 2P-1.
    {
      const auto& qw = op.weights();
      double wsum = 0.0;
      for (double w : qw) wsum += w;
      double err = std::fabs(wsum - 2.0);
      // integral of x^q on [-1,1] = (1 - (-1)^{q+1}) / (q+1)
      const int qmax = 2 * P - 1;
      for (int q = 0; q <= qmax; ++q) {
        double approx = 0.0;
        for (int j = 0; j < n; ++j) approx += qw[j] * std::pow(x[j], q);
        const double exact = (q % 2 == 0) ? 2.0 / (q + 1) : 0.0;
        err = std::max(err, std::fabs(approx - exact));
      }
      char buf[64]; std::snprintf(buf, sizeof buf, "P=%d: quadrature exact (deg<=2P-1)", P);
      check(err < 1e-11, buf, err);
    }
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
