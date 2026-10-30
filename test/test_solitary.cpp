// =============================================================================
// test_solitary.cpp -- validation TEST 2: solitary-wave permanent form
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// TEST 2 of the validation ladder: solitary-wave permanent-form propagation.
//
// Classical SGN (Serre) solitary wave (flat bottom, depth h0, amplitude a):
//   h(x,t) = h0 + a sech^2( kap (x - cs t) ),  kap^2 = 3a / (4 h0^2 (h0+a)),
//   cs = sqrt(g (h0 + a)),  u = cs (1 - h0/h).
//
// Consistent auxiliaries for the travelling wave (Escalante et al. 2019):
//   w = -(h/2) u_x = -cs h0 h_x / (2h)
//   p = (1/gamma) d_x( (u-cs) h w ) = -(cs h0 / gamma) w_x
//     [uses the exact travelling-wave identity (u-cs) h = -cs h0]
// This is an EXACT solution of the original SGN model but only an O(1/alpha^2)
// approximation for the hyperbolic relaxation -> small transient expected; the
// measured celerity converges QUADRATICALLY to cs as alpha grows (gated).
//
// Gates:
//   a) mass conserved to machine precision over a full domain loop
//   b) amplitude retention: |max h(T) - max h(0)| / a small (no decay)
//   c) celerity within ~1% of cs at alpha=3 (model+init error dominates)
//   d) quadratic relaxation: celerity error at alpha=6 ~ 4x smaller than
//      at alpha=3 (gate: < 0.5x)
//
// Build: part of `make run-test`.

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

struct SolitonRun {
  double cNum = 0.0;      // measured celerity
  double ampDrift = 0.0;  // |max h(T) - max h(0)| / a
  double massDrift = 0.0; // relative
  bool ok = false;        // finished, finite
};

// Peak height by 3-point parabola through the nodal maximum (sub-node accuracy).
static double peakHeight(const double* h, int N) {
  int im = 0;
  for (int i = 1; i < N; ++i)
    if (h[i] > h[im]) im = i;
  const int i0 = (im - 1 + N) % N, i2 = (im + 1) % N;
  const double y0 = h[i0], y1 = h[im], y2 = h[i2];
  const double denom = y0 - 2.0 * y1 + y2;
  if (std::fabs(denom) < 1e-300) return y1;
  const double d = 0.5 * (y0 - y2) / denom; // offset in node units, |d|<~1
  return y1 - 0.25 * (y0 - y2) * d;
}

static SolitonRun runSoliton(double alpha, double h0, double a, double g,
                             double gamma, double L, int P, int ne) {
  const double cs = std::sqrt(g * (h0 + a));
  const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a)));
  const double x0 = 0.5 * L;

  sgn::Params par;
  par.g = g;
  par.gamma = gamma;
  par.c = alpha * std::sqrt(g * h0);

  sgn::SGN1D sys(P, ne, L, par);
  const int N = sys.nDof();
  const auto x = sys.xCoords();

  std::vector<double> h(N), u(N), w(N), p(N);
  for (int i = 0; i < N; ++i) {
    const double xi = x[i] - x0;
    const double t = std::tanh(kap * xi);
    const double s = 1.0 - t * t; // sech^2
    const double hh = h0 + a * s;
    const double hx = -2.0 * kap * a * s * t;
    const double hxx = 4.0 * kap * kap * a * s * t * t -
                       2.0 * kap * kap * a * s * s;
    h[i] = hh;
    u[i] = cs * (1.0 - h0 / hh);
    w[i] = -cs * h0 * hx / (2.0 * hh);
    const double wx = -0.5 * cs * h0 * (hxx * hh - hx * hx) / (hh * hh);
    p[i] = -(cs * h0 / gamma) * wx;
  }
  auto U = sys.makeState(h, u, w, p);

  const double M0 = sys.mass(U);
  const double amp0 = peakHeight(&U[0], N) - h0;

  // Quadrature weights * J for the position-phase projection.
  std::vector<double> qwJ(N);
  {
    sgn::FROperators1D op(P);
    const double J = (L / ne) / 2.0;
    for (int e = 0; e < ne; ++e)
      for (int j = 0; j <= P; ++j) qwJ[e * (P + 1) + j] = op.weights()[j] * J;
  }
  const double k1 = 2.0 * kPi / L;
  auto posPhase = [&](const sgn::State& Us) {
    std::complex<double> m(0.0, 0.0);
    for (int i = 0; i < N; ++i)
      m += qwJ[i] * (Us[i] - h0) *
           std::exp(std::complex<double>(0.0, -k1 * x[i]));
    return std::arg(m); // = -k1 * x_peak (mod 2pi), decreases as wave advances
  };

  // One full domain loop in 8 legs (phase step per leg = 2pi/8 < pi).
  const double T = L / cs;
  const int nLeg = 8;
  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-12;
  opt.rtol = 1e-9;

  SolitonRun out;
  double phPrev = posPhase(U);
  double phTot = 0.0;
  for (int s = 0; s < nLeg; ++s) {
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, s * T / nLeg,
                                           (s + 1) * T / nLeg,
                                           sys.cflTimeStep(U, 0.5), opt);
    if (st.aborted) return out;
    const double ph = posPhase(U);
    double dph = ph - phPrev;
    while (dph > kPi) dph -= 2.0 * kPi;
    while (dph < -kPi) dph += 2.0 * kPi;
    phTot += dph;
    phPrev = ph;
  }
  for (double v : U)
    if (!std::isfinite(v)) return out;

  // Wave advances +x -> phase of the k1-mode decreases: x_travel = -phTot/k1.
  out.cNum = (-phTot / k1) / T;
  out.ampDrift = std::fabs((peakHeight(&U[0], N) - h0) - amp0) / a;
  out.massDrift = std::fabs(sys.mass(U) - M0) / std::fabs(M0);
  out.ok = true;
  return out;
}

int main() {
  std::printf("=== TEST 2: solitary wave, permanent form ===\n\n");

  const double g = 9.81, h0 = 1.0, a = 0.2, gamma = 1.5;
  const double L = 40.0;
  const int P = 4, ne = 80;
  const double cs = std::sqrt(g * (h0 + a));

  const auto r3 = runSoliton(3.0, h0, a, g, gamma, L, P, ne);
  const auto r6 = runSoliton(6.0, h0, a, g, gamma, L, P, ne);

  check(r3.ok, "alpha=3: run finishes, state finite", r3.ok ? 0.0 : 1.0);
  check(r6.ok, "alpha=6: run finishes, state finite", r6.ok ? 0.0 : 1.0);
  if (r3.ok && r6.ok) {
    const double e3 = std::fabs(r3.cNum - cs) / cs;
    const double e6 = std::fabs(r6.cNum - cs) / cs;
    check(r3.massDrift < 1e-13, "alpha=3: mass conserved (machine)", r3.massDrift);
    check(r3.ampDrift < 0.02, "alpha=3: amplitude retained (<2% of a)", r3.ampDrift);
    check(e3 < 0.01, "alpha=3: celerity within 1% of cs", e3);
    check(e6 < 0.5 * e3, "relaxation: err(alpha=6) < 0.5 err(alpha=3)", e6 / e3);
    std::printf(
        "       [info] cs=%.6f  c(a=3)=%.6f (err %.2e)  c(a=6)=%.6f (err %.2e)\n"
        "       [info] ratio e6/e3=%.3f (quadratic -> 0.25)   ampDrift: a3=%.2e a6=%.2e\n",
        cs, r3.cNum, e3, r6.cNum, e6, e6 / e3, r3.ampDrift, r6.ampDrift);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
