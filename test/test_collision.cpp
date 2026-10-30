// =============================================================================
// test_collision.cpp -- validation TEST 3: head-on solitary-wave collision
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// TEST 3 of the validation ladder: head-on solitary-wave collision.
//
// Two equal SGN solitons launched towards each other. Known SGN physics
// (Su & Mirie 1980 for the Euler analogue; standard SGN benchmark):
//   - the collision peak exceeds the linear superposition 2a (superlinear
//     run-up, ~ 2a + a^2/(2 h0) + ...),
//   - the solitons re-emerge with (almost) their original amplitude, plus a
//     small dispersive-radiation residual and a phase lag.
//
// Left-moving soliton init: for a travelling wave U(x - Vt) with V = -cs the
// identity (u-V)h = +cs h0 holds, so u and w flip sign while p keeps the SAME
// formula in h as the right-mover (non-hydrostatic pressure is even under
// direction reversal):
//   u = -+ cs (1 - h0/h),  w = -+ [ -cs h0 h_x / (2h) ],
//   p = -(cs^2 h0^2 / (2 gamma)) (h_xx h - h_x^2) / h^2   (both directions)
//
// Gates:
//   a) symmetric IC on a symmetric grid -> h stays mirror-symmetric (checks
//      the discretization has no spurious asymmetry; tolerance ~ round-off
//      growth over the run)
//   b) mass conserved to machine precision
//   c) collision peak: 2a < eta_max < 2.5a (superlinear but bounded)
//   d) post-collision amplitude retained within 5% of a
//   e) finite state, clean adaptive run
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

// Peak height by 3-point parabola through the nodal maximum.
static double peakHeight(const double* h, int N) {
  int im = 0;
  for (int i = 1; i < N; ++i)
    if (h[i] > h[im]) im = i;
  const int i0 = (im - 1 + N) % N, i2 = (im + 1) % N;
  const double y0 = h[i0], y1 = h[im], y2 = h[i2];
  const double denom = y0 - 2.0 * y1 + y2;
  if (std::fabs(denom) < 1e-300) return y1;
  const double d = 0.5 * (y0 - y2) / denom;
  return y1 - 0.25 * (y0 - y2) * d;
}

int main() {
  std::printf("=== TEST 3: head-on solitary-wave collision ===\n\n");

  const double g = 9.81, h0 = 1.0, a = 0.2, gamma = 1.5, alpha = 3.0;
  const double L = 80.0;
  const int P = 4, ne = 160;
  const double cs = std::sqrt(g * (h0 + a));
  const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a)));

  sgn::Params par;
  par.g = g;
  par.gamma = gamma;
  par.c = alpha * std::sqrt(g * h0);

  sgn::SGN1D sys(P, ne, L, par);
  const int N = sys.nDof();
  const auto x = sys.xCoords();

  // Superposition of a right-mover at L/4 and a left-mover at 3L/4 (tails at
  // separation L/2: sech^2(kap*L/4) ~ 1e-12 -> superposition is clean).
  std::vector<double> h(N), u(N), w(N), p(N);
  for (int i = 0; i < N; ++i) {
    double hh = h0, uu = 0.0, ww = 0.0, pp = 0.0;
    for (int sgnDir : {+1, -1}) {
      const double xc = (sgnDir > 0) ? 0.25 * L : 0.75 * L;
      const double t = std::tanh(kap * (x[i] - xc));
      const double s = 1.0 - t * t;
      const double hs = h0 + a * s;
      const double hx = -2.0 * kap * a * s * t;
      const double hxx = 4.0 * kap * kap * a * s * t * t -
                         2.0 * kap * kap * a * s * s;
      hh += a * s; // depth perturbations add
      uu += sgnDir * cs * (1.0 - h0 / hs);
      ww += sgnDir * (-cs * h0 * hx / (2.0 * hs));
      pp += -(cs * cs * h0 * h0 / (2.0 * gamma)) * (hxx * hs - hx * hx) /
            (hs * hs);
    }
    h[i] = hh; u[i] = uu; w[i] = ww; p[i] = pp;
  }
  auto U = sys.makeState(h, u, w, p);

  const double M0 = sys.mass(U);
  const double amp0 = peakHeight(&U[0], N) - h0;

  // Mirror-symmetry monitor about the collision point x = L/2. The node set is
  // symmetric (GLL, uniform elements), so node i mirrors to node N-1-i.
  auto asymmetry = [&](const sgn::State& Us) {
    double m = 0.0;
    for (int i = 0; i < N / 2; ++i)
      m = std::fmax(m, std::fabs(Us[i] - Us[N - 1 - i]));
    return m;
  };
  const double asym0 = asymmetry(U);

  // Time windows: peaks close at 2*cs; collision at ~t=(L/4)/cs... track the
  // global max during the collision window, then let them separate again.
  const double tCollide = (0.25 * L) / cs; // half-separation / closing half-speed
  const double tEnd = 2.0 * tCollide;      // symmetric: re-separated by L/4 each
  const int nWin = 40;                     // sample max(h) densely around impact

  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-12;
  opt.rtol = 1e-9;

  double etaMax = 0.0;
  bool aborted = false;
  for (int s = 0; s < nWin; ++s) {
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, s * tEnd / nWin,
                                           (s + 1) * tEnd / nWin,
                                           sys.cflTimeStep(U, 0.5), opt);
    if (st.aborted) { aborted = true; break; }
    etaMax = std::fmax(etaMax, peakHeight(&U[0], N) - h0);
  }
  bool finite = true;
  for (double v : U)
    if (!std::isfinite(v)) { finite = false; break; }

  const double ampEnd = peakHeight(&U[0], N) - h0;
  const double dM = std::fabs(sys.mass(U) - M0) / std::fabs(M0);
  const double asymEnd = asymmetry(U);

  check(!aborted && finite, "clean run, finite state", finite ? 0.0 : 1.0);
  check(dM < 1e-13, "mass conserved (machine)", dM);
  check(asymEnd < 1e-9, "mirror symmetry preserved", asymEnd);
  check(etaMax > 2.0 * a && etaMax < 2.5 * a,
        "collision peak superlinear: 2a < eta_max < 2.5a", etaMax / a);
  check(std::fabs(ampEnd - amp0) / a < 0.05,
        "post-collision amplitude within 5%", std::fabs(ampEnd - amp0) / a);
  std::printf(
      "       [info] amp0=%.6f  eta_max=%.6f (%.4f a; Su-Mirie ~ %.4f a)\n"
      "       [info] amp_end=%.6f  asym: t0=%.2e tEnd=%.2e  dM=%.2e\n",
      amp0, etaMax, etaMax / a,
      2.0 + 0.5 * (a / h0) + 0.75 * (a / h0) * (a / h0), ampEnd, asym0,
      asymEnd, dM);

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
