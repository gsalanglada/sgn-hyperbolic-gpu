// =============================================================================
// alpha_sweep.cpp -- relaxation parameter alpha: accuracy vs cost (1D CPU)
// =============================================================================
//
// Relaxation-parameter (alpha) accuracy-cost study (paper Section 5.5).
//
// CALL CHAIN
//   main -> for each alpha: SGN1D (src/solver) -> solitary-wave initial state
//        -> 8 legs of integrateAdaptive (src/solver/rk.cpp) -> k-mode phase
//        -> celerity error, amplitude drift, accepted steps
//
// The hyperbolic-SGN solution converges to the underlying (stiff) SGN model as
// the artificial celerity c = alpha*sqrt(g*H0) grows: the relaxation error
// decays like 1/alpha^2. The explicit time step, however, is limited by the
// fast acoustic speed ~ sqrt(g h + c^2) ~ alpha*sqrt(g h), so the cost (number
// of accepted steps for a fixed physical time) grows ~ linearly in alpha. This
// program quantifies both on a clean test with an exact reference: the SGN
// (gamma = 3/2) solitary wave, whose exact celerity is cs = sqrt(g(h0+a)) and
// which is recovered exactly in the alpha -> infinity limit.
//
// For each alpha it propagates the soliton one periodic-domain transit, measures
// the celerity error vs cs (phase of the domain k-mode) and the amplitude drift,
// and records the number of accepted steps (a deterministic, reproducible cost
// proxy) and the wall time.
//
//   usage: ./build/alpha_sweep
//   output: results/alpha_sweep.csv
//           columns: alpha, cel_err_rel, amp_drift, n_steps, wall_s
//
// Expected: cel_err_rel ~ C/alpha^2 (until the FR discretization floor),
//           n_steps growing at most linearly in alpha (measured: sub-linearly,
//           since g h is not negligible against c^2 until alpha is large).
//           Together these define the efficient operating point (~alpha=3):
//           beyond it, cost keeps growing for accuracy already below typical
//           model (SGN-vs-reality) error.
// =============================================================================

#include "../src/solver/sgn1d.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  const double g = 9.81, h0 = 1.0, a = 0.2, gamma = 1.5; // SGN soliton
  const double L = 40.0;                                 // periodic domain [m]
  const int P = 4, ne = 80;                              // degree, elements
  const double cs = std::sqrt(g * (h0 + a));             // exact celerity
  const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a))); // inverse width
  const double x0 = 0.5 * L;                             // initial crest position

  std::FILE* f = std::fopen("results/alpha_sweep.csv", "w");
  if (!f) { std::fprintf(stderr, "cannot open results/alpha_sweep.csv\n"); return 1; }
  std::fprintf(f, "alpha,cel_err_rel,amp_drift,n_steps,wall_s\n");
  std::printf("# alpha   cel_err     amp_drift   n_steps   wall[s]\n");

  // --- one complete run per alpha --------------------------------------------
  for (double alpha : {1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0}) {
    sgn::Params par;
    par.g = g; par.gamma = gamma;
    par.c = alpha * std::sqrt(g * h0);

    sgn::SGN1D sys(P, ne, L, par);
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
      return std::arg(m);
    };

    // initial amplitude: largest nodal h minus h0
    const double amp0 = [&] {
      int im = 0; for (int i = 1; i < N; ++i) if (U[i] > U[im]) im = i;
      return U[im] - h0;
    }();

    const double T = L / cs;   // one domain transit
    const int nLeg = 8;
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-12; opt.rtol = 1e-9;
    long nSteps = 0;

    const auto t0 = std::chrono::steady_clock::now();
    double phPrev = posPhase(U), phTot = 0.0;
    bool ok = true;
    for (int s = 0; s < nLeg; ++s) {
      const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, s * T / nLeg,
                                             (s + 1) * T / nLeg,
                                             sys.cflTimeStep(U, 0.5), opt);
      if (st.aborted) { ok = false; break; }
      nSteps += st.accepted;
      double ph = posPhase(U), dph = ph - phPrev;
      while (dph > kPi) dph -= 2.0 * kPi;
      while (dph < -kPi) dph += 2.0 * kPi;
      phTot += dph; phPrev = ph;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    if (!ok) { std::printf("# alpha=%.1f ABORTED\n", alpha); continue; }

    // numerical celerity = displacement (-total phase / k1) over the transit time
    const double cNum = (-phTot / k1) / T;
    const double celErr = std::fabs(cNum - cs) / cs;
    double ampEnd = 0.0;
    { int im = 0; for (int i = 1; i < N; ++i) if (U[i] > U[im]) im = i;
      ampEnd = U[im] - h0; }
    const double ampDrift = std::fabs(ampEnd - amp0) / a;

    std::fprintf(f, "%.2f,%.6e,%.6e,%ld,%.4f\n", alpha, celErr, ampDrift,
                 nSteps, wall);
    std::printf("  %5.1f   %.3e   %.3e   %6ld   %.3f\n", alpha, celErr,
                ampDrift, nSteps, wall);
  }
  std::fclose(f);
  std::printf("# -> results/alpha_sweep.csv\n");
  return 0;
}
