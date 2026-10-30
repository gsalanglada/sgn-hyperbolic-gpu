// =============================================================================
// rk_compare.cpp -- fixed-step RK4 vs adaptive BS3(2), accuracy vs cost (1D CPU)
// =============================================================================
//
// Fixed-step RK4 vs adaptive embedded Bogacki-Shampine 3(2): accuracy vs cost.
//
// CALL CHAIN
//   main -> SGN1D (src/solver) -> makeSoliton
//        -> RK4 sweep:      rk4Step (src/solver/rk.cpp) per step, phase per leg
//        -> adaptive sweep: integrateAdaptive (rk.cpp) per leg, phase per leg
//   The celerity is measured from the phase of the domain k-mode, unwrapped over
//   8 legs of one domain transit.
//
// Justifies the adaptive integrator used throughout the paper. On the SGN
// solitary wave (exact reference celerity cs = sqrt(g(h0+a))), we propagate one
// domain transit and measure the celerity error against the number of RHS
// evaluations (the deterministic, hardware-independent cost). Fixed RK4 is run
// at a sweep of time steps (4 RHS/step); the adaptive integrator is run at a
// sweep of tolerances (variable RHS/step). The adaptive method should reach a
// given error with fewer RHS evaluations wherever the stable step varies in
// time -- which is the case once dispersive interactions develop.
//
//   usage: ./build/rk_compare
//   output: results/rk_compare.csv   method, control, cel_err, rhs_evals, wall_s
//
// (rhs_evals for RK4 = 4 * nsteps; for adaptive it is counted by the integrator.)
// =============================================================================

#include "../src/solver/sgn1d.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  // SGN solitary wave of amplitude a over depth h0 in a periodic domain of length L
  const double g = 9.81, h0 = 1.0, a = 0.2, gamma = 1.5, alpha = 3.0;
  const double L = 40.0;
  const int P = 4, ne = 80, N = ne * (P + 1);     // degree, elements, nodes
  const double cs = std::sqrt(g * (h0 + a));      // exact solitary-wave celerity
  const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a))); // inverse width
  const double x0 = 0.5 * L;                      // initial crest position

  sgn::Params par; par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * h0);
  sgn::SGN1D sys(P, ne, L, par);
  const auto x = sys.xCoords();

  // exact SGN solitary wave (h, u, w, p consistent with the travelling-wave
  // identities), returned as a fresh state so every run starts identically
  auto makeSoliton = [&]() {
    std::vector<double> h(N), u(N), w(N), p(N);
    for (int i = 0; i < N; ++i) {
      const double t = std::tanh(kap * (x[i] - x0));
      const double s = 1.0 - t * t;
      const double hh = h0 + a * s;
      const double hx = -2.0 * kap * a * s * t;
      const double hxx = 4.0 * kap * kap * a * s * t * t - 2.0 * kap * kap * a * s * s;
      h[i] = hh; u[i] = cs * (1.0 - h0 / hh);
      w[i] = -cs * h0 * hx / (2.0 * hh);
      p[i] = -(cs * h0 / gamma) * (-0.5 * cs * h0 * (hxx * hh - hx * hx) / (hh * hh));
    }
    return sys.makeState(h, u, w, p);
  };

  // celerity via the domain k-mode phase over the transit:
  //   qwJ     quadrature weight * Jacobian at every node (for domain integrals)
  //   phase   argument of the first Fourier mode of eta = h - h0; the wave's
  //           displacement is -phase/k1, so its speed is that over the time T
  std::vector<double> qwJ(N);
  { sgn::FROperators1D op(P); const double J = (L / ne) / 2.0;
    for (int e = 0; e < ne; ++e) for (int j = 0; j <= P; ++j)
      qwJ[e * (P + 1) + j] = op.weights()[j] * J; }
  const double k1 = 2.0 * kPi / L;
  auto phase = [&](const sgn::State& U) {
    std::complex<double> m(0, 0);
    for (int i = 0; i < N; ++i)
      m += qwJ[i] * (U[i] - h0) * std::exp(std::complex<double>(0, -k1 * x[i]));
    return std::arg(m);
  };
  const double T = L / cs;             // one transit -> back to start
  const int nLeg = 8;                  // sub-intervals for phase unwrapping

  auto celErr = [&](const sgn::State& U0, double phTot) {
    const double cNum = (-phTot / k1) / T;
    return std::fabs(cNum - cs) / cs;
  };

  std::FILE* f = std::fopen("results/rk_compare.csv", "w");
  std::fprintf(f, "method,control,cel_err,rhs_evals,wall_s\n");
  std::printf("# method     control      cel_err     rhs_evals   wall[s]\n");

  // ---- Fixed RK4: sweep dt (as a fraction of a reference CFL step) ---------
  auto U0 = makeSoliton();
  const double dtCFL = sys.cflTimeStep(U0, 0.5);
  for (double frac : {2.0, 1.0, 0.5, 0.25, 0.125, 0.0625}) {
    auto U = makeSoliton();
    const double dt = frac * dtCFL;
    const long nStepPerLeg = (long)std::ceil((T / nLeg) / dt);
    const double dtA = (T / nLeg) / nStepPerLeg; // exact fit per leg
    long rhs = 0; bool bad = false;
    const auto t0 = std::chrono::steady_clock::now();
    // integrate leg by leg; accumulate the phase change of each leg, unwrapped
    // into (-pi, pi], so the total phase over the transit is not aliased
    double phPrev = phase(U), phTot = 0.0;
    for (int leg = 0; leg < nLeg; ++leg) {
      double tt = leg * (T / nLeg);
      for (long s = 0; s < nStepPerLeg; ++s) { sgn::rk4Step(sys.rhsFn(), tt, dtA, U); tt += dtA; rhs += 4; }
      double ph = phase(U), d = ph - phPrev;
      while (d > kPi) d -= 2 * kPi; while (d < -kPi) d += 2 * kPi;
      phTot += d; phPrev = ph;
      for (double v : U) if (!std::isfinite(v)) { bad = true; break; }
      if (bad) break;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (bad) { std::printf("  rk4       dt=%.3g   UNSTABLE\n", dt); std::fprintf(f, "rk4,%.4e,nan,%ld,%.4f\n", dt, rhs, wall); continue; }
    const double e = celErr(U0, phTot);
    std::fprintf(f, "rk4,%.4e,%.6e,%ld,%.4f\n", dt, e, rhs, wall);
    std::printf("  rk4       dt=%.3g   %.3e   %8ld   %.3f\n", dt, e, rhs, wall);
  }

  // ---- Adaptive BS3(2): sweep tolerance ------------------------------------
  for (double tol : {1e-6, 1e-8, 1e-10, 1e-12}) {
    auto U = makeSoliton();
    sgn::AdaptiveRKOptions opt; opt.atol = tol; opt.rtol = tol;
    long rhs = 0; bool bad = false;
    const auto t0 = std::chrono::steady_clock::now();
    double phPrev = phase(U), phTot = 0.0;
    for (int leg = 0; leg < nLeg; ++leg) {
      const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, leg * (T / nLeg),
                                             (leg + 1) * (T / nLeg),
                                             sys.cflTimeStep(U, 0.5), opt);
      rhs += st.rhsEvals;
      if (st.aborted) { bad = true; break; }
      double ph = phase(U), d = ph - phPrev;
      while (d > kPi) d -= 2 * kPi; while (d < -kPi) d += 2 * kPi;
      phTot += d; phPrev = ph;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (bad) { std::printf("  adaptive  tol=%.0e  ABORTED\n", tol); continue; }
    const double e = celErr(U0, phTot);
    std::fprintf(f, "adaptive,%.0e,%.6e,%ld,%.4f\n", tol, e, rhs, wall);
    std::printf("  adaptive  tol=%.0e  %.3e   %8ld   %.3f\n", tol, e, rhs, wall);
  }
  std::fclose(f);
  std::printf("# -> results/rk_compare.csv\n");
  return 0;
}
