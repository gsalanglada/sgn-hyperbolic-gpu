// =============================================================================
// convergence_time.cpp -- temporal order of the production integrator (1D CPU)
// =============================================================================
//
// Temporal convergence of the PRODUCTION time integrator (paper Section 5.2).
//
// CALL CHAIN
//   main -> SGN1D with a Gaussian bump -> smooth initial state U0
//        -> for each scheme: reference march(nRef) -> march(nStep) sweep
//           (march = repeated bs3Step or rk4Step from src/solver/rk.cpp)
//        -> l2diff(U, Uref) -> observed order, flagged near the round-off floor
//
// The spatial study (convergence1d/2d) removes the time error by using a tiny
// fixed RK4 step. This driver does the mirror experiment: it fixes the mesh and
// refines dt, so what is measured is the order of the time integrator alone.
//
// The subtlety is the reference. Comparing against an analytic solution would
// hit the SPATIAL error floor of the fixed mesh long before the temporal error
// became visible, and the measured "order" would flatten to zero. Instead the
// reference is computed on the SAME mesh with the SAME spatial operator at a
// much smaller step (dtRef = dtFinest/16). The spatial error is then identical
// in both solutions and cancels exactly in the difference, leaving the temporal
// error alone.
//
// Two integrators are swept:
//   - bs3Step: the third-order update used by the production adaptive loop
//              (integrateAdaptive), with the controller switched off. Expect 3.
//   - rk4Step: classical RK4, as a control on the measurement itself. Expect 4.
//
// The problem is a smooth periodic wave over a Gaussian bump, so the flux, the
// bathymetry source and the non-conservative product are all active.
//
//   usage:  ./build/convergence_time
//   output: results/convergence_time.csv   scheme, nStep, dt, L2err, order
// =============================================================================

#include "../src/solver/sgn1d.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  const double g = 9.81, gamma = 1.5, alpha = 3.0, H0 = 1.0;
  const double L = 20.0;                          // periodic domain [m]
  const double c = alpha * std::sqrt(g * H0);     // relaxation celerity
  const int P = 4, ne = 64;                       // FIXED mesh for the whole study
  const double tEnd = 0.05;                       // integration interval [s]

  // Smooth bathymetry: Gaussian bump, so H_x != 0 and the source and the
  // non-conservative term contribute to the time derivative.
  auto bump = [L](double x) {
    const double d = x - 0.5 * L;
    return 1.0 - 0.3 * std::exp(-d * d / 4.0);
  };

  sgn::Params par; par.g = g; par.gamma = gamma; par.c = c;
  sgn::SGN1D sys(P, ne, L, par, bump);
  const int N = sys.nDof();
  const auto x = sys.xCoords();

  // Smooth initial condition (no discontinuity -> the temporal order is not
  // limited by the regularity of the solution).
  const double k = 2.0 * kPi / L;
  std::vector<double> h(N), u(N), w(N), p(N, 0.0);
  for (int i = 0; i < N; ++i) {
    h[i] = bump(x[i]) + 0.05 * std::sin(k * x[i]);
    u[i] = 0.05 * std::cos(k * x[i]);
    w[i] = 0.0;
  }
  const auto U0 = sys.makeState(h, u, w, p);

  sgn::RhsFn rhs = [&](double t, const sgn::State& U, sgn::State& dU) {
    sys.rhs(U, dU, t);
  };

  // L2 norm of the difference of two states, over all variables.
  sgn::FROperators1D op(P);
  const double J = (L / ne) / 2.0;
  const int np = P + 1;
  auto l2diff = [&](const sgn::State& A, const sgn::State& B) {
    double e2 = 0.0;
    for (int v = 0; v < sgn::NVAR; ++v)
      for (int e = 0; e < ne; ++e)
        for (int j = 0; j < np; ++j) {
          const int i = e * np + j;
          const double d = A[v * N + i] - B[v * N + i];
          e2 += op.weights()[j] * J * d * d;
        }
    return std::sqrt(e2);
  };

  // Integrate U0 over [0, tEnd] with nStep equal fixed steps of `scheme`.
  auto march = [&](const std::string& scheme, long nStep) {
    sgn::State U = U0;
    const double dt = tEnd / static_cast<double>(nStep);
    double t = 0.0;
    for (long s = 0; s < nStep; ++s) {
      if (scheme == "bs3") sgn::bs3Step(rhs, t, dt, U);
      else                 sgn::rk4Step(rhs, t, dt, U);
      t += dt;
    }
    return U;
  };

  // Separate sweeps per scheme. A fourth-order method reaches the round-off
  // floor at a much coarser step than a third-order one, so sweeping both over
  // the same range would leave the RK4 control saturated (and its apparent
  // order meaningless) on its finest steps.
  const std::vector<long> stepsBs3 = {40, 80, 160, 320, 640};
  const std::vector<long> stepsRk4 = {20, 40, 80, 160};
  const long nRef = 640 * 16;   // reference: 16x finer than the finest sweep

  // Round-off floor: accumulating independent rounding errors over nRef steps
  // behaves like a random walk, so the difference between two solutions cannot
  // resolve below ~sqrt(nRef) * eps * ||U||. Errors near this level say nothing
  // about the order of the scheme.
  const double eps = 2.220446049250313e-16;
  const sgn::State Uzero(U0.size(), 0.0);
  const double normU = l2diff(U0, Uzero);
  const double floorEst = std::sqrt((double)nRef) * eps * normU;

  std::FILE* f = std::fopen("results/convergence_time.csv", "w");
  std::fprintf(f, "scheme,nStep,dt,L2err,order\n");

  std::printf("# temporal convergence, P=%d, ne=%d, tEnd=%.3f\n", P, ne, tEnd);
  std::printf("# reference: same mesh, same operator, nStep=%ld "
              "(dt = %.3e)\n", nRef, tEnd / (double)nRef);
  std::printf("# ||U||_2 = %.3e, estimated round-off floor ~ %.2e\n",
              normU, floorEst);

  for (const std::string& scheme : {std::string("bs3"), std::string("rk4")}) {
    const bool isBs3 = (scheme == "bs3");
    std::printf("#\n# %s%s\n", scheme.c_str(),
                isBs3 ? "  (production integrator, expect order 3)"
                      : "  (control, expect order 4)");
    std::printf("#  nStep         dt        L2err     order\n");

    const sgn::State Uref = march(scheme, nRef);
    double prev = 0.0;
    for (long ns : (isBs3 ? stepsBs3 : stepsRk4)) {
      const double dt = tEnd / static_cast<double>(ns);
      const sgn::State U = march(scheme, ns);
      if (!std::isfinite(U[0])) {   // step above the stability limit
        std::printf("  %5ld  %.4e   UNSTABLE (blew up)\n", ns, dt);
        prev = 0.0;
        continue;
      }
      const double err = l2diff(U, Uref);
      const double order = (prev > 0.0) ? std::log2(prev / err) : 0.0;
      std::fprintf(f, "%s,%ld,%.6e,%.6e,%.3f\n", scheme.c_str(), ns, dt, err,
                   order);
      std::printf("  %5ld  %.4e  %.4e   %s%.2f%s\n", ns, dt, err,
                  order > 0 ? " " : "  --", order,
                  err < 20.0 * floorEst ? "   <- at round-off floor" : "");
      prev = err;
    }
  }

  std::fclose(f);
  std::printf("#\n# -> results/convergence_time.csv\n");
  return 0;
}
