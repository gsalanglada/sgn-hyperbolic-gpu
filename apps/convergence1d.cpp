// =============================================================================
// convergence1d.cpp -- spatial order and cost vs accuracy of the 1D solver (MMS)
// =============================================================================
//
// Spatial convergence of the 1D FR discretization via the method of
// manufactured solutions (MMS), paper Section 5.1 (Tables 2 and 3). A smooth analytic solution U*(x,t) is chosen;
// the continuous residual G = d_t U* - R_cont(U*) is added as a forcing so that
// U* solves the forced system exactly. The numerical solution then converges to
// U* at the spatial order of the scheme. Time error is removed by using fixed
// RK4 with a small step (order 4, dt tiny -> time error << spatial error).
//
// Manufactured solution (flat bottom, periodic, phi = k x - omega t):
//   h = 1 + ah cos phi,  u = au cos phi,  w = aw sin phi,  p = ap cos phi
//
//   usage: ./build/convergence1d
//   output: results/convergence1d.csv   P, ne, dof, steps, hL2err, order,
//                                       wall_s, rhs_evals, floor, at_floor
//
// Observed: order P+1 for odd P and P for even P, for every degree P = 1..6
// (even-odd pattern, see apps/mms_bisect.cpp and paper Appendix B).
//
// CALL CHAIN
//   main -> for P = 1..6 and a mesh sequence (coarser for P >= 5):
//           SGN1D (flat bottom) -> manufactured state -> rk4Step with
//           rhs + forcing (timed) -> L2 error of h, order, round-off floor flag
// =============================================================================

#include "../src/solver/sgn1d.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  const double g = 9.81, gamma = 1.5, alpha = 3.0;
  const double L = 1.0;
  const double k = 2.0 * kPi / L, omega = 1.0;
  const double c = alpha * std::sqrt(g * 1.0), c2 = c * c;
  const double ah = 0.10, au = 0.20, aw = 0.05, ap = 0.03;

  // Manufactured primitives and their exact x/t derivatives at (x,t).
  auto prim = [&](double x, double t, double& h, double& u, double& w, double& p,
                  double& hx, double& ux, double& wx, double& px,
                  double& ht, double& ut, double& wt, double& pt) {
    const double ph = k * x - omega * t, C = std::cos(ph), S = std::sin(ph);
    h = 1.0 + ah * C; u = au * C; w = aw * S; p = ap * C;
    hx = -ah * k * S; ux = -au * k * S; wx = aw * k * C; px = -ap * k * S;
    ht = ah * omega * S; ut = au * omega * S; wt = -aw * omega * C; pt = ap * omega * S;
  };

  // Manufactured forcing G = d_t U* - R_cont(U*), R_cont = -d_x F - B d_xU + S.
  auto forcing = [&](double x, double t, double G[sgn::NVAR]) {
    double h, u, w, p, hx, ux, wx, px, ht, ut, wt, pt;
    prim(x, t, h, u, w, p, hx, ux, wx, px, ht, ut, wt, pt);
    // conservative time derivatives
    const double dtU1 = ht;
    const double dtU2 = ht * u + h * ut;
    const double dtU3 = ht * w + h * wt;
    const double dtU4 = ht * p + h * pt;
    // continuous spatial flux derivatives
    const double dxF1 = hx * u + h * ux;
    const double dxF2 = hx * u * u + 2.0 * h * u * ux + g * h * hx + hx * p + h * px;
    const double dxF3 = hx * u * w + h * ux * w + h * u * wx;
    const double dxF4 = (hx * u + h * ux) * (p + c2) + h * u * px;
    // R_cont = -dxF - B d_xU + S ; B4 = -c^2 u hx ; S = (0,0,gamma p,-2 c^2 w)
    const double R1 = -dxF1;
    const double R2 = -dxF2;
    const double R3 = -dxF3 + gamma * p;
    const double R4 = -dxF4 - (-c2 * u * hx) + (-2.0 * c2 * w);
    G[0] = dtU1 - R1; G[1] = dtU2 - R2; G[2] = dtU3 - R3; G[3] = dtU4 - R4;
  };

  std::FILE* f = std::fopen("results/convergence1d.csv", "w");
  // dof, steps and wall time turn the order study into a cost-versus-accuracy
  // one: the practical question is not which P converges fastest per element,
  // but which P reaches a target error for the least work. They are not the
  // same question, because the stable step shrinks with P (dx_eff ~ 1/(2P+1))
  // and the per-node arithmetic grows, so a higher order buys accuracy at a
  // superlinear cost per degree of freedom.
  std::fprintf(f, "P,ne,dof,steps,hL2err,order,wall_s,rhs_evals,floor,at_floor\n");
  std::printf("#  P   ne      hL2err       order     wall[s]   RHS evals\n");

  for (int P = 1; P <= 6; ++P) {
    // The standard sequence cannot be reused at every degree. At P=4 the finest
    // mesh already reaches 3e-11; two further degrees would put it under the
    // round-off floor (~sqrt(N) eps ||U||), where the observed order measures
    // cancellation rather than truncation. Shift the sequence coarser for the
    // high degrees instead of reporting noise as a convergence rate. P <= 4 is
    // left untouched so the published table is unaffected.
    const std::vector<int> meshes = (P <= 4)
        ? std::vector<int>{8, 16, 32, 64, 128}
        : std::vector<int>{4, 8, 16, 32, 64};
    double prevErr = 0.0;
    for (int ne : meshes) {
      sgn::Params par; par.g = g; par.gamma = gamma; par.c = c;
      sgn::SGN1D sys(P, ne, L, par);                 // flat bottom (H=0)
      const int N = sys.nDof();
      const auto x = sys.xCoords();
      // IC = manufactured solution at t=0
      std::vector<double> h(N), u(N), wv(N), pv(N);
      for (int i = 0; i < N; ++i) {
        double H, U, W, Pp, d;
        prim(x[i], 0.0, H, U, W, Pp, d, d, d, d, d, d, d, d);
        h[i] = H; u[i] = U; wv[i] = W; pv[i] = Pp;
      }
      auto Uc = sys.makeState(h, u, wv, pv);

      // forced RHS: dU = rhs(U) + G(x,t)
      sgn::RhsFn rhsF = [&](double t, const sgn::State& U, sgn::State& dU) {
        sys.rhs(U, dU);
        for (int i = 0; i < N; ++i) {
          double G[sgn::NVAR];
          forcing(x[i], t, G);
          for (int v = 0; v < sgn::NVAR; ++v) dU[v * N + i] += G[v];
        }
      };

      // integrate to tEnd with small fixed RK4 (isolate spatial error)
      const double tEnd = 0.2;
      const double dx = L / ne / (2.0 * P + 1.0);
      const double sig = std::sqrt(g * 1.2 + c2);       // ~ max signal speed
      double dt = 0.15 * dx / sig;
      const long nStep = (long)std::ceil(tEnd / dt);
      dt = tEnd / nStep;
      // Time only the integration: setup and the error quadrature are not part
      // of the cost being compared.
      double t = 0.0;
      const auto tic = std::chrono::steady_clock::now();
      for (long s = 0; s < nStep; ++s) { sgn::rk4Step(rhsF, t, dt, Uc); t += dt; }
      const double wall =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - tic)
              .count();
      const long rhsEvals = 4 * nStep;   // RK4

      // L2 error of h against the manufactured solution at tEnd
      sgn::FROperators1D op(P);
      const double J = (L / ne) / 2.0;
      double e2 = 0.0;
      const int np = P + 1;
      for (int e = 0; e < ne; ++e)
        for (int j = 0; j < np; ++j) {
          const int i = e * np + j;
          double H, U, W, Pp, d;
          prim(x[i], tEnd, H, U, W, Pp, d, d, d, d, d, d, d, d);
          const double err = Uc[i] - H;               // h error
          e2 += op.weights()[j] * J * err * err;
        }
      const double err = std::sqrt(e2);
      const double order = (prevErr > 0.0) ? std::log2(prevErr / err) : 0.0;
      // Round-off floor for an L2 norm accumulated over N nodes with |h| ~ 1.
      // An error within an order of magnitude of it is not a truncation error
      // and the rate computed from it is not a convergence rate.
      const double floorEst =
          std::sqrt((double)N) * 2.220446049250313e-16;
      const bool nearFloor = err < 10.0 * floorEst;
      std::fprintf(f, "%d,%d,%d,%ld,%.6e,%.3f,%.4f,%ld,%.3e,%d\n",
                   P, ne, N, nStep, err, order, wall, rhsEvals, floorEst,
                   nearFloor ? 1 : 0);
      std::printf("  %d  %4d   %.4e   %s%.2f   %8.3f   %9ld%s\n", P, ne, err,
                  order > 0 ? " " : "  --", order, wall, rhsEvals,
                  nearFloor ? "   <- AT ROUND-OFF FLOOR" : "");
      prevErr = err;
    }
  }
  std::fclose(f);
  std::printf("# -> results/convergence1d.csv  (expect order ~ P+1)\n");
  return 0;
}
