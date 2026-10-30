// =============================================================================
// ncp_tradeoff.cpp -- central vs upwind interface state of the non-conservative
//                     product: accuracy and well-balancing (1D CPU)
// =============================================================================
//
// Tests a hypothesis instead of asserting it.
//
// The production scheme evaluates -c^2 u d_x h with the CENTRAL interface state
// h* = (h- + h+)/2. The hypothesis was that this central choice causes the
// even-odd convergence pattern (order P+1 at odd P, order P at even P), and that
// an upwinded state would restore the optimal order at the cost of exact
// well-balancing. This driver checks both halves.
//
// OUTCOME (paper Appendix B): both halves are false. Central and upwind give the
// same errors to four significant figures at every refinement, and BOTH keep the
// lake at rest exactly (with GLL points the faces are nodes, so at rest h- = h+
// = H and every choice of h* coincides). The parity loss comes from elsewhere;
// apps/mms_bisect.cpp locates it in the base FR/GLL discretization.
//
// CALL CHAIN
//   (a) for central/upwind, P = 2, 4, ne = 8..128: SGN1D + setNcpInterface
//       -> manufactured state -> rk4Step with rhs + forcing -> L2 error of h
//   (b) for central/upwind, ne = 32..128: SGN1D with a bump -> one rhs() of the
//       lake at rest -> max |RHS|
//
// Two experiments, run for both interface treatments:
//
//   (a) MMS convergence at EVEN degrees (P = 2, 4), where the loss occurs.
//       Same manufactured solution and forcing as convergence1d.
//   (b) Lake at rest over a smooth bump: the max-norm of the discrete
//       right-hand side, which must be identically zero for a well-balanced
//       scheme.
//
//   usage:  ./build/ncp_tradeoff
//   output: results/ncp_tradeoff.csv   scheme, P, ne, hL2err, order
//           results/ncp_wellbalanced.csv   scheme, ne, maxAbsRhs
// =============================================================================

#include "../src/solver/sgn1d.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  const double g = 9.81, gamma = 1.5, alpha = 3.0;
  const double L = 1.0;
  const double k = 2.0 * kPi / L, omega = 1.0;
  const double c = alpha * std::sqrt(g * 1.0), c2 = c * c;
  const double ah = 0.10, au = 0.20, aw = 0.05, ap = 0.03;

  // ---- manufactured solution + forcing, identical to convergence1d ---------
  auto prim = [&](double x, double t, double& h, double& u, double& w, double& p,
                  double& hx, double& ux, double& wx, double& px,
                  double& ht, double& ut, double& wt, double& pt) {
    const double ph = k * x - omega * t, C = std::cos(ph), S = std::sin(ph);
    h = 1.0 + ah * C; u = au * C; w = aw * S; p = ap * C;
    hx = -ah * k * S; ux = -au * k * S; wx = aw * k * C; px = -ap * k * S;
    ht = ah * omega * S; ut = au * omega * S; wt = -aw * omega * C;
    pt = ap * omega * S;
  };
  auto forcing = [&](double x, double t, double G[sgn::NVAR]) {
    double h, u, w, p, hx, ux, wx, px, ht, ut, wt, pt;
    prim(x, t, h, u, w, p, hx, ux, wx, px, ht, ut, wt, pt);
    const double dtU1 = ht;
    const double dtU2 = ht * u + h * ut;
    const double dtU3 = ht * w + h * wt;
    const double dtU4 = ht * p + h * pt;
    const double dxF1 = hx * u + h * ux;
    const double dxF2 = hx * u * u + 2.0 * h * u * ux + g * h * hx + hx * p + h * px;
    const double dxF3 = hx * u * w + h * ux * w + h * u * wx;
    const double dxF4 = (hx * u + h * ux) * (p + c2) + h * u * px;
    G[0] = dtU1 - (-dxF1);
    G[1] = dtU2 - (-dxF2);
    G[2] = dtU3 - (-dxF3 + gamma * p);
    G[3] = dtU4 - (-dxF4 - (-c2 * u * hx) + (-2.0 * c2 * w));
  };

  std::FILE* fc = std::fopen("results/ncp_tradeoff.csv", "w");
  std::fprintf(fc, "scheme,P,ne,hL2err,order\n");

  std::printf("# Non-conservative interface state: accuracy / well-balancing"
              " trade-off\n");
  std::printf("# (a) MMS convergence at EVEN polynomial degrees\n");

  // ---- (a) MMS convergence, both interface treatments ----------------------
  for (int upw = 0; upw < 2; ++upw) {            // 0 = central, 1 = upwind
    const std::string tag = upw ? "upwind" : "central";
    std::printf("#\n# %s\n#  P   ne      hL2err       order\n", tag.c_str());
    for (int P : {2, 4}) {
      double prevErr = 0.0;
      for (int ne : {8, 16, 32, 64, 128}) {
        sgn::Params par; par.g = g; par.gamma = gamma; par.c = c;
        sgn::SGN1D sys(P, ne, L, par);              // flat bottom
        sys.setNcpInterface(upw ? sgn::SGN1D::NcpInterface::Upwind
                                : sgn::SGN1D::NcpInterface::Central);
        const int N = sys.nDof();
        const auto x = sys.xCoords();
        std::vector<double> h(N), u(N), wv(N), pv(N);
        for (int i = 0; i < N; ++i) {
          double H, U, W, Pp, d;
          prim(x[i], 0.0, H, U, W, Pp, d, d, d, d, d, d, d, d);
          h[i] = H; u[i] = U; wv[i] = W; pv[i] = Pp;
        }
        auto Uc = sys.makeState(h, u, wv, pv);

        sgn::RhsFn rhsF = [&](double t, const sgn::State& U, sgn::State& dU) {
          sys.rhs(U, dU);
          for (int i = 0; i < N; ++i) {
            double G[sgn::NVAR];
            forcing(x[i], t, G);
            for (int v = 0; v < sgn::NVAR; ++v) dU[v * N + i] += G[v];
          }
        };

        // small fixed RK4 step: time error negligible against spatial error
        const double tEnd = 0.2;
        const double dx = L / ne / (2.0 * P + 1.0);
        const double sig = std::sqrt(g * 1.2 + c2);
        double dt = 0.15 * dx / sig;
        const long nStep = (long)std::ceil(tEnd / dt);
        dt = tEnd / nStep;
        double t = 0.0;
        for (long s = 0; s < nStep; ++s) { sgn::rk4Step(rhsF, t, dt, Uc); t += dt; }

        // L2 error of h against the manufactured solution at tEnd (GLL quadrature)
        sgn::FROperators1D op(P);
        const double J = (L / ne) / 2.0;
        double e2 = 0.0;
        const int np = P + 1;
        for (int e = 0; e < ne; ++e)
          for (int j = 0; j < np; ++j) {
            const int i = e * np + j;
            double H, U, W, Pp, d;
            prim(x[i], tEnd, H, U, W, Pp, d, d, d, d, d, d, d, d);
            const double err = Uc[i] - H;
            e2 += op.weights()[j] * J * err * err;
          }
        const double err = std::sqrt(e2);
        const double order = (prevErr > 0.0) ? std::log2(prevErr / err) : 0.0;
        std::fprintf(fc, "%s,%d,%d,%.6e,%.3f\n", tag.c_str(), P, ne, err, order);
        std::printf("  %d  %4d   %.4e   %s%.2f\n", P, ne, err,
                    order > 0 ? " " : "  --", order);
        prevErr = err;
      }
    }
  }
  std::fclose(fc);

  // ---- (b) lake at rest over a smooth bump --------------------------------
  std::FILE* fw = std::fopen("results/ncp_wellbalanced.csv", "w");
  std::fprintf(fw, "scheme,ne,maxAbsRhs\n");
  std::printf("#\n# (b) lake at rest over a Gaussian bump: max |RHS|"
              "  (exact well-balancing => identically 0)\n");
  std::printf("#  scheme     ne     max|RHS|\n");

  const double Lb = 20.0;
  auto bump = [Lb](double x) {
    const double d = x - 0.5 * Lb;
    return 1.0 - 0.3 * std::exp(-d * d / 4.0);
  };

  for (int upw = 0; upw < 2; ++upw) {
    const std::string tag = upw ? "upwind" : "central";
    for (int ne : {32, 64, 128}) {
      sgn::Params par; par.g = g; par.gamma = gamma; par.c = c;
      sgn::SGN1D sys(4, ne, Lb, par, bump);
      sys.setNcpInterface(upw ? sgn::SGN1D::NcpInterface::Upwind
                              : sgn::SGN1D::NcpInterface::Central);
      const int N = sys.nDof();
      const auto x = sys.xCoords();
      // Lake at rest: h = H, u = w = p = 0.
      std::vector<double> h(N), u(N, 0.0), w(N, 0.0), p(N, 0.0);
      for (int i = 0; i < N; ++i) h[i] = bump(x[i]);
      auto U = sys.makeState(h, u, w, p);
      sgn::State dU;
      sys.rhs(U, dU);
      double mx = 0.0;
      for (double v : dU) mx = std::max(mx, std::fabs(v));
      std::fprintf(fw, "%s,%d,%.6e\n", tag.c_str(), ne, mx);
      std::printf("  %-8s %5d   %.6e%s\n", tag.c_str(), ne, mx,
                  mx == 0.0 ? "   <- exactly zero" : "   <- NOT well-balanced");
    }
  }
  std::fclose(fw);

  std::printf("#\n# -> results/ncp_tradeoff.csv, results/ncp_wellbalanced.csv\n");
  return 0;
}
