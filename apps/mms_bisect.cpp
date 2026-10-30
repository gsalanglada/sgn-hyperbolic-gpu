// =============================================================================
// mms_bisect.cpp -- bisecting the even-odd convergence loss (1D CPU)
// =============================================================================
//
// Locating the origin of the even-odd convergence loss (paper Appendix B).
//
// CALL CHAIN
//   main -> for bathymetry in {flat, variable}, relaxation in {c=0, c=3 sqrt(g h0)},
//           P = 1..4, ne = 8..128:
//           SGN1D -> manufactured state -> rk4Step with rhs + forcing
//           (forcing includes the bathymetric flux split and sources)
//        -> L2 error of h and observed order; odd degrees flagged if below P+1
//
// OUTCOME: the loss at even P survives in all four configurations, including the
// purely conservative flat-bottom c = 0 system, so it belongs to the base FR/GLL
// discretization, not to any SGN-specific term.
//
// The manufactured-solution study of the production scheme loses one order at
// even polynomial degrees. Replacing the central interface state of the
// non-conservative product by an upwinded one changes nothing (apps/
// ncp_tradeoff.cpp), so that term is not the cause. This driver bisects the
// remaining possibilities by switching off, one at a time, the two features
// that distinguish the relaxed SGN operator from a plain conservative FR
// discretization:
//
//   * the relaxation celerity c, which carries BOTH the non-conservative
//     product -c^2 u d_x h AND the stiff coupling in the pressure equation.
//     Setting c = 0 leaves a purely conservative hyperbolic system.
//   * the bathymetry H, which contributes the pre-balanced flux split and the
//     topographic source terms.
//
// Four configurations result. Reading the table:
//
//   if the parity loss survives c = 0 with a flat bottom, it belongs to the
//   base FR/GLL machinery and has nothing to do with anything specific to the
//   SGN system; if it disappears, the relaxation terms are implicated.
//
// Each configuration is run for P = 1..4. The ODD degrees are the control: a
// manufactured forcing with an algebra error would spoil every degree, so odd
// degrees recovering the clean rate P+1 certify that the forcing for that
// configuration is right.
//
//   usage:  ./build/mms_bisect
//   output: results/mms_bisect.csv   bathymetry, c, P, ne, hL2err, order
// =============================================================================

#include "../src/solver/sgn1d.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  const double g = 9.81, gamma = 1.5;
  const double L = 1.0;
  const double k = 2.0 * kPi / L, omega = 1.0;
  const double ah = 0.10, au = 0.20, aw = 0.05, ap = 0.03;
  const double aH = 0.20;                 // bathymetry amplitude when enabled

  std::FILE* f = std::fopen("results/mms_bisect.csv", "w");
  std::fprintf(f, "bathymetry,c,P,ne,hL2err,order\n");
  std::printf("# bisecting the even-odd order loss\n");
  std::printf("# odd P is the control: it must give P+1 for the forcing to be"
              " trusted\n");

  // --- the four configurations: bathy x relax --------------------------------
  for (int bathy = 0; bathy < 2; ++bathy) {        // 0 = flat, 1 = sinusoidal H
    for (int relax = 0; relax < 2; ++relax) {      // 0 = c = 0, 1 = alpha = 3
      const double alpha = relax ? 3.0 : 0.0;
      const double c = alpha * std::sqrt(g * 1.0), c2 = c * c;

      // Static bathymetry and its exact derivative.
      auto Hfun = [&](double x) {
        return bathy ? 1.0 + aH * std::sin(k * x) : 0.0;
      };
      auto Hxfun = [&](double x) {
        return bathy ? aH * k * std::cos(k * x) : 0.0;
      };

      // Manufactured primitives and their exact derivatives.
      auto prim = [&](double x, double t, double& h, double& u, double& w,
                      double& p, double& hx, double& ux, double& wx, double& px,
                      double& ht, double& ut, double& wt, double& pt) {
        const double ph = k * x - omega * t;
        const double C = std::cos(ph), S = std::sin(ph);
        h = 1.0 + ah * C; u = au * C; w = aw * S; p = ap * C;
        hx = -ah * k * S; ux = -au * k * S; wx = aw * k * C; px = -ap * k * S;
        ht = ah * omega * S; ut = au * omega * S;
        wt = -aw * omega * C; pt = ap * omega * S;
      };

      // G = d_t U* - R_cont(U*), with the bathymetric flux split and sources.
      auto forcing = [&](double x, double t, double G[sgn::NVAR]) {
        double h, u, w, p, hx, ux, wx, px, ht, ut, wt, pt;
        prim(x, t, h, u, w, p, hx, ux, wx, px, ht, ut, wt, pt);
        const double H = Hfun(x), Hx = Hxfun(x);

        const double dtU1 = ht;
        const double dtU2 = ht * u + h * ut;
        const double dtU3 = ht * w + h * wt;
        const double dtU4 = ht * p + h * pt;

        const double dxF1 = hx * u + h * ux;
        const double dxF2 = hx * u * u + 2.0 * h * u * ux +
                            g * (h * hx - H * Hx) + hx * p + h * px;
        const double dxF3 = hx * u * w + h * ux * w + h * u * wx;
        const double dxF4 = (hx * u + h * ux) * (p + c2) + h * u * px;

        const double S2 = (g * (h - H) + gamma * p) * Hx;
        const double S3 = gamma * p;
        const double S4 = c2 * u * hx - 2.0 * c2 * (w + u * Hx);

        G[0] = dtU1 - (-dxF1);
        G[1] = dtU2 - (-dxF2 + S2);
        G[2] = dtU3 - (-dxF3 + S3);
        G[3] = dtU4 - (-dxF4 + S4);
      };

      const std::string tag =
          std::string(bathy ? "variable H" : "flat H   ") + ", " +
          (relax ? "c = 3*sqrt(g h0)" : "c = 0           ");
      std::printf("#\n# %s\n#  P   ne      hL2err       order\n", tag.c_str());

      for (int P = 1; P <= 4; ++P) {
        double prevErr = 0.0;
        for (int ne : {8, 16, 32, 64, 128}) {
          sgn::Params par; par.g = g; par.gamma = gamma; par.c = c;
          sgn::SGN1D sys(P, ne, L, par,
                         bathy ? std::function<double(double)>(Hfun)
                               : std::function<double(double)>{});
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

          // Small fixed RK4 so the time error stays far below the spatial one.
          const double tEnd = 0.2;
          const double dx = L / ne / (2.0 * P + 1.0);
          const double sig = std::sqrt(g * 1.4 + c2);
          double dt = 0.15 * dx / sig;
          const long nStep = (long)std::ceil(tEnd / dt);
          dt = tEnd / nStep;
          double t = 0.0;
          for (long s = 0; s < nStep; ++s) {
            sgn::rk4Step(rhsF, t, dt, Uc);
            t += dt;
          }

          // L2 error of h at tEnd (GLL quadrature)
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
          std::fprintf(f, "%s,%.4f,%d,%d,%.6e,%.3f\n",
                       bathy ? "variable" : "flat", c, P, ne, err, order);
          std::printf("  %d  %4d   %.4e   %s%.2f%s\n", P, ne, err,
                      order > 0 ? " " : "  --", order,
                      (order > 0 && P % 2 == 1 && order < P + 0.5)
                          ? "   <- ODD DEGREE BELOW P+1: forcing suspect"
                          : "");
          prevErr = err;
        }
      }
    }
  }
  std::fclose(f);
  std::printf("#\n# -> results/mms_bisect.csv\n");
  return 0;
}
