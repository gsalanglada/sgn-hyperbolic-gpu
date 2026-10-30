// =============================================================================
// convergence2d.cpp -- spatial order of the 2D solver (manufactured solutions)
// =============================================================================
//
// Spatial convergence of the 2D tensor-product FR discretization via the method
// of manufactured solutions. Same construction as convergence1d.cpp, extended
// to the 5-variable 2D system. Manufactured solution (flat bottom, periodic,
// phi = kx x + ky y - omega t):
//   h = 1 + ah cos phi, u = au cos phi, v = av cos phi,
//   w = aw sin phi, p = ap cos phi
//
//   usage: ./build/convergence2d
//   output: results/convergence2d.csv   P, N, hL2err, order
//
// Observed (paper Table 2): order P+1 for odd P and P for even P, the same
// even-odd pattern as in 1D (the 2D P=3 rate is still approaching 4 on these
// coarse meshes).
//
// CALL CHAIN
//   main -> for P = 1..4, N = 4..32: SGN2D (flat bottom) -> manufactured initial
//        state -> fixed-step rk4Step with rhs + forcing -> L2 error of h
//
// The manufactured forcing G = d_t U* - R_cont(U*) is written out term by term
// below from the continuous 2D system (see src/physics/sgn_flux2d.h).
// =============================================================================

#include "../src/solver/sgn2d.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  const double g = 9.81, gamma = 1.5, alpha = 3.0;
  const double L = 1.0;
  const double kx = 2.0 * kPi / L, ky = 2.0 * kPi / L, omega = 1.0;
  const double c = alpha * std::sqrt(g * 1.0), c2 = c * c;
  const double ah = 0.10, au = 0.15, av = 0.12, aw = 0.05, ap = 0.03; // mode amplitudes

  // Manufactured primitives at (x,y,t) and their exact x-, y- and t-derivatives.
  auto prim = [&](double x, double y, double t,
                  double& h, double& u, double& v, double& w, double& p,
                  double& hx, double& ux, double& vx, double& wx, double& px,
                  double& hy, double& uy, double& vy, double& wy, double& py,
                  double& ht, double& ut, double& vt, double& wt, double& pt) {
    const double ph = kx * x + ky * y - omega * t;
    const double C = std::cos(ph), S = std::sin(ph);
    h = 1.0 + ah * C; u = au * C; v = av * C; w = aw * S; p = ap * C;
    hx = -ah * kx * S; ux = -au * kx * S; vx = -av * kx * S; wx = aw * kx * C; px = -ap * kx * S;
    hy = -ah * ky * S; uy = -au * ky * S; vy = -av * ky * S; wy = aw * ky * C; py = -ap * ky * S;
    ht = ah * omega * S; ut = au * omega * S; vt = av * omega * S; wt = -aw * omega * C; pt = ap * omega * S;
  };

  auto forcing = [&](double x, double y, double t, double G[sgn2d::NVAR]) {
    double h, u, v, w, p, hx, ux, vx, wx, px, hy, uy, vy, wy, py, ht, ut, vt, wt, pt;
    prim(x, y, t, h, u, v, w, p, hx, ux, vx, wx, px, hy, uy, vy, wy, py,
         ht, ut, vt, wt, pt);
    // conservative time derivatives
    const double dtU1 = ht;
    const double dtU2 = ht * u + h * ut;
    const double dtU3 = ht * v + h * vt;
    const double dtU4 = ht * w + h * wt;
    const double dtU5 = ht * p + h * pt;
    // x-flux derivatives
    const double dxF1 = hx * u + h * ux;
    const double dxF2 = hx * u * u + 2.0 * h * u * ux + g * h * hx + hx * p + h * px;
    const double dxF3 = hx * u * v + h * ux * v + h * u * vx;
    const double dxF4 = hx * u * w + h * ux * w + h * u * wx;
    const double dxF5 = (hx * u + h * ux) * (p + c2) + h * u * px;
    // y-flux derivatives
    const double dyG1 = hy * v + h * vy;
    const double dyG2 = hy * u * v + h * uy * v + h * u * vy;
    const double dyG3 = hy * v * v + 2.0 * h * v * vy + g * h * hy + hy * p + h * py;
    const double dyG4 = hy * v * w + h * vy * w + h * v * wy;
    const double dyG5 = (hy * v + h * vy) * (p + c2) + h * v * py;
    // R_cont = -dxF - dyG - Bx d_xU - By d_yU + S
    const double R1 = -(dxF1 + dyG1);
    const double R2 = -(dxF2 + dyG2);
    const double R3 = -(dxF3 + dyG3);
    const double R4 = -(dxF4 + dyG4) + gamma * p;
    const double R5 = -(dxF5 + dyG5) + c2 * u * hx + c2 * v * hy - 2.0 * c2 * w;
    G[0] = dtU1 - R1; G[1] = dtU2 - R2; G[2] = dtU3 - R3;
    G[3] = dtU4 - R4; G[4] = dtU5 - R5;
  };

  std::FILE* f = std::fopen("results/convergence2d.csv", "w");
  std::fprintf(f, "P,N,hL2err,order\n");
  std::printf("#  P    N      hL2err       order\n");

  for (int P = 1; P <= 4; ++P) {
    double prevErr = 0.0;
    for (int Nc : {4, 8, 16, 32}) {
      sgn2d::Params par; par.g = g; par.gamma = gamma; par.c = c;
      sgn::SGN2D sys(P, Nc, Nc, L, L, par);          // flat bottom (H=0)
      const int Nd = sys.nDof();
      std::vector<double> xc, yc; sys.nodeCoords(xc, yc);
      std::vector<double> h(Nd), u(Nd), vv(Nd), wv(Nd), pv(Nd);
      for (int i = 0; i < Nd; ++i) {
        double H, U, V, W, Pp, d;
        prim(xc[i], yc[i], 0.0, H, U, V, W, Pp, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
        h[i] = H; u[i] = U; vv[i] = V; wv[i] = W; pv[i] = Pp;
      }
      auto Uc = sys.makeState(h, u, vv, wv, pv);

      // forced right-hand side: discrete R(U) plus the manufactured forcing G
      sgn::RhsFn rhsF = [&](double t, const sgn::State& U, sgn::State& dU) {
        sys.rhs(U, dU);
        for (int i = 0; i < Nd; ++i) {
          double G[sgn2d::NVAR];
          forcing(xc[i], yc[i], t, G);
          for (int vN = 0; vN < sgn2d::NVAR; ++vN) dU[(size_t)vN * Nd + i] += G[vN];
        }
      };

      // small fixed RK4 step, so time error is negligible against spatial error
      const double tEnd = 0.1;
      const double dxeff = (L / Nc) / (2.0 * P + 1.0);
      const double sig = std::sqrt(g * 1.2 + c2);
      double dt = 0.15 * dxeff / sig;
      const long nStep = (long)std::ceil(tEnd / dt);
      dt = tEnd / nStep;
      double t = 0.0;
      for (long s = 0; s < nStep; ++s) { sgn::rk4Step(rhsF, t, dt, Uc); t += dt; }

      // L2 error of h
      sgn::FROperators1D op(P);
      const double Jx = sys.mesh().Jx(), Jy = sys.mesh().Jy();
      const int np = P + 1, npp = np * np;
      double e2 = 0.0;
      for (int e = 0; e < sys.nElem(); ++e)
        for (int b = 0; b < np; ++b)
          for (int a = 0; a < np; ++a) {
            const int i = e * npp + a + np * b;
            double H, U, V, W, Pp, d;
            prim(xc[i], yc[i], tEnd, H, U, V, W, Pp, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d);
            const double err = Uc[i] - H;
            e2 += op.weights()[a] * op.weights()[b] * Jx * Jy * err * err;
          }
      const double err = std::sqrt(e2);
      const double order = (prevErr > 0.0) ? std::log2(prevErr / err) : 0.0;
      std::fprintf(f, "%d,%d,%.6e,%.3f\n", P, Nc, err, order);
      std::printf("  %d  %3d   %.4e   %s%.2f\n", P, Nc, err,
                  order > 0 ? " " : "  --", order);
      prevErr = err;
    }
  }
  std::fclose(f);
  std::printf("# -> results/convergence2d.csv  (expect order ~ P+1)\n");
  return 0;
}
