// =============================================================================
// val_figures.cpp -- data for the validation figures (1D CPU)
// =============================================================================
//
// Dumps profile / time-series data for the validation-ladder figures that the
// regression tests only check numerically. CPU, reuses the tested solver.
//
//   usage: ./build/val_figures
//   output:
//     results/val_solitary.csv        x, eta_initial, eta_final   (permanent form)
//     results/val_collision.csv       x, eta at 5 instants        (equal solitons)
//     results/val_collision_uneq.csv  x, eta at 5 instants        (unequal solitons)
//     results/val_wb.csv              t, max|eta|, max|RHS|       (well-balancing)
//     results/val_wb_bathy.csv        x, H(x)                     (bump, for the plot)
//
// CALL CHAIN
//   main -> four independent blocks, each: SGN1D -> initial state (soliton()
//   below, or lake at rest) -> integrateAdaptive (src/solver/rk.cpp) between
//   snapshot times -> CSV. scripts/make_figures.py turns the CSVs into figures.
// =============================================================================

#include "../src/solver/sgn1d.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

// SGN solitary wave initial state on a given system (flat bottom, depth h0).
//   a    amplitude,  x0  crest position,  dir  +1 right-moving, -1 left-moving
//   cs   celerity sqrt(g(h0+a)); kap inverse width of the sech^2 profile;
//   u, w, p follow from the travelling-wave identities of the SGN model.
static sgn::State soliton(const sgn::SGN1D& sys, double g, double h0, double a,
                          double gamma, double x0, int dir) {
  const int N = sys.nDof();
  const auto x = sys.xCoords();
  const double cs = std::sqrt(g * (h0 + a));
  const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a)));
  std::vector<double> h(N), u(N), w(N), p(N);
  for (int i = 0; i < N; ++i) {
    const double t = std::tanh(kap * (x[i] - x0));
    const double s = 1.0 - t * t;
    const double hh = h0 + a * s;
    const double hx = -2.0 * kap * a * s * t;
    const double hxx = 4.0 * kap * kap * a * s * t * t - 2.0 * kap * kap * a * s * s;
    h[i] = hh;
    u[i] = dir * cs * (1.0 - h0 / hh);
    w[i] = dir * (-cs * h0 * hx / (2.0 * hh));
    p[i] = -(cs * h0 / gamma) *
           (-0.5 * cs * h0 * (hxx * hh - hx * hx) / (hh * hh));
  }
  return sys.makeState(h, u, w, p);
}

int main() {
  const double g = 9.81, gamma = 1.5, alpha = 3.0;

  // ---------------- Solitary wave: permanent-form propagation --------------
  {
    const double h0 = 1.0, a = 0.2, L = 40.0;
    const int P = 4, ne = 80, N = ne * (P + 1);
    sgn::Params par; par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * h0);
    sgn::SGN1D sys(P, ne, L, par);
    const auto x = sys.xCoords();
    auto U0 = soliton(sys, g, h0, a, gamma, 0.5 * L, +1);
    auto U = U0;
    const double cs = std::sqrt(g * (h0 + a));
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-12; opt.rtol = 1e-9;
    const double T = L / cs; // one full periodic transit -> back to start
    sgn::integrateAdaptive(sys.rhsFn(), U, 0.0, T, sys.cflTimeStep(U, 0.5), opt);
    std::FILE* f = std::fopen("results/val_solitary.csv", "w");
    std::fprintf(f, "x,eta0,etaN\n");
    for (int i = 0; i < N; ++i)
      std::fprintf(f, "%.5f,%.8e,%.8e\n", x[i], U0[i] - h0, U[i] - h0);
    std::fclose(f);
    std::printf("# solitary: one transit T=%.3f done -> val_solitary.csv\n", T);
  }

  // ---------------- Head-on collision: snapshots ---------------------------
  {
    const double h0 = 1.0, a = 0.2, L = 80.0;
    const int P = 4, ne = 160, N = ne * (P + 1);
    sgn::Params par; par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * h0);
    sgn::SGN1D sys(P, ne, L, par);
    const auto x = sys.xCoords();
    // superpose a right-mover at L/4 and a left-mover at 3L/4
    auto UR = soliton(sys, g, h0, a, gamma, 0.25 * L, +1);
    auto UL = soliton(sys, g, h0, a, gamma, 0.75 * L, -1);
    sgn::State U(sys.nTotal());
    for (int v = 0; v < sgn::NVAR; ++v)
      for (int i = 0; i < N; ++i) {
        const int idx = v * N + i;
        // add perturbations: h -> h0 + (hR-h0)+(hL-h0); momenta add
        if (v == 0) U[idx] = UR[idx] + UL[idx] - h0;
        else U[idx] = UR[idx] + UL[idx];
      }
    const double cs = std::sqrt(g * (h0 + a));
    // each soliton travels L/4 to the centre: collision time
    const double tCollide = (0.25 * L) / cs;
    const double tEnd = 2.0 * tCollide;
    const double snap[5] = {0.0, 0.6 * tCollide, tCollide, 1.4 * tCollide, tEnd};
    std::vector<std::vector<double>> eta(5, std::vector<double>(N));
    for (int i = 0; i < N; ++i) eta[0][i] = U[i] - h0;
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-12; opt.rtol = 1e-9;
    for (int s = 1; s < 5; ++s) {
      sgn::integrateAdaptive(sys.rhsFn(), U, snap[s - 1], snap[s],
                             sys.cflTimeStep(U, 0.5), opt);
      for (int i = 0; i < N; ++i) eta[s][i] = U[i] - h0;
    }
    std::FILE* f = std::fopen("results/val_collision.csv", "w");
    std::fprintf(f, "x,eta_t0,eta_t1,eta_t2,eta_t3,eta_t4\n");
    for (int i = 0; i < N; ++i) {
      std::fprintf(f, "%.5f", x[i]);
      for (int s = 0; s < 5; ++s) std::fprintf(f, ",%.8e", eta[s][i]);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::printf("# collision: snapshots at t/tc = 0,0.6,1,1.4,2 -> val_collision.csv\n");
  }

  // ---------------- Head-on collision of UNEQUAL solitons ------------------
  {
    const double h0 = 1.0, aR = 0.25, aL = 0.10, L = 80.0;
    const int P = 4, ne = 160, N = ne * (P + 1);
    sgn::Params par; par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * h0);
    sgn::SGN1D sys(P, ne, L, par);
    const auto x = sys.xCoords();
    auto UR = soliton(sys, g, h0, aR, gamma, 0.25 * L, +1); // larger, faster
    auto UL = soliton(sys, g, h0, aL, gamma, 0.75 * L, -1); // smaller, slower
    sgn::State U(sys.nTotal());
    for (int v = 0; v < sgn::NVAR; ++v)
      for (int i = 0; i < N; ++i) {
        const int idx = v * N + i;
        U[idx] = (v == 0) ? UR[idx] + UL[idx] - h0 : UR[idx] + UL[idx];
      }
    const double csR = std::sqrt(g * (h0 + aR)), csL = std::sqrt(g * (h0 + aL));
    const double tCollide = (0.5 * L) / (csR + csL); // closing speed = csR+csL
    const double tEnd = 2.0 * tCollide;
    const double snap[5] = {0.0, 0.6 * tCollide, tCollide, 1.4 * tCollide, tEnd};
    std::vector<std::vector<double>> eta(5, std::vector<double>(N));
    for (int i = 0; i < N; ++i) eta[0][i] = U[i] - h0;
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-12; opt.rtol = 1e-9;
    for (int s = 1; s < 5; ++s) {
      sgn::integrateAdaptive(sys.rhsFn(), U, snap[s - 1], snap[s],
                             sys.cflTimeStep(U, 0.5), opt);
      for (int i = 0; i < N; ++i) eta[s][i] = U[i] - h0;
    }
    std::FILE* f = std::fopen("results/val_collision_uneq.csv", "w");
    std::fprintf(f, "x,eta_t0,eta_t1,eta_t2,eta_t3,eta_t4\n");
    for (int i = 0; i < N; ++i) {
      std::fprintf(f, "%.5f", x[i]);
      for (int s = 0; s < 5; ++s) std::fprintf(f, ",%.8e", eta[s][i]);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::printf("# unequal collision (aR=0.25, aL=0.10) -> val_collision_uneq.csv\n");
  }

  // ---------------- Well-balancing: lake at rest over a bump ----------------
  {
    const double H0 = 1.0, L = 20.0;
    const int P = 4, ne = 32, N = ne * (P + 1);
    sgn::Params par; par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);
    auto bump = [L](double x) {
      const double dx = x - 0.5 * L;
      return 1.0 - 0.4 * std::exp(-dx * dx / 8.0);
    };
    sgn::SGN1D sys(P, ne, L, par, bump);
    const auto x = sys.xCoords();
    std::vector<double> h(N), z(N, 0.0);
    for (int i = 0; i < N; ++i) h[i] = bump(x[i]);
    auto U = sys.makeState(h, z, z, z);
    // bathymetry profile for the plot
    std::FILE* fb = std::fopen("results/val_wb_bathy.csv", "w");
    std::fprintf(fb, "x,H\n");
    for (int i = 0; i < N; ++i) std::fprintf(fb, "%.5f,%.6f\n", x[i], bump(x[i]));
    std::fclose(fb);

    std::FILE* f = std::fopen("results/val_wb.csv", "w");
    std::fprintf(f, "t,max_abs_eta,max_abs_rhs\n");
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-10; opt.rtol = 1e-8;
    const int nS = 40; const double tEnd = 5.0;
    for (int s = 0; s <= nS; ++s) {
      const double t = s * tEnd / nS;
      if (s > 0)
        sgn::integrateAdaptive(sys.rhsFn(), U, (s - 1) * tEnd / nS, t,
                               sys.cflTimeStep(U, 0.5), opt);
      double me = 0.0;
      for (int i = 0; i < N; ++i) me = std::fmax(me, std::fabs(U[i] - bump(x[i])));
      sgn::State R; sys.rhs(U, R);
      double mr = 0.0; for (double r : R) mr = std::fmax(mr, std::fabs(r));
      std::fprintf(f, "%.4f,%.6e,%.6e\n", t, me, mr);
    }
    std::fclose(f);
    std::printf("# well-balancing: lake at rest to t=%.1f -> val_wb.csv\n", tEnd);
  }

  std::printf("# all validation-figure data written to results/\n");
  return 0;
}
