// =============================================================================
// test_cuda_sgn2d.cu -- 2D GPU solver vs CPU oracle (src/cuda/sgn2d_cuda.*)
// =============================================================================
//
// HOW TO RUN   source setup_env.src && make run-cuda   (on a GPU node)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// GPU-vs-CPU gates for the 2D CUDA port.
//   1) RHS parity on a smooth state over 2D bathymetry (round-off only).
//   2) Lake at rest over a 2D bump on the GPU: RHS == 0.
//   3) Adaptive integration parity vs the CPU oracle: same controller ->
//      matching step counts expected; trajectory within tolerance; mass
//      conserved on the device trajectory.
//
// Build/run on a GPU node:  make run-cuda

#include "../src/cuda/sgn2d_cuda.h"

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

namespace { constexpr double kPi = 3.14159265358979323846; }

int main() {
  std::printf("=== CUDA 2D port: GPU == CPU gates ===\n\n");

  const double g = 9.81, H0 = 1.0, gamma = 1.5, alpha = 3.0;
  sgn2d::Params par;
  par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);

  const double Lx = 10.0, Ly = 8.0;
  const int P = 4, Nx = 8, Ny = 6;

  auto bump2d = [Lx, Ly](double x, double y) {
    const double dx = x - 0.5 * Lx, dy = y - 0.5 * Ly;
    return 1.0 - 0.4 * std::exp(-(dx * dx + dy * dy) / 6.0);
  };

  sgn::SGN2D host(P, Nx, Ny, Lx, Ly, par, bump2d);
  const int Nd = host.nDof();
  std::vector<double> x, y;
  host.nodeCoords(x, y);
  const double kx = 2.0 * kPi / Lx, ky = 2.0 * kPi / Ly;

  std::vector<double> h(Nd), uu(Nd), vv(Nd), ww(Nd), pp(Nd);
  for (int i = 0; i < Nd; ++i) {
    h[i] = bump2d(x[i], y[i]) + 0.04 * std::sin(kx * x[i]) * std::cos(ky * y[i]);
    uu[i] = 0.08 * std::cos(kx * x[i]);
    vv[i] = -0.05 * std::sin(ky * y[i]);
    ww[i] = 0.02 * std::sin(kx * x[i]);
    pp[i] = 0.01 * std::cos(ky * y[i]);
  }
  auto U = host.makeState(h, uu, vv, ww, pp);

  // 1) RHS parity.
  {
    sgn::State Rc, Rg;
    host.rhs(U, Rc);
    sgn::SGN2DCuda dev(host);
    dev.upload(U);
    dev.rhsOnce(Rg);
    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < Rc.size(); ++i) {
      diff = std::fmax(diff, std::fabs(Rc[i] - Rg[i]));
      scale = std::fmax(scale, std::fabs(Rc[i]));
    }
    check(diff / scale < 1e-12, "2D RHS parity GPU==CPU (rel, bathymetry)",
          diff / scale);
  }

  // 2) Well-balancing on the GPU.
  {
    std::vector<double> hr(Nd), z(Nd, 0.0);
    for (int i = 0; i < Nd; ++i) hr[i] = bump2d(x[i], y[i]);
    auto Ur = host.makeState(hr, z, z, z, z);
    sgn::SGN2DCuda dev(host);
    dev.upload(Ur);
    sgn::State Rg;
    dev.rhsOnce(Rg);
    double err = 0.0;
    for (double r : Rg) err = std::fmax(err, std::fabs(r));
    check(err < 1e-12, "GPU 2D lake at rest: max|RHS| == 0", err);
  }

  // 3) Adaptive parity + device mass conservation.
  {
    const double tEnd = 0.3;
    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-10; opt.rtol = 1e-8;
    const double dt0 = host.cflTimeStep(U, 0.5);

    auto Uh = U;
    const auto stH = sgn::integrateAdaptive(host.rhsFn(), Uh, 0.0, tEnd, dt0, opt);

    sgn::SGN2DCuda dev(host);
    dev.upload(U);
    const auto stG = dev.integrateAdaptive(0.0, tEnd, dt0, opt);
    sgn::State Ug;
    dev.download(Ug);

    bool finite = true;
    for (double v : Ug) if (!std::isfinite(v)) { finite = false; break; }
    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < Uh.size(); ++i) {
      diff = std::fmax(diff, std::fabs(Uh[i] - Ug[i]));
      scale = std::fmax(scale, std::fabs(Uh[i]));
    }
    const double dM = std::fabs(host.mass(Ug) - host.mass(U)) /
                      std::fabs(host.mass(U));

    check(!stG.aborted && finite, "GPU 2D adaptive run finishes, finite",
          stG.tReached);
    check(diff / scale < 1e-6, "GPU 2D trajectory == CPU (rel)", diff / scale);
    check(dM < 1e-13, "GPU 2D trajectory conserves mass (machine)", dM);
    std::printf("       [info] steps: CPU acc=%ld rej=%ld | GPU acc=%ld rej=%ld\n",
                stH.accepted, stH.rejected, stG.accepted, stG.rejected);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
