// =============================================================================
// test_cuda_sgn1d.cu -- 1D GPU solver vs CPU oracle (src/cuda/sgn1d_cuda.*)
// =============================================================================
//
// HOW TO RUN   source setup_env.src && make run-cuda   (on a GPU node)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// GPU-vs-CPU gates for the CUDA port of the 1D hyperbolic-SGN driver.
//   1) RHS parity on a smooth state over bathymetry: max rel diff ~ round-off
//      (same physics headers, different summation order only).
//   2) Lake at rest over a bump on the GPU: RHS == 0 (well-balancing survives
//      the port).
//   3) Adaptive integration parity: same tolerances, same controller ->
//      device solution matches host solution within integration tolerance;
//      mass conserved to machine precision on the device trajectory.
//
// Build/run on a GPU node:  make run-cuda   (see Makefile, needs nvcc)

#include "../src/cuda/sgn1d_cuda.h"

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
  std::printf("=== CUDA port: GPU == CPU gates ===\n\n");

  const double g = 9.81, H0 = 1.0, gamma = 1.5, alpha = 3.0;
  const double L = 20.0;
  const int P = 4, ne = 32;

  sgn::Params par;
  par.g = g;
  par.gamma = gamma;
  par.c = alpha * std::sqrt(g * H0);

  auto bump = [L](double x) {
    const double dx = x - 0.5 * L;
    return 1.0 - 0.4 * std::exp(-dx * dx / 8.0);
  };

  sgn::SGN1D host(P, ne, L, par, bump);
  const int N = host.nDof();
  const auto x = host.xCoords();
  const double k = 2.0 * kPi / L;

  // Smooth state over the bump.
  std::vector<double> h(N), u(N), w(N), p(N);
  for (int i = 0; i < N; ++i) {
    h[i] = bump(x[i]) + 0.05 * std::sin(k * x[i]);
    u[i] = 0.1 * std::cos(k * x[i]);
    w[i] = 0.02 * std::sin(k * x[i]);
    p[i] = 0.01 * std::cos(k * x[i]);
  }
  auto U = host.makeState(h, u, w, p);

  // 1) RHS parity.
  {
    sgn::State Rc, Rg;
    host.rhs(U, Rc);
    sgn::SGN1DCuda dev(host);
    dev.upload(U);
    dev.rhsOnce(Rg);
    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < Rc.size(); ++i) {
      diff = std::fmax(diff, std::fabs(Rc[i] - Rg[i]));
      scale = std::fmax(scale, std::fabs(Rc[i]));
    }
    check(diff / scale < 1e-12, "RHS parity GPU==CPU (rel, bathymetry)",
          diff / scale);
  }

  // 2) Well-balancing on the GPU.
  {
    std::vector<double> hr(N), z(N, 0.0);
    for (int i = 0; i < N; ++i) hr[i] = bump(x[i]);
    auto Ur = host.makeState(hr, z, z, z);
    sgn::SGN1DCuda dev(host);
    dev.upload(Ur);
    sgn::State Rg;
    dev.rhsOnce(Rg);
    double err = 0.0;
    for (double r : Rg) err = std::fmax(err, std::fabs(r));
    check(err < 1e-13, "GPU lake at rest over bump: max|RHS| == 0", err);
  }

  // 3) Adaptive integration parity + device-trajectory mass conservation.
  {
    const double tEnd = 1.0;
    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-10;
    opt.rtol = 1e-8;
    const double dt0 = host.cflTimeStep(U, 0.5);

    // Host trajectory.
    auto Uh = U;
    const auto stH = sgn::integrateAdaptive(host.rhsFn(), Uh, 0.0, tEnd, dt0, opt);

    // Device trajectory.
    sgn::SGN1DCuda dev(host);
    dev.upload(U);
    const auto stG = dev.integrateAdaptive(0.0, tEnd, dt0, opt);
    sgn::State Ug;
    dev.download(Ug);

    bool finite = true;
    for (double v : Ug)
      if (!std::isfinite(v)) { finite = false; break; }

    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < Uh.size(); ++i) {
      diff = std::fmax(diff, std::fabs(Uh[i] - Ug[i]));
      scale = std::fmax(scale, std::fabs(Uh[i]));
    }
    const double dM = std::fabs(host.mass(Ug) - host.mass(U)) /
                      std::fabs(host.mass(U));

    check(!stG.aborted && finite, "GPU adaptive run finishes, finite",
          stG.tReached);
    check(diff / scale < 1e-6, "GPU trajectory == CPU (rel, tol-limited)",
          diff / scale);
    check(dM < 1e-13, "GPU trajectory conserves mass (machine)", dM);
    std::printf("       [info] steps: CPU acc=%ld rej=%ld | GPU acc=%ld rej=%ld\n",
                stH.accepted, stH.rejected, stG.accepted, stG.rejected);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
