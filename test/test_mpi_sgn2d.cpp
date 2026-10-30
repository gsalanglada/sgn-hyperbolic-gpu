// =============================================================================
// test_mpi_sgn2d.cpp -- 2D MPI solver vs serial, bitwise (src/parallel/sgn2d_mpi.*)
// =============================================================================
//
// HOW TO RUN   make run-mpi   (runs under mpirun with 1, 4 and MPIRANKS ranks)
// PASS         rank 0 evaluates every gate, prints "ok"/"FAIL" and "ALL TESTS PASSED"
//              when all hold; the exit code (0 or 1) is broadcast so every rank returns it.
// =============================================================================
//
// MPI-vs-serial reproducibility gates for the distributed 2D SGN driver.
// Same philosophy as the 1D MPI gate: each rank redundantly builds the full
// serial SGN2D and compares its owned element-columns against it.
//   1) RHS parity: BITWISE (diff exactly 0), bathymetry active.
//   2) Fixed-step RK4 trajectory (10 steps): BITWISE.
//   3) Adaptive run: finishes, global mass machine-conserved, trajectory
//      matches serial to round-off.
//
// Build/run:  make run-mpi   (runs 3, 1 and 4 ranks; Nx=10 splits unevenly)

#include "../src/parallel/sgn2d_mpi.h"
#include "../src/solver/sgn2d.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <mpi.h>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int failures = 0;
  auto gate = [&](const char* what, double globVal, double tol, bool exact) {
    if (rank == 0) {
      const bool ok = exact ? (globVal == 0.0) : (globVal < tol);
      std::printf("  %s: %-52s  %.3e\n", ok ? "ok  " : "FAIL", what, globVal);
      if (!ok) ++failures;
    }
  };

  const double g = 9.81, H0 = 1.0, gamma = 1.5, alpha = 3.0;
  const double Lx = 10.0, Ly = 8.0;
  const int P = 4, Nx = 10, Ny = 6;
  const double kx = 2.0 * kPi / Lx, ky = 2.0 * kPi / Ly;

  sgn2d::Params par;
  par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);

  auto bump2d = [Lx, Ly](double x, double y) {
    const double dx = x - 0.5 * Lx, dy = y - 0.5 * Ly;
    return 1.0 - 0.4 * std::exp(-(dx * dx + dy * dy) / 6.0);
  };
  auto ic = [&](double x, double y, double* q) { // h,u,v,w,p
    q[0] = bump2d(x, y) + 0.03 * std::sin(kx * x) * std::cos(ky * y);
    q[1] = 0.08 * std::cos(kx * x);
    q[2] = -0.05 * std::sin(ky * y);
    q[3] = 0.0;
    q[4] = 0.0;
  };

  if (rank == 0)
    std::printf("=== MPI 2D == serial reproducibility (size=%d) ===\n\n", size);

  sgn::SGN2D ser(P, Nx, Ny, Lx, Ly, par, bump2d);
  const int Nd = ser.nDof();
  const int np = P + 1, npp = np * np;

  auto buildSerialState = [&]() {
    std::vector<double> x, y;
    ser.nodeCoords(x, y);
    std::vector<double> h(Nd), u(Nd), v(Nd), w(Nd), p(Nd);
    for (int i = 0; i < Nd; ++i) {
      double q[5]; ic(x[i], y[i], q);
      h[i]=q[0]; u[i]=q[1]; v[i]=q[2]; w[i]=q[3]; p[i]=q[4];
    }
    return ser.makeState(h, u, v, w, p);
  };

  sgn::SGN2DMpi mpi(P, Nx, Ny, Lx, Ly, par, MPI_COMM_WORLD, bump2d);
  const int Nl = mpi.nLocalDof();
  const int a0 = mpi.partition().a, w0 = mpi.partition().w;

  auto buildMpiState = [&]() {
    std::vector<double> x, y;
    mpi.nodeCoordsLocal(x, y);
    std::vector<double> h(Nl), u(Nl), v(Nl), w(Nl), p(Nl);
    for (int i = 0; i < Nl; ++i) {
      double q[5]; ic(x[i], y[i], q);
      h[i]=q[0]; u[i]=q[1]; v[i]=q[2]; w[i]=q[3]; p[i]=q[4];
    }
    return mpi.makeStateLocal(h, u, v, w, p);
  };

  // Owned (var, local elem e, node loc) -> serial flat index.
  auto serIdx = [&](int var, int e, int loc) {
    const int lx = e % w0, ley = e / w0;
    const int gElem = (a0 + lx) + Nx * ley;
    return (size_t)var * Nd + (size_t)gElem * npp + loc;
  };
  auto maxOwnedDiff = [&](const sgn::State& Um, const sgn::State& Us) {
    double m = 0.0;
    for (int var = 0; var < sgn2d::NVAR; ++var)
      for (int e = 0; e < mpi.nOwnedElem(); ++e)
        for (int loc = 0; loc < npp; ++loc)
          m = std::fmax(m, std::fabs(Um[(size_t)var * Nl + (size_t)e * npp + loc] -
                                     Us[serIdx(var, e, loc)]));
    return m;
  };

  // --- Gate 1: RHS bitwise --------------------------------------------------
  {
    auto Us = buildSerialState();
    auto Um = buildMpiState();
    sgn::State Rs, Rm;
    ser.rhs(Us, Rs);
    mpi.rhs(Um, Rm);
    double localMax = maxOwnedDiff(Rm, Rs);
    double globMax = 0.0;
    MPI_Reduce(&localMax, &globMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    gate("2D RHS: MPI owned == serial (bitwise)", globMax, 0.0, true);
  }

  // --- Gate 2: RK4 bitwise ---------------------------------------------------
  {
    auto Us = buildSerialState();
    auto Um = buildMpiState();
    const double dt = 0.2 * ser.cflTimeStep(Us, 0.5);
    for (int s = 0; s < 10; ++s) {
      sgn::rk4Step(ser.rhsFn(), 0.0, dt, Us);
      mpi.rk4Step(Um, dt);
    }
    double localMax = maxOwnedDiff(Um, Us);
    double globMax = 0.0;
    MPI_Reduce(&localMax, &globMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    gate("2D RK4 10 steps: MPI owned == serial (bitwise)", globMax, 0.0, true);
  }

  // --- Gate 3: adaptive + conservation ---------------------------------------
  {
    auto Us = buildSerialState();
    auto Um = buildMpiState();
    const double M0 = mpi.mass(Um);
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-10; opt.rtol = 1e-8;
    const auto st = mpi.integrateAdaptive(Um, 0.0, 0.3, mpi.cflTimeStep(Um, 0.5), opt);
    const auto stS = sgn::integrateAdaptive(ser.rhsFn(), Us, 0.0, 0.3,
                                            ser.cflTimeStep(Us, 0.5), opt);

    int localBad = st.aborted ? 1 : 0;
    for (double vv : Um) if (!std::isfinite(vv)) { localBad = 1; break; }
    int globBad = 0;
    MPI_Reduce(&localBad, &globBad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    gate("2D adaptive: finite & finished (all ranks)", (double)globBad, 0.5, false);

    const double dM = std::fabs(mpi.mass(Um) - M0) / std::fabs(M0);
    gate("2D adaptive: global mass conserved (machine)", dM, 1e-13, false);

    double localMax = maxOwnedDiff(Um, Us), localScale = 0.0;
    for (double vv : Us) localScale = std::fmax(localScale, std::fabs(vv));
    double gMax = 0.0, gScale = 0.0;
    MPI_Reduce(&localMax, &gMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localScale, &gScale, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      gate("2D adaptive: trajectory == serial (rel)", gMax / gScale, 1e-10, false);
      std::printf("       [info] MPI steps acc=%ld rej=%ld | serial acc=%ld rej=%ld\n",
                  st.accepted, st.rejected, stS.accepted, stS.rejected);
    }
  }

  int ret = 0;
  if (rank == 0) {
    std::printf("\n");
    if (failures == 0) std::printf("ALL TESTS PASSED\n");
    else { std::printf("%d TEST(S) FAILED\n", failures); ret = 1; }
  }
  MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return ret;
}
