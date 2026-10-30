// =============================================================================
// test_mpi_sgn1d.cpp -- 1D MPI solver vs serial, bitwise (src/parallel/sgn1d_mpi.*)
// =============================================================================
//
// HOW TO RUN   make run-mpi   (runs under mpirun with 1, 4 and MPIRANKS ranks)
// PASS         rank 0 evaluates every gate, prints "ok"/"FAIL" and "ALL TESTS PASSED"
//              when all hold; the exit code (0 or 1) is broadcast so every rank returns it.
// =============================================================================
//
// MPI-vs-serial reproducibility gates for the distributed 1D SGN driver.
// Run under mpirun with ANY number of ranks (1..N). Each rank ALSO builds the
// full serial SGN1D redundantly and compares its owned slice against it.
//
//   1) RHS parity: MPI owned RHS == serial RHS  BITWISE (diff exactly 0).
//   2) Fixed-step RK4 trajectory (20 steps): MPI owned state == serial BITWISE.
//   3) Adaptive run: finishes on all ranks, global mass conserved to machine
//      precision, owned state matches serial to round-off (tol-limited).
//
// Bathymetry (a submerged bump) is active so the H(x) code path is exercised.
//
// Build/run:  make run-mpi     (needs mpicxx; runs mpirun -n 3 by default)

#include "../src/parallel/sgn1d_mpi.h"
#include "../src/solver/sgn1d.h"

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
  const double L = 20.0;
  const int P = 4, ne = 24; // 24 elements: divisible and not, across ranks
  const double k = 2.0 * kPi / L;

  sgn::Params par;
  par.g = g; par.gamma = gamma;
  par.c = alpha * std::sqrt(g * H0);

  auto bump = [L](double x) {
    const double dx = x - 0.5 * L;
    return 1.0 - 0.4 * std::exp(-dx * dx / 8.0);
  };
  auto ic = [&](double x, double* prim) { // h,u,w,p
    prim[0] = bump(x) + 0.03 * std::sin(k * x);
    prim[1] = 0.1 * std::cos(k * x);
    prim[2] = 0.0;
    prim[3] = 0.0;
  };

  if (rank == 0)
    std::printf("=== MPI == serial reproducibility (size=%d) ===\n\n", size);

  // Serial reference (redundant on every rank).
  sgn::SGN1D ser(P, ne, L, par, bump);
  const int Ng = ser.nDof();
  const int np = P + 1;
  auto buildSerialState = [&]() {
    const auto xs = ser.xCoords();
    std::vector<double> h(Ng), u(Ng), w(Ng), p(Ng);
    for (int i = 0; i < Ng; ++i) { double q[4]; ic(xs[i], q); h[i]=q[0]; u[i]=q[1]; w[i]=q[2]; p[i]=q[3]; }
    return ser.makeState(h, u, w, p);
  };

  // Distributed solver + matching IC on the owned block.
  sgn::SGN1DMpi mpi(P, ne, L, par, MPI_COMM_WORLD, bump);
  const int Nl = mpi.nLocalDof();
  const int a = mpi.partition().a;
  auto buildMpiState = [&]() {
    const auto xl = mpi.xCoordsLocal();
    std::vector<double> h(Nl), u(Nl), w(Nl), p(Nl);
    for (int i = 0; i < Nl; ++i) { double q[4]; ic(xl[i], q); h[i]=q[0]; u[i]=q[1]; w[i]=q[2]; p[i]=q[3]; }
    return mpi.makeStateLocal(h, u, w, p);
  };

  // Map an owned (var v, elem e, node j) to the serial global flat index.
  auto serIdx = [&](int v, int e, int j) {
    const int gElem = a + e;
    return v * Ng + gElem * np + j;
  };

  // --- Gate 1: RHS bitwise parity ------------------------------------------
  {
    auto Us = buildSerialState();
    auto Um = buildMpiState();
    sgn::State Rs, Rm;
    ser.rhs(Us, Rs);
    mpi.rhs(Um, Rm);
    double localMax = 0.0;
    for (int v = 0; v < sgn::NVAR; ++v)
      for (int e = 0; e < mpi.nOwnedElem(); ++e)
        for (int j = 0; j < np; ++j)
          localMax = std::fmax(localMax,
              std::fabs(Rm[v * Nl + e * np + j] - Rs[serIdx(v, e, j)]));
    double globMax = 0.0;
    MPI_Reduce(&localMax, &globMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    gate("RHS: MPI owned == serial (bitwise)", globMax, 0.0, /*exact=*/true);
  }

  // --- Gate 2: fixed-step RK4 trajectory bitwise ---------------------------
  {
    auto Us = buildSerialState();
    auto Um = buildMpiState();
    const double dt = 0.2 * ser.cflTimeStep(Us, 0.5);
    for (int s = 0; s < 20; ++s) {
      sgn::rk4Step(ser.rhsFn(), 0.0, dt, Us);
      mpi.rk4Step(Um, dt);
    }
    double localMax = 0.0;
    for (int v = 0; v < sgn::NVAR; ++v)
      for (int e = 0; e < mpi.nOwnedElem(); ++e)
        for (int j = 0; j < np; ++j)
          localMax = std::fmax(localMax,
              std::fabs(Um[v * Nl + e * np + j] - Us[serIdx(v, e, j)]));
    double globMax = 0.0;
    MPI_Reduce(&localMax, &globMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    gate("RK4 20 steps: MPI owned == serial (bitwise)", globMax, 0.0, true);
  }

  // --- Gate 3: adaptive run + global conservation --------------------------
  {
    auto Us = buildSerialState();
    auto Um = buildMpiState();
    const double M0 = mpi.mass(Um);
    sgn::AdaptiveRKOptions opt; opt.atol = 1e-10; opt.rtol = 1e-8;
    const double dt0 = mpi.cflTimeStep(Um, 0.5);
    const auto st = mpi.integrateAdaptive(Um, 0.0, 0.5, dt0, opt);
    const auto stS = sgn::integrateAdaptive(ser.rhsFn(), Us, 0.0, 0.5,
                                            ser.cflTimeStep(Us, 0.5), opt);

    // finite + finished on all ranks (reduce a flag).
    int localBad = st.aborted ? 1 : 0;
    for (double vv : Um) if (!std::isfinite(vv)) { localBad = 1; break; }
    int globBad = 0;
    MPI_Reduce(&localBad, &globBad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    gate("adaptive: finite & finished (all ranks)", (double)globBad, 0.5, false);

    const double dM = std::fabs(mpi.mass(Um) - M0) / std::fabs(M0);
    gate("adaptive: global mass conserved (machine)", dM, 1e-13, false);

    // trajectory vs serial (round-off; norm reduction order differs).
    double localMax = 0.0, localScale = 0.0;
    for (int v = 0; v < sgn::NVAR; ++v)
      for (int e = 0; e < mpi.nOwnedElem(); ++e)
        for (int j = 0; j < np; ++j) {
          localMax = std::fmax(localMax,
              std::fabs(Um[v * Nl + e * np + j] - Us[serIdx(v, e, j)]));
          localScale = std::fmax(localScale, std::fabs(Us[serIdx(v, e, j)]));
        }
    double gMax = 0.0, gScale = 0.0;
    MPI_Reduce(&localMax, &gMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localScale, &gScale, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
      gate("adaptive: trajectory == serial (rel, round-off)", gMax / gScale,
           1e-10, false);
    if (rank == 0)
      std::printf("       [info] MPI steps acc=%ld rej=%ld | serial acc=%ld rej=%ld\n",
                  st.accepted, st.rejected, stS.accepted, stS.rejected);
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
