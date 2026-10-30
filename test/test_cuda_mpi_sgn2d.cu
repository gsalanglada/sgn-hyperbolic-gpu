// =============================================================================
// test_cuda_mpi_sgn2d.cu -- multi-GPU solver vs single GPU / CPU-MPI (src/cuda/sgn2d_cuda_mpi.*)
// =============================================================================
//
// HOW TO RUN   source setup_env.src && make run-cuda-mpi
//              (mpirun -n 2 and -n 4, one GPU per rank, on a GPU node)
// PASS         rank 0 evaluates every gate, prints "ok"/"FAIL" and "ALL TESTS PASSED"
//              when all hold; the exit code (0 or 1) is broadcast so every rank returns it.
// =============================================================================
//
// Multi-GPU gates: the final merge (CUDA kernels + CUDA-aware MPI halos).
// Run under mpirun with one GPU per rank.
//   1) Distributed GPU RHS == CPU-MPI RHS on owned elements (round-off only;
//      the CPU-MPI driver is itself bitwise-serial).
//   2) Distributed GPU RHS == single-GPU full-mesh RHS on owned elements
//      (same device arithmetic -> expected ~0; printed, gated at 1e-13 rel).
//   3) Adaptive multi-GPU run: finishes, global mass machine-conserved,
//      trajectory matches the CPU-MPI adaptive run (tolerance-limited).
//
// Build/run on an ACC node with N GPUs:
//   source setup_env.src && export UCX_TLS=^cuda_ipc
//   make run-cuda-mpi          (mpirun -n 2 and -n 4)

#include "../src/cuda/sgn2d_cuda_mpi.h"
#include "../src/cuda/sgn2d_cuda.h"
#include "../src/solver/sgn2d.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // One GPU per rank (node-local).
  {
    MPI_Comm shm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shm);
    int lrank = 0;
    MPI_Comm_rank(shm, &lrank);
    MPI_Comm_free(&shm);
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    cudaSetDevice(lrank % ndev);
  }

  int failures = 0;
  auto gate = [&](const char* what, double v, double tol) {
    if (rank == 0) {
      const bool ok = v < tol;
      std::printf("  %s: %-52s  %.3e\n", ok ? "ok  " : "FAIL", what, v);
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
  auto ic = [&](double x, double y, double* q) {
    q[0] = bump2d(x, y) + 0.03 * std::sin(kx * x) * std::cos(ky * y);
    q[1] = 0.08 * std::cos(kx * x);
    q[2] = -0.05 * std::sin(ky * y);
    q[3] = 0.0;
    q[4] = 0.0;
  };

  if (rank == 0)
    std::printf("=== multi-GPU == single-GPU == CPU (size=%d) ===\n\n", size);

  sgn::SGN2DMpi hostMpi(P, Nx, Ny, Lx, Ly, par, MPI_COMM_WORLD, bump2d);
  const int Nl = hostMpi.nLocalDof();
  const int np = P + 1, npp = np * np;
  const int a0 = hostMpi.partition().a, w0 = hostMpi.partition().w;

  auto buildMpiState = [&]() {
    std::vector<double> x, y;
    hostMpi.nodeCoordsLocal(x, y);
    std::vector<double> h(Nl), u(Nl), v(Nl), w(Nl), p(Nl);
    for (int i = 0; i < Nl; ++i) {
      double q[5]; ic(x[i], y[i], q);
      h[i]=q[0]; u[i]=q[1]; v[i]=q[2]; w[i]=q[3]; p[i]=q[4];
    }
    return hostMpi.makeStateLocal(h, u, v, w, p);
  };

  // --- Gate 1: distributed GPU RHS vs CPU-MPI RHS ----------------------------
  {
    auto Um = buildMpiState();
    sgn::State Rc;
    hostMpi.rhs(Um, Rc);

    sgn::SGN2DCudaMpi dev(hostMpi, MPI_COMM_WORLD);
    dev.upload(Um);
    sgn::State Rg;
    dev.rhsOnce(Rg);

    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < Rc.size(); ++i) {
      diff = std::fmax(diff, std::fabs(Rc[i] - Rg[i]));
      scale = std::fmax(scale, std::fabs(Rc[i]));
    }
    double gd = 0.0, gs = 0.0;
    MPI_Reduce(&diff, &gd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&scale, &gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
      gate("multi-GPU RHS == CPU-MPI (rel)", gd / gs, 1e-12);
  }

  // --- Gate 2: distributed GPU RHS vs single-GPU full mesh -------------------
  {
    // Full-mesh serial + single-GPU reference, redundantly on every rank.
    sgn::SGN2D serFull(P, Nx, Ny, Lx, Ly, par, bump2d);
    const int Nd = serFull.nDof();
    std::vector<double> x, y;
    serFull.nodeCoords(x, y);
    std::vector<double> h(Nd), u(Nd), v(Nd), w(Nd), p(Nd);
    for (int i = 0; i < Nd; ++i) {
      double q[5]; ic(x[i], y[i], q);
      h[i]=q[0]; u[i]=q[1]; v[i]=q[2]; w[i]=q[3]; p[i]=q[4];
    }
    auto Ufull = serFull.makeState(h, u, v, w, p);
    sgn::SGN2DCuda devFull(serFull);
    devFull.upload(Ufull);
    sgn::State Rfull;
    devFull.rhsOnce(Rfull);

    auto Um = buildMpiState();
    sgn::SGN2DCudaMpi dev(hostMpi, MPI_COMM_WORLD);
    dev.upload(Um);
    sgn::State Rg;
    dev.rhsOnce(Rg);

    double diff = 0.0, scale = 0.0;
    for (int var = 0; var < sgn2d::NVAR; ++var)
      for (int e = 0; e < hostMpi.nOwnedElem(); ++e)
        for (int loc = 0; loc < npp; ++loc) {
          const int lx = e % w0, ley = e / w0;
          const int gElem = (a0 + lx) + Nx * ley;
          const double rf = Rfull[(size_t)var * Nd + (size_t)gElem * npp + loc];
          diff = std::fmax(diff, std::fabs(
              Rg[(size_t)var * Nl + (size_t)e * npp + loc] - rf));
          scale = std::fmax(scale, std::fabs(rf));
        }
    double gd = 0.0, gs = 0.0;
    MPI_Reduce(&diff, &gd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&scale, &gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
      gate("multi-GPU RHS == single-GPU (rel; expect ~0)", gd / gs, 1e-13);
  }

  // --- Gate 3: adaptive multi-GPU run ----------------------------------------
  {
    auto Um = buildMpiState();
    auto Uc = Um;
    const double M0 = hostMpi.mass(Um);
    sgn::AdaptiveRKOptions opt;
    opt.atol = 1e-10; opt.rtol = 1e-8;
    const double dt0 = hostMpi.cflTimeStep(Um, 0.5);

    const auto stC = hostMpi.integrateAdaptive(Uc, 0.0, 0.3, dt0, opt);

    sgn::SGN2DCudaMpi dev(hostMpi, MPI_COMM_WORLD);
    dev.upload(Um);
    const auto stG = dev.integrateAdaptive(0.0, 0.3, dt0, opt);
    sgn::State Ug;
    dev.download(Ug);

    int localBad = stG.aborted ? 1 : 0;
    for (double vv : Ug) if (!std::isfinite(vv)) { localBad = 1; break; }
    int globBad = 0;
    MPI_Reduce(&localBad, &globBad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
      gate("multi-GPU adaptive: finite & finished", (double)globBad, 0.5);

    const double dM = std::fabs(hostMpi.mass(Ug) - M0) / std::fabs(M0);
    gate("multi-GPU adaptive: global mass (machine)", dM, 1e-13);

    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < Ug.size(); ++i) {
      diff = std::fmax(diff, std::fabs(Ug[i] - Uc[i]));
      scale = std::fmax(scale, std::fabs(Uc[i]));
    }
    double gd = 0.0, gs = 0.0;
    MPI_Reduce(&diff, &gd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&scale, &gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      gate("multi-GPU trajectory == CPU-MPI (rel)", gd / gs, 1e-6);
      std::printf("       [info] steps: CPU-MPI acc=%ld rej=%ld | multi-GPU acc=%ld rej=%ld\n",
                  stC.accepted, stC.rejected, stG.accepted, stG.rejected);
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
