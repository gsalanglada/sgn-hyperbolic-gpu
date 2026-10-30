// =============================================================================
// sgn_bench.cu -- multi-GPU strong / weak scaling benchmark (2D solver)
// =============================================================================
//
//   usage:  srun -n <ranks> ./build/sgn_bench <Nx> <Ny> <P> <tEnd> [cfl] [reps]
//     Nx, Ny  global elements in x and y (Nx is split across ranks)
//     P       polynomial degree
//     tEnd    simulated time [s]
//     cfl     CFL number of the initial step guess (default 0.5)
//     reps    timed repetitions; the median wall time is reported (default 1)
//
// One GPU per rank (slab in x). Runs the PRODUCTION adaptive loop (device-
// resident BS3(2) + one global Allreduce pair per step) on a smooth wave over
// a Gaussian bump, with no I/O inside the timed region. At a fixed mesh the
// accepted-step count has been observed to be identical for every rank count,
// so strong-scaling wall times are directly comparable; for weak scaling
// compare time per step.
//
// CALL CHAIN
//   main
//    |- SGN2DMpi (src/parallel)       partition + host setup + initial state
//    |- SGN2DCudaMpi (src/cuda)       device buffers, upload
//    |- integrateAdaptive (warm-up)   untimed
//    `- integrateAdaptive x reps      timed: rhsDevice -> kernels + halo exchange
//
// Rank 0 prints one comment line and one CSV record:
//   ranks,Nx,Ny,P,dof,acc,rej,rhsEvals,wall_med_s,ms_per_step,gdof_per_rhs_s
// =============================================================================

#include "../src/cuda/sgn2d_cuda_mpi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main(int argc, char** argv) {
  // --- MPI start-up: every rank runs this same main ---------------------------
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // this process' id, 0..size-1
  MPI_Comm_size(MPI_COMM_WORLD, &size);   // number of processes (= GPUs)

  // --- command line ---------------------------------------------------------
  const int Nx = (argc > 1) ? std::atoi(argv[1]) : 256;
  const int Ny = (argc > 2) ? std::atoi(argv[2]) : 128;
  const int P = (argc > 3) ? std::atoi(argv[3]) : 4;
  const double tEnd = (argc > 4) ? std::atof(argv[4]) : 0.05;
  const double cfl = (argc > 5) ? std::atof(argv[5]) : 0.5;
  const int reps = (argc > 6) ? std::atoi(argv[6]) : 1; // timed repetitions

  // One GPU per rank (node-local). SGN2DCudaMpi selects the device the same way
  // in its constructor; doing it here as well only makes it explicit.
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

  // --- physical and geometric setup -------------------------------------------
  const double g = 9.81, H0 = 1.0, gamma = 1.5, alpha = 3.0;  // SGN, c = 3 sqrt(g H0)
  const double Lx = 20.0, Ly = 10.0;                          // domain [m]
  const double kx = 2.0 * kPi / Lx, ky = 2.0 * kPi / Ly;      // one wavelength per period

  sgn2d::Params par;
  par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * H0);

  // still depth: 1 m with a 0.3 m Gaussian bump in the middle
  auto bump2d = [Lx, Ly](double x, double y) {
    const double dx = x - 0.5 * Lx, dy = y - 0.5 * Ly;
    return 1.0 - 0.3 * std::exp(-(dx * dx + dy * dy) / 4.0);
  };

  // host-side distributed solver: this rank's slab, bathymetry and node coords
  sgn::SGN2DMpi hostMpi(P, Nx, Ny, Lx, Ly, par, MPI_COMM_WORLD, bump2d);
  const int Nl = hostMpi.nLocalDof();
  std::vector<double> x, y;
  hostMpi.nodeCoordsLocal(x, y);
  // initial condition: smooth 2D perturbation of h and a sinusoidal u, w = p = 0
  std::vector<double> h(Nl), u(Nl), v(Nl), w(Nl), p(Nl, 0.0);
  for (int i = 0; i < Nl; ++i) {
    h[i] = bump2d(x[i], y[i]) +
           0.05 * std::sin(kx * x[i]) * std::cos(ky * y[i]);
    u[i] = 0.05 * std::cos(kx * x[i]);
    v[i] = 0.0;
    w[i] = 0.0;
  }
  auto U = hostMpi.makeStateLocal(h, u, v, w, p);
  const double M0 = hostMpi.mass(U);             // initial mass (collective)
  const double dt0 = hostMpi.cflTimeStep(U, cfl); // initial step guess (collective)

  // device solver built from the host setup; upload the initial state
  sgn::SGN2DCudaMpi dev(hostMpi, MPI_COMM_WORLD);
  dev.upload(U);

  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-8;
  opt.rtol = 1e-6;

  // Warm-up (JIT, allocator, MPI endpoints) outside the timed region.
  dev.integrateAdaptive(0.0, 5.0 * dt0, dt0, opt);

  // Timed region, repeated `reps` times; state re-uploaded before each run so
  // every repetition does identical work. A barrier + device sync brackets each
  // timing; we report the median and the [min,max] spread of the wall time.
  std::vector<double> walls;
  sgn::AdaptiveRKStats st;
  for (int r = 0; r < reps; ++r) {
    dev.upload(U);                       // reset state
    MPI_Barrier(MPI_COMM_WORLD);         // all ranks start the clock together
    const double t0 = MPI_Wtime();
    st = dev.integrateAdaptive(0.0, tEnd, dt0, opt);
    MPI_Barrier(MPI_COMM_WORLD);         // integrateAdaptive syncs on device
    walls.push_back(MPI_Wtime() - t0);
  }
  std::sort(walls.begin(), walls.end());
  const double wallMed = walls[walls.size() / 2];
  const double wallMin = walls.front(), wallMax = walls.back();

  // mass drift check on the final state
  sgn::State Uend;
  dev.download(Uend);
  const double dM = std::fabs(hostMpi.mass(Uend) - M0) / std::fabs(M0);

  // --- report (rank 0 only) -------------------------------------------------------
  if (rank == 0) {
    const double dofG = 5.0 * (double)Nx * Ny * (P + 1) * (P + 1);   // global DOF (NVAR = 5)
    const double msStep = 1e3 * wallMed / (double)st.accepted;
    // throughput = DOF x RHS evaluations / wall time, in billions per second
    const double gdofRhs = dofG * (double)st.rhsEvals / wallMed / 1e9; // per-RHS
    std::printf("# aborted=%d dM=%.2e dtLast=%.3e reps=%d wall[min,max]=[%.4f,%.4f]\n",
                st.aborted ? 1 : 0, dM, st.dtLast, reps, wallMin, wallMax);
    std::printf("ranks,Nx,Ny,P,dof,acc,rej,rhsEvals,wall_med_s,ms_per_step,"
                "gdof_per_rhs_s\n");
    std::printf("%d,%d,%d,%d,%.0f,%ld,%ld,%ld,%.4f,%.4f,%.2f\n", size, Nx, Ny,
                P, dofG, st.accepted, st.rejected, st.rhsEvals, wallMed, msStep,
                gdofRhs);
  }

  MPI_Finalize();
  return 0;
}
