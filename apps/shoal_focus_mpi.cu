// =============================================================================
// shoal_focus_mpi.cu -- dispersive focusing over a submerged shoal (multi-GPU)
// =============================================================================
//
// Dispersive focusing of a tsunami-like solitary wave over a submerged shoal,
// run on the DISTRIBUTED multi-GPU solver as a production computation
// (paper Section 7.1 and the geometry sweep of Section 7.2).
//
//   usage:  srun -n <ranks> ./build/shoal_focus_mpi <Nx> <Ny> <P> <tEnd>
//                                                  [samples] [dShoal] [rShoal]
//     samples  number of output samples over [0, tEnd] (default 100)
//
// dShoal is the crest elevation of the shoal (still depth over the crest is
// h0 - dShoal) and rShoal its Gaussian radius. Sweeping the pair maps how the
// focusing amplification depends on the obstacle geometry, which is the
// question a bathymetry-based mitigation study has to answer.
//
// Physics. A solitary wave of amplitude a over still depth h0 propagates in +x
// across a Gaussian submerged shoal. Over the shoal the crest slows and the
// wavefront refracts; behind it the converging front focuses and the peak
// elevation rises well above the incident amplitude, while dispersion sheds a
// trailing crescent. The quantity of interest is the focusing amplification
//
//     A = max over (x > x_shoal, y, t) of  eta(x,y,t) / a,
//
// i.e. how much higher than the incident wave the sea surface gets downstream
// of the shoal. This is the quantity that matters for coastal hazard, and it is
// resolution-sensitive: the focus is a small region with steep gradients, so a
// converged value needs meshes that are expensive on a single device.
//
// Reproducibility of the reported number. Unlike a sum, a maximum is
// associative AND exact in floating point, so MPI_Allreduce(MPI_MAXLOC)
// introduces no rank-count dependence of its own. This covers the REDUCTION
// only: the adaptive trajectories feeding it can still differ between rank
// counts (the error norm is a sum), and over ~1e5 steps they do, slightly. The
// agreement of A across rank counts is therefore an observed result, not a
// guarantee by construction (paper Sections 4 and 7.1).
//
// CALL CHAIN
//   main -> SGN2DMpi (setup) -> SGN2DCudaMpi (device) ->
//     loop over samples: integrateAdaptive -> download -> local max ->
//     MPI_Allreduce(MAXLOC) -> MPI_Bcast of the location
//
// Output (rank 0):
//   results/shoal_focus_<Nx>x<Ny>_n<ranks>_d<dShoal>_r<rShoal>.csv
//       t, etaMaxDown, etaMaxGlobal   (running values per sample)
//   stdout summary CSV:
//       ranks,Nx,Ny,P,dof,dShoal,rShoal,hCrest,A,xA,yA,tA,etaRms,acc,rej,wall_s,massDrift
// The rank count is in the file name so the same mesh can be run on different
// numbers of GPUs without clobbering the earlier result.
// =============================================================================

#include "../src/cuda/sgn2d_cuda_mpi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // --- command line ---------------------------------------------------------
  const int Nx = (argc > 1) ? std::atoi(argv[1]) : 384;
  const int Ny = (argc > 2) ? std::atoi(argv[2]) : 192;
  const int P = (argc > 3) ? std::atoi(argv[3]) : 4;
  const double tEnd = (argc > 4) ? std::atof(argv[4]) : 8.0;
  const int nSample = (argc > 5) ? std::atoi(argv[5]) : 100;
  const double dShoalArg = (argc > 6) ? std::atof(argv[6]) : 0.65;
  const double rShoalArg = (argc > 7) ? std::atof(argv[7]) : 3.0;

  // One GPU per rank (node-local), as in the benchmark driver.
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

  // Same configuration as the single-GPU showcase, so the two are comparable.
  const double g = 9.81, h0 = 1.0, a = 0.3, gamma = 1.5, alpha = 3.0; // a = wave amplitude
  const double Lx = 40.0, Ly = 20.0;           // domain [m]
  const double xShoal = 18.0, yShoal = 10.0;   // shoal centre
  const double rShoal = rShoalArg, dShoal = dShoalArg;
  const double x0 = 6.0;                       // initial crest position

  sgn2d::Params par;
  par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * h0);

  // still depth: h0 minus a Gaussian shoal of height dShoal and radius rShoal
  auto shoal = [=](double x, double y) {
    const double dx = x - xShoal, dy = y - yShoal;
    return h0 - dShoal * std::exp(-(dx * dx + dy * dy) /
                                  (2.0 * rShoal * rShoal));
  };

  // host-side distributed setup for this rank
  sgn::SGN2DMpi host(P, Nx, Ny, Lx, Ly, par, MPI_COMM_WORLD, shoal);
  const int Nl = host.nLocalDof();
  std::vector<double> x, y;
  host.nodeCoordsLocal(x, y);

  // Still depth at the owned nodes, needed to form eta = h - H.
  std::vector<double> H(Nl);
  for (int i = 0; i < Nl; ++i) H[i] = shoal(x[i], y[i]);

  // SGN solitary wave over depth h0, y-uniform, riding on the local still depth.
  //   cs   solitary-wave celerity sqrt(g (h0 + a))
  //   kap  inverse width of the sech^2 profile
  //   s    sech^2 profile value; hs, hx, hxx the flat-bottom depth and its
  //        x-derivatives, used for the travelling-wave identities of u, w, p
  const double cs = std::sqrt(g * (h0 + a));
  const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a)));
  std::vector<double> h(Nl), u(Nl), v(Nl, 0.0), w(Nl), p(Nl);
  for (int i = 0; i < Nl; ++i) {
    const double th = std::tanh(kap * (x[i] - x0));
    const double s = 1.0 - th * th;
    const double hs = h0 + a * s;
    const double hx = -2.0 * kap * a * s * th;
    const double hxx = 4.0 * kap * kap * a * s * th * th -
                       2.0 * kap * kap * a * s * s;
    h[i] = H[i] + a * s;
    u[i] = cs * (1.0 - h0 / hs);
    w[i] = -cs * h0 * hx / (2.0 * hs);
    p[i] = -(cs * h0 / gamma) *
           (-0.5 * cs * h0 * (hxx * hs - hx * hx) / (hs * hs));
  }
  auto U = host.makeStateLocal(h, u, v, w, p);
  const double M0 = host.mass(U);

  // device solver
  sgn::SGN2DCudaMpi dev(host, MPI_COMM_WORLD);
  dev.upload(U);

  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-8;
  opt.rtol = 1e-6;
  double dt = host.cflTimeStep(U, 0.5);   // initial step; later carried between samples

  // Running maximum of eta downstream of the shoal, with its location.
  // MPI_MAXLOC keeps value and owning rank together so the location is
  // unambiguous; the reduction adds no rank-count dependence of its own.
  struct { double val; int rank; } locMax{-1e300, rank}, globMax{-1e300, rank};
  double xA = 0.0, yA = 0.0, tA = 0.0;   // location and time of the record
  double etaMaxDownAll = -1e300;         // record downstream maximum so far

  std::FILE* f = nullptr;
  if (rank == 0) {
    char name[160];
    // Rank count is part of the name so that running the same mesh on a
    // different number of GPUs (the rank-independence check) does not
    // overwrite the previous result.
    std::snprintf(name, sizeof name,
                  "results/shoal_focus_%dx%d_n%d_d%.2f_r%.2f.csv", Nx, Ny,
                  size, dShoal, rShoal);
    f = std::fopen(name, "w");
    std::fprintf(f, "t,etaMaxDown,etaMaxGlobal\n");
    std::printf("# shoal focusing, %d ranks, %dx%d P=%d, DOF=%.1fM, tEnd=%.2f\n",
                size, Nx, Ny, P, 5.0 * (double)Nx * Ny * (P + 1) * (P + 1) / 1e6,
                tEnd);
    std::printf("# shoal: crest elevation %.3f m (still depth %.3f m over the"
                " crest), radius %.3f m\n", dShoal, h0 - dShoal, rShoal);
    std::printf("#   sample     t     etaMaxDown/a   etaMaxGlobal/a\n");
  }

  sgn::State Us;          // host copy of the owned state at each sample
  long acc = 0, rej = 0;  // accumulated step counts
  const double t0Wall = MPI_Wtime();

  // --- main loop: integrate to each sample time, then measure ----------------
  for (int s = 0; s <= nSample; ++s) {
    if (s > 0) {
      const double ta = (s - 1) * tEnd / nSample;
      const double tb = s * tEnd / nSample;
      const auto st = dev.integrateAdaptive(ta, tb, dt, opt);
      if (st.aborted) {
        if (rank == 0) std::fprintf(stderr, "ABORT at t=%.4f\n", st.tReached);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      dt = st.dtLast;     // continue with the last accepted step size
      acc += st.accepted;
      rej += st.rejected;
    }
    const double t = s * tEnd / nSample;

    dev.download(Us);
    // Local maxima of eta = h - H, downstream window and whole domain.
    double dLoc = -1e300, gLoc = -1e300;
    int iBest = -1;         // owned node index of the local downstream maximum
    for (int i = 0; i < Nl; ++i) {
      const double eta = Us[i] - H[i];          // component 0 of the state is h
      if (eta > gLoc) gLoc = eta;
      if (x[i] > xShoal && eta > dLoc) { dLoc = eta; iBest = i; }
    }
    locMax.val = dLoc; locMax.rank = rank;
    globMax.val = gLoc; globMax.rank = rank;

    // global maxima and the rank that owns each
    decltype(locMax) dRed, gRed;
    MPI_Allreduce(&locMax, &dRed, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&globMax, &gRed, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);

    if (dRed.val > etaMaxDownAll) {
      etaMaxDownAll = dRed.val;
      tA = t;
      // The owning rank supplies the location of the new record; every rank
      // takes part in the broadcast (collective).
      double xy[2] = {0.0, 0.0};
      if (rank == dRed.rank && iBest >= 0) { xy[0] = x[iBest]; xy[1] = y[iBest]; }
      MPI_Bcast(xy, 2, MPI_DOUBLE, dRed.rank, MPI_COMM_WORLD);
      xA = xy[0]; yA = xy[1];
    }

    if (rank == 0) {
      std::fprintf(f, "%.6f,%.8e,%.8e\n", t, dRed.val, gRed.val);
      if (s % 10 == 0)
        std::printf("  %6d  %7.3f   %12.5f   %12.5f\n", s, t, dRed.val / a,
                    gRed.val / a);
    }
  }

  const double wall = MPI_Wtime() - t0Wall;
  dev.download(Us);
  const double dM = std::fabs(host.mass(Us) - M0) / std::fabs(M0);

  // Field-level diagnostic: RMS of eta over the downstream window at the final
  // time. Unlike the peak amplification, this is dominated by the short
  // dispersive waves shed behind the shoal, so it is the quantity that actually
  // exercises the resolution. At fixed P the nodal distribution is self-similar
  // from mesh to mesh, so the plain nodal RMS is a legitimate scalar to compare
  // under refinement.
  //
  // NOTE: this reduction is a SUM, so unlike the maximum used for A it is not
  // bitwise rank-count independent. It is reported as a convergence diagnostic,
  // not as a reproducibility claim.
  double s2 = 0.0;
  long   cnt = 0;
  for (int i = 0; i < Nl; ++i) {
    if (x[i] > xShoal) {
      const double eta = Us[i] - H[i];
      s2 += eta * eta;
      ++cnt;
    }
  }
  double s2All = 0.0;
  long cntAll = 0;
  MPI_Allreduce(&s2, &s2All, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&cnt, &cntAll, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  const double etaRms = (cntAll > 0) ? std::sqrt(s2All / (double)cntAll) : 0.0;

  // --- summary (rank 0) ---------------------------------------------------------
  if (rank == 0) {
    std::fclose(f);
    const double dof = 5.0 * (double)Nx * Ny * (P + 1) * (P + 1);
    std::printf("#\n# focusing amplification A = eta_max/a = %.4f"
                "  at (x,y)=(%.3f,%.3f) t=%.3f\n",
                etaMaxDownAll / a, xA, yA, tA);
    // hCrest is the still depth over the shoal crest; the blockage ratio
    // dShoal/h0 and the radius are what a mitigation study would vary.
    std::printf("# downstream eta RMS at tEnd = %.8e  (field-level diagnostic:"
                " short dispersive waves, converges far more slowly than A)\n",
                etaRms);
    std::printf("ranks,Nx,Ny,P,dof,dShoal,rShoal,hCrest,A,xA,yA,tA,etaRms,"
                "acc,rej,wall_s,massDrift\n");
    std::printf("%d,%d,%d,%d,%.0f,%.4f,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f,%.8e,"
                "%ld,%ld,%.3f,%.2e\n",
                size, Nx, Ny, P, dof, dShoal, rShoal, h0 - dShoal,
                etaMaxDownAll / a, xA, yA, tA, etaRms, acc, rej, wall, dM);
  }

  MPI_Finalize();
  return 0;
}
