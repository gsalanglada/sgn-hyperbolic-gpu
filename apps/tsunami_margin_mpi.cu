// =============================================================================
// tsunami_margin_mpi.cu -- basin-scale dispersive tsunami (multi-GPU)
// =============================================================================
//
// Dispersive tsunami propagation across an ocean basin onto a continental
// margin: a capability case for the distributed multi-GPU solver (paper
// Section 7.3).
//
//   usage: srun -n <ranks> ./build/tsunami_margin_mpi <Nx> <Ny> <P> <tEnd_h>
//              [samples] [vtkEvery] [vtkCells] [alpha] [wSlopeKm]
//     tEnd_h    simulated time in hours
//     samples   measurement samples over [0, tEnd] (default 120)
//     vtkEvery  0 = no VTK output (default); k = one frame every k samples
//     vtkCells  max raster cells per direction of a VTK frame (default 800)
//     alpha     relaxation parameter (default 3); 0 = shallow-water baseline
//     wSlopeKm  continental-slope half-width in km (default 50)
//
// Frames are coarsened to at most vtkCells per direction, so a frame stays a
// few MB whatever the mesh.
//
// CALL CHAIN
//   main -> SGN2DMpi (setup, bathymetry) -> gauge ownership (MPI_MINLOC)
//        -> SGN2DCudaMpi (device) -> loop over samples:
//             integrateAdaptive -> download -> shelf / deep / global maxima
//             (MPI_MAXLOC) -> gauges (MPI_SUM of one-hot vectors)
//             -> optional writeVtk2DMpi (src/io/vtk2d_mpi.h)
//        -> summary (shoaling ratio vs Green's law, cost, single-GPU estimate)
//
// WHY THIS CASE. Long-wave dispersion is weak locally but accumulates over
// propagation distance: the leading crest sheds energy into a trailing wave
// train, and by the time a tsunami has crossed an ocean basin the difference
// from a non-dispersive prediction is no longer small. Capturing that requires
// (i) a domain thousands of kilometres across, (ii) a resolution fine enough
// that numerical dispersion stays well below the physical dispersion one is
// trying to model, and (iii) integration over hours of physical time. The three
// together put the problem out of practical reach of a single device: the paper
// runs 6.2e7 DOF (1000x500, 16 GPUs) and 2.5e8 DOF (2000x1000, 64 GPUs); the
// latter would take about 6 h on one GPU at the measured throughput, against
// about 254 s on 64.
//
// GEOMETRY. A deep basin flanked by continental slopes and shelves on both
// sides. Making the basin symmetric is what lets the periodic solver model an
// open ocean honestly: the westward and eastward halves of the wave each run
// onto their own margin, so nothing wraps around into the measurement region.
// The eastern margin is deliberately curved, which refracts the incoming front
// and focuses energy on the promontory -- the two-dimensional effect that makes
// this a 2D problem rather than a stack of independent 1D ones.
//
// SOURCE. An elliptical Gaussian sea-surface displacement, the standard
// idealization of instantaneous co-seismic seafloor uplift, with the fluid
// initially at rest.
//
// SCOPE. The bathymetry is analytic rather than surveyed, and the run stops at
// the shelf: without wetting and drying the solver cannot take the wave to the
// shoreline. This is the usual division of labour in operational practice,
// where a dispersive propagation model hands off to a dedicated inundation
// model at a depth contour. Swapping in surveyed bathymetry only requires
// replacing the depth function below.
//
// Output (rank 0):
//   results/tsunami_margin_<Nx>x<Ny>_n<ranks>[_swe]_w<wSlopeKm>.csv
//       t_s, etaMaxShelf, etaMaxGlobal, g1..g5   (gauge elevations, metres)
//   results/tsunami_<Nx>x<Ny>[_swe]_<frame>.vtk  (only if vtkEvery > 0)
//   stdout summary line, see the header printed at the end.
// =============================================================================

#include "../src/cuda/sgn2d_cuda_mpi.h"
#include "../src/io/vtk2d_mpi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

namespace { constexpr double kPi = 3.14159265358979323846; }
namespace { constexpr int kGauge = 5; }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // srun redirects stdout to a pipe, which makes it block-buffered: without
  // this the progress lines of a long run only appear when it finishes (or
  // never, if it is killed). MPI writes its own warnings to unbuffered stderr,
  // which is why those show up immediately and these otherwise would not.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const int Nx = (argc > 1) ? std::atoi(argv[1]) : 1000;
  const int Ny = (argc > 2) ? std::atoi(argv[2]) : 500;
  const int P = (argc > 3) ? std::atoi(argv[3]) : 4;
  const double tHours = (argc > 4) ? std::atof(argv[4]) : 2.0;
  const int nSample = (argc > 5) ? std::atoi(argv[5]) : 120;
  const int vtkEvery = (argc > 6) ? std::atoi(argv[6]) : 0;
  const int vtkCells = (argc > 7) ? std::atoi(argv[7]) : 800;
  // alpha = 0 switches the relaxation off and reduces the system to
  // NON-DISPERSIVE shallow water on the same mesh, with the same code, the same
  // bathymetry and the same initial condition. With c = 0 the pressure equation
  // becomes d_t(hp) + d_x(hup) = 0 and the vertical-momentum equation
  // d_t(hw) + d_x(huw) = gamma p; since the run starts from w = p = 0 both stay
  // identically zero, the momentum flux loses its hp term, and what remains is
  // the shallow-water system. This is the baseline that isolates dispersion
  // from the non-adiabatic slope and the two-dimensional refraction, which a
  // shallow-water model reproduces just as well.
  const double alphaArg = (argc > 8) ? std::atof(argv[8]) : 3.0;
  const bool swe = (alphaArg == 0.0);
  // Continental-slope half-width in km. Narrower means a more abrupt transition
  // and stronger reflection; wider is closer to the adiabatic limit Green's law
  // assumes. Varying it separates the reflection contribution to the shoaling
  // deficit from the dispersive one, since dispersion accumulates mainly during
  // the deep-water transit rather than on the slope itself.
  const double wSlopeKm = (argc > 9) ? std::atof(argv[9]) : 50.0;
  const double tEnd = tHours * 3600.0;   // simulated time [s]

  {   // one GPU per rank (node-local; see SGN2DCudaMpi constructor)
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

  // ---- physical configuration ---------------------------------------------
  const double g = 9.81, gamma = 1.5;
  const double alpha = alphaArg;
  const double Lx = 4.0e6, Ly = 2.0e6;          // 4000 km x 2000 km
  const double hDeep = 4000.0, hShelf = 150.0;  // abyssal plain, shelf
  const double xW = 1.3e6, xE = 2.7e6;          // slope centres (west, east)
  const double wSlope = wSlopeKm * 1.0e3;       // continental slope width
  const double bend = 2.0e5;                    // curvature of the east margin
  const double xSrc = 2.0e6, ySrc = 1.0e6;      // source centre
  const double aSrc = 2.0;                      // uplift amplitude [m]
  const double sxSrc = 5.0e4, sySrc = 2.0e5;    // source semi-axes (x,y)

  // c is set from the greatest depth so that the relaxation is never under-
  // resolved anywhere in the domain; over the shelf it is correspondingly
  // stiffer, which moves the local solution closer to the SGN limit.
  sgn2d::Params par;
  par.g = g; par.gamma = gamma; par.c = alpha * std::sqrt(g * hDeep);

  // Depth: deep between the two slopes, shelf outside. The eastern slope is
  // displaced by a smooth promontory so the front refracts in two dimensions.
  auto depth = [=](double x, double y) {
    const double xEy = xE + bend * std::cos(2.0 * kPi * (y - ySrc) / Ly);
    const double s = 0.5 * (std::tanh((x - xW) / wSlope) -
                            std::tanh((x - xEy) / wSlope));
    return hShelf + (hDeep - hShelf) * s;
  };

  sgn::SGN2DMpi host(P, Nx, Ny, Lx, Ly, par, MPI_COMM_WORLD, depth);
  const int Nl = host.nLocalDof();
  std::vector<double> x, y;
  host.nodeCoordsLocal(x, y);

  std::vector<double> H(Nl);
  for (int i = 0; i < Nl; ++i) H[i] = depth(x[i], y[i]);

  // ---- initial condition: co-seismic uplift, fluid at rest ----------------
  std::vector<double> h(Nl), u(Nl, 0.0), v(Nl, 0.0), w(Nl, 0.0), p(Nl, 0.0);
  for (int i = 0; i < Nl; ++i) {
    const double dx = (x[i] - xSrc) / sxSrc, dy = (y[i] - ySrc) / sySrc;
    h[i] = H[i] + aSrc * std::exp(-(dx * dx + dy * dy));
  }
  auto U = host.makeStateLocal(h, u, v, w, p);
  const double M0 = host.mass(U);

  // ---- gauges along the eastern shelf edge --------------------------------
  // Each gauge is owned by exactly one rank (the one holding the nearest node),
  // decided once by a MINLOC on distance. Sampling then sums a vector whose
  // entries are zero everywhere except on the owner, so the reduction is exact
  // regardless of rank count.
  double gx[kGauge], gy[kGauge];
  for (int k = 0; k < kGauge; ++k) {
    gy[k] = ySrc + (k - 2) * 0.18 * Ly;
    gx[k] = xE + bend * std::cos(2.0 * kPi * (gy[k] - ySrc) / Ly) + 1.2e5;
  }
  int gIdx[kGauge];
  int gOwner[kGauge];
  for (int k = 0; k < kGauge; ++k) {
    double best = 1e300;
    int bi = -1;
    for (int i = 0; i < Nl; ++i) {
      const double dx = x[i] - gx[k], dy = y[i] - gy[k];
      const double d2 = dx * dx + dy * dy;
      if (d2 < best) { best = d2; bi = i; }
    }
    struct { double d; int r; } mine{best, rank}, red;
    MPI_Allreduce(&mine, &red, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    gIdx[k] = bi;
    gOwner[k] = red.r;
  }

  sgn::SGN2DCudaMpi dev(host, MPI_COMM_WORLD);
  dev.upload(U);

  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-8;
  opt.rtol = 1e-6;
  double dt = host.cflTimeStep(U, 0.5);

  std::FILE* f = nullptr;
  if (rank == 0) {
    char name[192];
    std::snprintf(name, sizeof name,
                  "results/tsunami_margin_%dx%d_n%d%s_w%03.0f.csv",
                  Nx, Ny, size, swe ? "_swe" : "", wSlopeKm);
    f = std::fopen(name, "w");
    std::fprintf(f, "t_s,etaMaxShelf,etaMaxGlobal");
    for (int k = 0; k < kGauge; ++k) std::fprintf(f, ",g%d", k + 1);
    std::fprintf(f, "\n");
    const double dof = 5.0 * (double)Nx * Ny * (P + 1) * (P + 1);
    std::printf("# dispersive tsunami over a continental margin\n");
    std::printf("# %d ranks, %dx%d elements, P=%d, %.3g DOF, dx=%.0f m,"
                " tEnd=%.2f h\n", size, Nx, Ny, P, dof, Lx / (Nx * (P + 1)),
                tEnd / 3600.0);
    std::printf("# basin %.0fx%.0f km, depth %.0f m -> shelf %.0f m,"
                " source %.1f m\n", Lx / 1e3, Ly / 1e3, hDeep, hShelf, aSrc);
    std::printf("# model: %s\n",
                swe ? "SHALLOW WATER (alpha = 0, non-dispersive baseline)"
                    : "hyperbolic SGN (alpha = 3)");
    std::printf("# continental slope half-width: %.0f km\n", wSlopeKm);
    std::printf("#  sample     t[min]   etaMaxShelf[m]  etaMaxGlobal[m]\n");
  }

  double etaShelfPeak = -1e300, xPk = 0.0, yPk = 0.0, tPk = 0.0;
  double etaIncident = -1e300;   // amplitude carried to the margin
  sgn::State Us;
  long acc = 0, rej = 0;
  double vtkTime = 0.0;
  const double t0Wall = MPI_Wtime();

  for (int s = 0; s <= nSample; ++s) {
    if (s > 0) {
      const double ta = (s - 1) * tEnd / nSample;
      const double tb = s * tEnd / nSample;
      const auto st = dev.integrateAdaptive(ta, tb, dt, opt);
      if (st.aborted) {
        if (rank == 0) std::fprintf(stderr, "ABORT at t=%.3f s\n", st.tReached);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      dt = st.dtLast;
      acc += st.accepted;
      rej += st.rejected;
    }
    const double t = s * tEnd / nSample;
    dev.download(Us);

    // Shelf window: east of the (curved) slope centre. Deep window: a band of
    // undisturbed depth just upstream of it, which measures the amplitude the
    // wave actually carries to the margin. Comparing the shelf peak with the
    // source amplitude would be meaningless, because a localized source spreads
    // geometrically on the way; the shoaling ratio has to be referred to the
    // incident wave, and this is where it is measured.
    double shelfLoc = -1e300, globLoc = -1e300, deepLoc = -1e300;
    int iBest = -1;
    for (int i = 0; i < Nl; ++i) {
      const double eta = Us[i] - H[i];
      if (eta > globLoc) globLoc = eta;
      const double xEy = xE + bend * std::cos(2.0 * kPi * (y[i] - ySrc) / Ly);
      if (x[i] > xEy && eta > shelfLoc) { shelfLoc = eta; iBest = i; }
      if (x[i] > xEy - 5.0e5 && x[i] < xEy - 2.5e5 && eta > deepLoc)
        deepLoc = eta;
    }
    struct { double v; int r; } sLoc{shelfLoc, rank}, gLoc{globLoc, rank},
                                dLoc{deepLoc, rank}, sRed, gRed, dRed;
    MPI_Allreduce(&sLoc, &sRed, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Allreduce(&gLoc, &gRed, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Allreduce(&dLoc, &dRed, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    if (dRed.v > etaIncident) etaIncident = dRed.v;

    double gv[kGauge], gAll[kGauge];
    for (int k = 0; k < kGauge; ++k)
      gv[k] = (rank == gOwner[k] && gIdx[k] >= 0)
                  ? Us[gIdx[k]] - H[gIdx[k]] : 0.0;
    MPI_Allreduce(gv, gAll, kGauge, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (sRed.v > etaShelfPeak) {
      etaShelfPeak = sRed.v;
      tPk = t;
      double xy[2] = {0.0, 0.0};
      if (rank == sRed.r && iBest >= 0) { xy[0] = x[iBest]; xy[1] = y[iBest]; }
      MPI_Bcast(xy, 2, MPI_DOUBLE, sRed.r, MPI_COMM_WORLD);
      xPk = xy[0]; yPk = xy[1];
    }

    if (rank == 0) {
      std::fprintf(f, "%.3f,%.8e,%.8e", t, sRed.v, gRed.v);
      for (int k = 0; k < kGauge; ++k) std::fprintf(f, ",%.8e", gAll[k]);
      std::fprintf(f, "\n");
      if (s % 10 == 0)
        std::printf("  %6d/%d  %9.2f   %12.4f   %14.4f   [%.0f s elapsed]\n",
                    s, nSample, t / 60.0, sRed.v, gRed.v,
                    MPI_Wtime() - t0Wall);
    }

    // Visualization frames. The state is already on the host from the sampling
    // download above, so a frame costs one gather and no extra device traffic.
    // The time spent here is accumulated separately so that the reported
    // compute time is not polluted by output, and one run can serve both as the
    // timing measurement and as the source of the animation.
    if (vtkEvery > 0 && s % vtkEvery == 0) {
      MPI_Barrier(MPI_COMM_WORLD);
      const double tv = MPI_Wtime();
      char vname[192];
      std::snprintf(vname, sizeof vname, "results/tsunami_%dx%d%s_%04d.vtk",
                    Nx, Ny, swe ? "_swe" : "", s / vtkEvery);
      sgn::writeVtk2DMpi(host, Us, vname, MPI_COMM_WORLD, vtkCells);
      MPI_Barrier(MPI_COMM_WORLD);
      vtkTime += MPI_Wtime() - tv;
    }
  }

  const double wall = MPI_Wtime() - t0Wall;
  dev.download(Us);
  const double dM = std::fabs(host.mass(Us) - M0) / std::fabs(M0);

  if (rank == 0) {
    std::fclose(f);
    const double dof = 5.0 * (double)Nx * Ny * (P + 1) * (P + 1);
    // Green's law gives the non-dispersive shoaling factor for a long wave
    // climbing from hDeep to hShelf. It is a statement about the RATIO between
    // the incident and the shoaled amplitude, so it is referred to the wave
    // measured arriving at the margin, not to the source: a localized source
    // spreads geometrically and reaches the margin with far less than aSrc.
    // NOTE: the printed line below attributes the excess over Green's law to
    // "2D refraction and dispersion". The alpha = 0 baseline run of the paper
    // shows most of the deficit is already present in shallow water (finite
    // slope width and refraction); only a few percent is dispersive.
    const double greenRatio = std::pow(hDeep / hShelf, 0.25);
    std::printf("#\n# shelf peak eta = %.4f m at (x,y)=(%.1f,%.1f) km,"
                " t=%.1f min\n", etaShelfPeak, xPk / 1e3, yPk / 1e3,
                tPk / 60.0);
    std::printf("# incident amplitude at the margin = %.4f m"
                " (source was %.2f m; the difference is 2D spreading)\n",
                etaIncident, aSrc);
    std::printf("# shoaling ratio measured %.3f vs Green's law %.3f"
                "  -> excess %.1f%% from 2D refraction and dispersion\n",
                etaShelfPeak / etaIncident, greenRatio,
                100.0 * (etaShelfPeak / etaIncident / greenRatio - 1.0));
    if (tPk > 0.98 * tEnd)
      std::printf("# WARNING: the peak is at the end of the window --"
                  " the run is too short, extend tEnd\n");
    // Single-GPU extrapolation from the measured per-device throughput, for the
    // capability statement: how long this run would take without the cluster.
    const double rhs = 3.0 * (double)(acc + rej);
    const double gdofs = dof * rhs / (wall - vtkTime) / 1e9;
    std::printf("# compute %.1f s (+%.1f s output), %ld accepted steps,"
                " %.1f GDOF/s aggregate, %.2f GDOF/s per GPU\n",
                wall - vtkTime, vtkTime, acc, gdofs, gdofs / size);
    std::printf("# same work on one GPU at the measured 3.09 GDOF/s:"
                " %.1f h\n", dof * rhs / 3.09e9 / 3600.0);
    std::printf("ranks,Nx,Ny,P,dof,dx_m,tEnd_h,etaShelfPeak,xPk_km,yPk_km,"
                "tPk_min,etaIncident,shoalRatio,greenRatio,acc,rej,wall_s,"
                "vtk_s,gdof_s,massDrift\n");
    std::printf("%d,%d,%d,%d,%.0f,%.1f,%.3f,%.6f,%.3f,%.3f,%.3f,%.6f,%.4f,"
                "%.4f,%ld,%ld,%.3f,%.3f,%.2f,%.2e\n",
                size, Nx, Ny, P, dof, Lx / (Nx * (P + 1)), tEnd / 3600.0,
                etaShelfPeak, xPk / 1e3, yPk / 1e3, tPk / 60.0, etaIncident,
                etaShelfPeak / etaIncident, greenRatio, acc, rej,
                wall - vtkTime, vtkTime, gdofs, dM);
  }

  MPI_Finalize();
  return 0;
}
