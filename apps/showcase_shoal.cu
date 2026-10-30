// =============================================================================
// showcase_shoal.cu -- single-GPU solitary wave over a shoal (VTK animation)
// =============================================================================
//
// Showcase: a solitary wave (tsunami-like) crossing a submerged circular
// shoal — dispersive focusing and scattering behind the shoal. Single GPU;
// writes a VTK frame series for ParaView (Warp By Scalar on "eta"). Source of
// the qualitative panels of the paper's first figure; the quantitative study of
// the same configuration is apps/shoal_focus_mpi.cu (multi-GPU).
//
// CALL CHAIN
//   main -> SGN2D (src/solver, host setup) -> SGN2DCuda (src/cuda, one GPU)
//        -> loop over frames: integrateAdaptive -> download -> writeVtk2D
//           (src/io/vtk2d.h)
//
//   usage:  ./build/showcase_shoal [Nx=384] [Ny=192] [P=4] [tEnd=8] [frames=40]
//   output: results/shoal_0000.vtk ... (element-mean fields; ~1 MB/frame)
//
// Setup: rest depth 1 m; Gaussian shoal to 0.35 m depth, radius ~3 m, centred
// at (18, 10) in a 40 x 20 m periodic box; SGN solitary wave of a = 0.3 m
// launched at x = 6 (y-uniform). As the crest passes the shoal it slows,
// refracts and focuses; behind the shoal a crescent of dispersive trailing
// waves develops — physics a shallow-water solver cannot produce.

#include "../src/cuda/sgn2d_cuda.h"
#include "../src/io/vtk2d.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// =============================================================================

int main(int argc, char** argv) {
  // --- command line ---------------------------------------------------------
  const int Nx = (argc > 1) ? std::atoi(argv[1]) : 384;
  const int Ny = (argc > 2) ? std::atoi(argv[2]) : 192;
  const int P = (argc > 3) ? std::atoi(argv[3]) : 4;
  const double tEnd = (argc > 4) ? std::atof(argv[4]) : 8.0;
  const int nFrames = (argc > 5) ? std::atoi(argv[5]) : 40;

  // --- physics: SGN (gamma = 3/2), c = 3 sqrt(g h0); a = solitary-wave amplitude
  const double g = 9.81, h0 = 1.0, a = 0.3, gamma = 1.5, alpha = 3.0;
  const double Lx = 40.0, Ly = 20.0;   // periodic box [m]

  sgn2d::Params par;
  par.g = g;
  par.gamma = gamma;
  par.c = alpha * std::sqrt(g * h0);

  // still depth: 1 m minus a Gaussian shoal (height 0.65 m, radius 3 m) at (18, 10)
  auto shoal = [=](double x, double y) {
    const double dx = x - 18.0, dy = y - 10.0;
    return h0 - 0.65 * std::exp(-(dx * dx + dy * dy) / (2.0 * 3.0 * 3.0));
  };

  sgn::SGN2D host(P, Nx, Ny, Lx, Ly, par, shoal);
  const int Nd = host.nDof();
  std::vector<double> x, y;
  host.nodeCoords(x, y);

  // SGN solitary wave over depth h0, y-uniform, launched at x0 = 6.
  const double cs = std::sqrt(g * (h0 + a));
  const double kap = std::sqrt(3.0 * a / (4.0 * h0 * h0 * (h0 + a)));
  const double x0 = 6.0;
  std::vector<double> h(Nd), u(Nd), v(Nd, 0.0), w(Nd), p(Nd);
  for (int i = 0; i < Nd; ++i) {
    const double t = std::tanh(kap * (x[i] - x0));
    const double s = 1.0 - t * t;
    const double hh = shoal(x[i], y[i]) - h0 + h0 + a * s; // = H(x,y) + a s
    // Note: the soliton profile rides on the LOCAL still depth; far from the
    // shoal H = h0 and this is the exact solitary wave.
    const double hs = h0 + a * s;
    const double hx = -2.0 * kap * a * s * t;
    const double hxx =
        4.0 * kap * kap * a * s * t * t - 2.0 * kap * kap * a * s * s;
    h[i] = hh;
    u[i] = cs * (1.0 - h0 / hs);
    w[i] = -cs * h0 * hx / (2.0 * hs);
    p[i] = -(cs * h0 / gamma) *
           (-0.5 * cs * h0 * (hxx * hs - hx * hx) / (hs * hs));
  }
  auto U = host.makeState(h, u, v, w, p);

  // single-GPU solver built from the host setup
  sgn::SGN2DCuda dev(host);
  dev.upload(U);

  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-8;
  opt.rtol = 1e-6;
  double dt0 = host.cflTimeStep(U, 0.5);   // initial step; then carried between frames

  // --- frame loop: integrate to the frame time, download, write VTK -----------

  std::printf("# shoal showcase: %dx%d P=%d  DOF=%.1fM  tEnd=%.1f  %d frames\n",
              Nx, Ny, P, 5.0 * Nd / 1e6, tEnd, nFrames);
  char name[128];
  sgn::State Uf;
  for (int f = 0; f <= nFrames; ++f) {
    if (f > 0) {
      const double t0 = (f - 1) * tEnd / nFrames;
      const double t1 = f * tEnd / nFrames;
      const auto st = dev.integrateAdaptive(t0, t1, dt0, opt);
      if (st.aborted) { std::fprintf(stderr, "ABORT t=%.3f\n", st.tReached); return 1; }
      dt0 = st.dtLast;
    }
    dev.download(Uf);
    std::snprintf(name, sizeof name, "results/shoal_%04d.vtk", f);
    sgn::writeVtk2D(host, Uf, name);
    std::printf("  frame %3d / %d  (t = %.2f)  mass drift %.2e\n", f, nFrames,
                f * tEnd / nFrames,
                std::fabs(host.mass(Uf) - host.mass(U)) / host.mass(U));
  }
  std::printf("# done -> results/shoal_*.vtk  (ParaView: open series, Warp By"
              " Scalar 'eta', scale ~5)\n");
  return 0;
}
