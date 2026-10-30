// =============================================================================
// beji_battjes.cpp -- periodic waves over a submerged bar vs laboratory (1D CPU)
// =============================================================================
//
// Validation ladder TEST 4 (paper Section 5.8): periodic waves over a submerged trapezoidal bar
// (Beji & Battjes 1993 flume; the classic dispersive harmonic-generation
// benchmark). Non-breaking long-wave case: T = 2.02 s, incident amplitude
// a = 1.0 cm over 0.4 m still depth.
//
// Geometry (still depth H(x), wavemaker at x=0):
//   H = 0.40                      x < 6
//       0.40 - (x-6)/20           6  <= x < 12   (1:20 up-slope, to 0.10)
//       0.10                      12 <= x < 14   (crest)
//       0.10 + (x-14)/10          14 <= x < 17   (1:10 down-slope)
//       0.40                      x >= 17
// The mesh (hx = 0.1) aligns every kink with an element boundary, so H is
// polynomial inside each element and its FR gradient is exact.
//
// Open flume emulated on the periodic mesh: generation zone [0, 4] forced
// with the slow-branch eigenmode (k solved from OUR dispersion relation at
// omega = 2 pi / T), absorption zone [26, 35] relaxing to rest.
//
// Output:
//   results/beji_gauges.csv   t, eta at the 8 gauge stations
//   results/beji_profile.csv  t, x, eta snapshots (one per period + 4 in the last)
//   (non-default parameters: results/beji_g<gamma>_a<alpha>_n<nManning>_A<amp>_
//    {gauges,profile}.csv, so sweeps never overwrite the production run)
//   stdout: first four harmonic amplitudes at each gauge (DFT over the last
//           10 periods) -> compare against the published lab data.
//
// CALL CHAIN
//   main -> wavenumber k from sgn::slowCp2 (Newton; skipped for alpha = 0)
//        -> SGN1D with the bar bathymetry -> addRelaxationZone x2
//        -> integrateAdaptive to each T/64 sample (20 warm-up + 10 measured
//           periods) -> samplePrimitive at the gauges -> DFT harmonics
// =============================================================================
//
//   usage: ./build/beji_battjes [gamma=2.0] [alpha=3.0] [P=4] [ne] [nManning] [A=0.010]
//          alpha=0 -> non-dispersive shallow-water baseline (same mesh/forcing)

#include "../src/physics/sgn_dispersion.h"
#include "../src/solver/sgn1d.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

static double bejiDepth(double x) {
  if (x < 6.0) return 0.40;
  if (x < 12.0) return 0.40 - (x - 6.0) / 20.0;
  if (x < 14.0) return 0.10;
  if (x < 17.0) return 0.10 + (x - 14.0) / 10.0;
  return 0.40;
}

int main(int argc, char** argv) {
  const double gamma = (argc > 1) ? std::atof(argv[1]) : 2.0;
  const double alpha = (argc > 2) ? std::atof(argv[2]) : 3.0;
  const int P = (argc > 3) ? std::atoi(argv[3]) : 4;
  const int neArg = (argc > 4) ? std::atoi(argv[4]) : 350; // resolution check
  const double nMan = (argc > 5) ? std::atof(argv[5]) : 0.0; // Manning friction

  const double g = 9.81;
  const double h0 = 0.40;          // offshore still depth
  const double Twave = 2.02;       // incident period [s]
  // Incident amplitude [m]; 1.0 cm is the laboratory value. It is exposed
  // because the NON-DISPERSIVE run (alpha = 0) steepens without bound over the
  // crest: the steepening length is ~ lambda h / (3 a), which at the laboratory
  // amplitude is ~6 m against the ~5 m of crest plus lee slope, so the wave
  // breaks inside the bar and the unlimited operator cannot continue. Reducing
  // a pushes the steepening beyond the flume and lets both models complete the
  // measurement window, at the cost of a weaker (but still resolved) harmonic
  // signal -- the ratio A2/A1 scales like a/h, not like a.
  const double A = (argc > 6) ? std::atof(argv[6]) : 0.010;
  const double omega = 2.0 * kPi / Twave;

  const double L = 35.0;
  const int ne = neArg;            // default hx = 0.1; kinks stay on element
                                   // boundaries for ne multiple of 350/35=10/m

  sgn::Params par;
  par.g = g;
  par.gamma = gamma;
  par.c = alpha * std::sqrt(g * h0);
  par.nManning = nMan;

  // Solve k from OUR slow branch:  k * Cp_slow(k h0) = omega  (Newton).
  //
  // alpha = 0 is the NON-DISPERSIVE limit: the relaxation is switched off, the
  // dispersion relation collapses to the long-wave one, and k = omega/sqrt(g h0)
  // exactly -- which is the initial guess, so the Newton iteration is skipped.
  // (It would divide by alpha and produce NaN.) Everything downstream then
  // degenerates on its own: Ap = (Cp^2/h0 - g) A = 0 and Aw = -gamma Ap/(omega
  // h0) = 0, so the generation zone injects a pure shallow-water wave, and with
  // c = 0 the p and w fields stay identically zero. The run is then plain 2D
  // shallow water on the same mesh, bathymetry and forcing -- the baseline that
  // isolates what dispersion contributes to harmonic generation over the bar.
  double k = omega / std::sqrt(g * h0); // long-wave guess (exact if alpha = 0)
  for (int it = 0; alpha > 0.0 && it < 60; ++it) {
    const double cp = std::sqrt(sgn::slowCp2(k * h0, gamma, alpha) * g * h0);
    const double f = k * cp - omega;
    const double dk = 1e-7 * k;
    const double cp2 = std::sqrt(sgn::slowCp2((k + dk) * h0, gamma, alpha) * g * h0);
    const double fp = ((k + dk) * cp2 - k * cp) / dk;
    const double step = f / fp;
    k -= step;
    if (std::fabs(step) < 1e-14 * k) break;
  }
  const double Cp = omega / k;
  std::printf("# Beji-Battjes: T=%.3f s  A=%.3f m  gamma=%.2f alpha=%.1f P=%d\n",
              Twave, A, gamma, alpha, P);
  std::printf("# dispersion: k=%.6f 1/m  kh0=%.4f  Cp=%.4f m/s  (Airy Cp=%.4f)\n",
              k, k * h0, Cp, std::sqrt(sgn::airyCp2(k * h0) * g * h0));

  // Slow-branch eigenmode amplitudes at depth h0 (same as test_dispersion).
  const double Au = Cp / h0 * A;
  const double Ap = (Cp * Cp / h0 - g) * A;
  const double Aw = -gamma * Ap / (omega * h0);
  auto waveTarget = [=](double x, double t, double prim[4]) {
    const double ph = k * x - omega * t;
    // Smooth start-up ramp over 3 periods avoids an impulsive transient.
    const double r = std::fmin(1.0, t / (3.0 * Twave));
    const double ramp = r * r * (3.0 - 2.0 * r);
    prim[0] = h0 + ramp * A * std::cos(ph);
    prim[1] = ramp * Au * std::cos(ph);
    prim[2] = ramp * Aw * std::sin(ph);
    prim[3] = ramp * Ap * std::cos(ph);
  };
  auto restTarget = [=](double, double, double prim[4]) {
    prim[0] = 0.40; prim[1] = prim[2] = prim[3] = 0.0;
  };

  sgn::SGN1D sys(P, ne, L, par, bejiDepth);
  sys.addRelaxationZone(0.0, 4.0, 4.0 * omega, waveTarget);
  sys.addRelaxationZone(26.0, 35.0, 4.0 * omega, restTarget);

  // Quiescent start.
  const int N = sys.nDof();
  const auto xn = sys.xCoords();
  std::vector<double> h(N), z(N, 0.0);
  for (int i = 0; i < N; ++i) h[i] = bejiDepth(xn[i]);
  auto U = sys.makeState(h, z, z, z);

  // Gauge stations (standard Beji-Battjes positions, wavemaker at x=0).
  const std::vector<double> gaugeX = {5.7, 10.5, 12.5, 13.5,
                                      14.5, 15.7, 17.3, 19.0};

  // Time marching: 20 warm-up periods + 10 measured periods, sampled at T/64.
  const int nWarm = 20, nMeas = 10, sPerT = 64;
  const double dtS = Twave / sPerT;
  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-11;
  opt.rtol = 1e-8;

  // Canonical name for the default run (used by make_figures.py); a per-run
  // name for parameter-sweep runs so they do not overwrite each other.
  char gname[160], pname[160];
  const bool isDefault =
      (gamma == 2.0 && alpha == 3.0 && nMan == 0.0 && A == 0.010);
  if (isDefault) {
    std::snprintf(gname, sizeof gname, "results/beji_gauges.csv");
    std::snprintf(pname, sizeof pname, "results/beji_profile.csv");
  } else {
    // The profile name has to vary too: it used to be fixed, so a sweep run
    // silently overwrote the profiles of the default run.
    std::snprintf(gname, sizeof gname,
                  "results/beji_g%.1f_a%.1f_n%.3f_A%.4f_gauges.csv",
                  gamma, alpha, nMan, A);
    std::snprintf(pname, sizeof pname,
                  "results/beji_g%.1f_a%.1f_n%.3f_A%.4f_profile.csv",
                  gamma, alpha, nMan, A);
  }
  std::FILE* fg = std::fopen(gname, "w");
  std::FILE* fp = std::fopen(pname, "w");
  if (!fg || !fp) { std::fprintf(stderr, "cannot open results/ files\n"); return 1; }
  std::fprintf(fg, "t");
  for (size_t gi = 0; gi < gaugeX.size(); ++gi)
    std::fprintf(fg, ",eta%zu", gi + 1);
  std::fprintf(fg, "\n");

  const int nMeasSamples = nMeas * sPerT;
  std::vector<std::vector<double>> series(gaugeX.size());

  double t = 0.0;
  const int nTotal = (nWarm + nMeas) * sPerT;
  for (int s = 0; s < nTotal; ++s) {
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, t, t + dtS,
                                           sys.cflTimeStep(U, 0.5), opt);
    if (st.aborted) { std::fprintf(stderr, "ABORT at t=%.3f\n", st.tReached); return 1; }
    t += dtS;

    std::fprintf(fg, "%.6f", t);
    for (size_t gi = 0; gi < gaugeX.size(); ++gi) {
      double prim[4];
      sys.samplePrimitive(U, gaugeX[gi], prim);
      const double eta = prim[0] - bejiDepth(gaugeX[gi]);
      std::fprintf(fg, ",%.8e", eta);
      if (s >= nWarm * sPerT) series[gi].push_back(eta);
    }
    std::fprintf(fg, "\n");

    // Profile snapshots: one per period throughout, plus 4 phases of the final
    // period. The once-per-period trace is what makes a run that never reaches
    // the measurement window still comparable against one that does -- the
    // non-dispersive case breaks over the bar after about five periods, so the
    // final-period snapshots alone would leave nothing to plot against.
    if ((s % sPerT) == 0 || (s >= nTotal - sPerT && (s % (sPerT / 4)) == 0)) {
      for (int i = 0; i < N; ++i)
        std::fprintf(fp, "%.6f,%.6f,%.8e\n", t, xn[i], U[i] - bejiDepth(xn[i]));
      // Flush at every snapshot: a run that steepens into a bore does not fail,
      // it grinds to dtMin and has to be interrupted, and buffered output would
      // be lost exactly for the run whose partial history we want to keep.
      std::fflush(fp);
      std::fflush(fg);
    }
  }
  std::fclose(fg);
  std::fclose(fp);

  // Harmonic amplitudes A_n = 2|c_n| over the measured window.
  std::printf("#\n# harmonic amplitudes [m] (DFT over last %d periods)\n", nMeas);
  std::printf("# gauge   x[m]     A1          A2          A3          A4\n");
  for (size_t gi = 0; gi < gaugeX.size(); ++gi) {
    std::printf("#  %zu     %5.1f ", gi + 1, gaugeX[gi]);
    for (int n = 1; n <= 4; ++n) {
      std::complex<double> c(0.0, 0.0);
      for (int m = 0; m < nMeasSamples; ++m) {
        const double tm = m * dtS; // phase origin irrelevant for |c|
        c += series[gi][m] *
             std::exp(std::complex<double>(0.0, -n * omega * tm));
      }
      c /= (double)nMeasSamples;
      std::printf(" %.4e", 2.0 * std::abs(c));
    }
    std::printf("\n");
  }
  std::printf("# done. gauges -> %s\n", gname);
  return 0;
}
