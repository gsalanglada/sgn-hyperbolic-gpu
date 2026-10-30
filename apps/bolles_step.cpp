// =============================================================================
// bolles_step.cpp -- random-sea statistics over an abrupt depth change (1D CPU)
// =============================================================================
//
// Random-sea statistics over an abrupt depth change (paper Section 8), after the
// experiment of Bolles, Speer & Moore, PRF 4:011801 (2019)
// (arXiv:1808.07958; PDF in references/).
//
// Experiment: 6 m tank, upstream depth h1 = 12.5 cm, plexiglass step to
// h2 = 3 cm (depth ratio 0.24). Random paddle forcing with GAUSSIAN spectral
// weights centred at omega0 = 12.5 rad/s (2 Hz) with sigma_w = 12.5 rad/s,
// uniform random phases. Findings: eta Gaussian upstream; skewness/kurtosis
// maximal at x = +15 cm downstream of the step (independent of amplitude);
// eta follows a gamma distribution there; rogue-wave probability x47-65.
//
// This deck reproduces the setup with the hyperbolic-SGN solver:
//   - periodic domain L = 8 m: generation zone [0,1.2], step at x_s = 3.5 m
//     (tanh profile of characteristic width ws = 2 cm, about one element),
//     absorption zone [6.5,8] (the dampener).
//   - incident field: M random-phase slow-branch eigenmodes at depth h1 with
//     the experiment's Gaussian weights, truncated at kh1 <= 3.2 (the deeper-
//     water tail is outside Boussinesq validity; standard limitation). The
//     amplitude scale is set by the target upstream std sigma_eta.
//   - gauges every 5 cm from x_s-0.5 m to x_s+1.0 m; skewness/excess kurtosis
//     of eta after a spin-up.
// Expected signature: skew ~ 0, kurt ~ 0 upstream; a localized skewness/
// kurtosis maximum ~15 cm downstream of the step.
//
//   usage: ./build/bolles_step [seed=1] [sigma_eta=0.0025 m] [Trec=300 s]
//                              [nManning=0]
//     seed       random-phase realization (the paper's ensemble uses 6 seeds)
//     sigma_eta  target upstream standard deviation of eta [m]
//     Trec       recorded time after a 60 s spin-up [s]
//     nManning   Manning friction coefficient
//   output (per run, prefix results/bolles_s<seed>_a<sigma>_n<nManning>):
//           <prefix>_stats.csv   x_rel_cm, sigma, skewness, excess kurtosis
//           <prefix>_series.csv  t, eta at x_s-0.10, +0.05, +0.15, +0.30
//           <prefix>_hist.csv    histogram at the hot spot + gamma and Gaussian pdfs
//   and one appended row per run in results/bolles_sweep.csv (whole ensemble).
//
// CALL CHAIN
//   main -> random-phase modes (k from sgn::slowCp2, src/physics/sgn_dispersion.h)
//        -> SGN1D with the step bathymetry -> addRelaxationZone x2
//           (generation target = mode superposition, absorption = rest)
//        -> integrateAdaptive to each 1/20 s sample (spin-up, then record)
//        -> samplePrimitive at the gauges -> moments per gauge
//        -> gamma fit + rogue-wave probability at the skewness maximum
// =============================================================================

#include "../src/physics/sgn_dispersion.h"
#include "../src/solver/sgn1d.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace { constexpr double kPi = 3.14159265358979323846; }

int main(int argc, char** argv) {
  // --- command line ---------------------------------------------------------
  const unsigned seed = (argc > 1) ? (unsigned)std::atoi(argv[1]) : 1u;
  const double sigEta = (argc > 2) ? std::atof(argv[2]) : 0.0025;
  const double Trec = (argc > 3) ? std::atof(argv[3]) : 300.0;
  const double nMan = (argc > 4) ? std::atof(argv[4]) : 0.0;

  const double g = 9.81;
  const double h1 = 0.125, h2 = 0.03; // experiment depths
  const double gamma = 2.0;           // phase error +0.4% at kh~2 (our table)
  const double alpha = 3.0;
  const double L = 8.0, xs = 3.5, ws = 0.02; // near-abrupt step (plexiglass edge)
  const int P = 4, ne = 320;          // he = 2.5 cm

  sgn::Params par;
  par.g = g;
  par.gamma = gamma;
  par.c = alpha * std::sqrt(g * h1);
  par.nManning = nMan;

  // still depth: h1 upstream, h2 downstream, smooth tanh step at xs
  auto depth = [=](double x) {
    return h1 - (h1 - h2) * 0.5 * (1.0 + std::tanh((x - xs) / ws));
  };

  // --- Incident random sea: Gaussian-weight modes at depth h1 --------------
  // Experiment: equispaced omega_n with Gaussian weights (w0 = sw = 12.5
  // rad/s), N=3000 -> 300 s fundamental period. With few equispaced modes the
  // signal would REPEAT every 2 pi/dw and the statistics would sample a loop,
  // not a random sea -> we JITTER each mode's frequency uniformly inside its
  // bin, which destroys the periodicity at moderate M. Band truncated where
  // k h1 > khMax (deeper-water tail is outside Boussinesq validity).
  const double w0 = 12.5, sw = 12.5, khMax = 3.2;  // spectral centre, width, kh cut-off
  const double wLo = 1.0, wHi = 17.0; // ~ kh1 = khMax at the top
  const int M = 160;                  // frequency bins before truncation
  std::vector<double> om, kk, aa, ph; // kept modes: omega, k, amplitude, phase
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 2.0 * kPi);
  std::uniform_real_distribution<double> jit(-0.45, 0.45);
  const double dw = (wHi - wLo) / M;
  for (int n = 0; n < M; ++n) {
    const double w = wLo + (n + 0.5 + jit(rng)) * dw;
    // k from OUR slow branch at h1 (Newton).
    double k = w / std::sqrt(g * h1);
    for (int it = 0; it < 60; ++it) {
      const double cp = std::sqrt(sgn::slowCp2(k * h1, gamma, alpha) * g * h1);
      const double f = k * cp - w;
      const double dk = 1e-7 * k;
      const double cp2 = std::sqrt(sgn::slowCp2((k + dk) * h1, gamma, alpha) * g * h1);
      const double fp = ((k + dk) * cp2 - k * cp) / dk;
      k -= f / fp;
      if (!(k > 0.0)) { k = w / std::sqrt(g * h1); break; }
    }
    if (k * h1 > khMax) continue;
    om.push_back(w);
    kk.push_back(k);
    aa.push_back(std::exp(-(w - w0) * (w - w0) / (2.0 * sw * sw)));
    ph.push_back(uni(rng));
  }
  // Scale amplitudes so the upstream std of eta is sigEta: var = sum a^2/2.
  {
    double v = 0.0;
    for (double a : aa) v += 0.5 * a * a;
    const double s = sigEta / std::sqrt(v);
    for (double& a : aa) a *= s;
  }
  const int nM = (int)om.size();
  std::printf("# Bolles-step pilot: seed=%u sigma_eta=%.4f m  modes=%d "
              "(kh1 in [%.2f, %.2f])  gamma=%.1f alpha=%.1f\n",
              seed, sigEta, nM, kk.front() * h1, kk.back() * h1, gamma, alpha);

  // Per-run output prefix so sweeps/ensembles never overwrite each other.
  char pref[128];
  std::snprintf(pref, sizeof pref, "results/bolles_s%u_a%.4f_n%.3f", seed,
                sigEta, nMan);

  sgn::SGN1D sys(P, ne, L, par, depth);

  // Generation target: superposition of slow-branch eigenmodes at depth h1.
  // FAST evaluation: precompute a_m*cos(k_m x + d_m), a_m*sin(...) for every
  // zone node (keyed by the bit pattern of x), and cache cos/sin(w_m t) per
  // time level (the zone loop calls all nodes at the same t). Per node the
  // cost is then O(M) fma instead of O(M) sincos.
  struct TCache {
    double t = -1e300;
    std::vector<double> cw, sw;
  };
  auto cache = std::make_shared<TCache>();
  cache->cw.resize(nM);
  cache->sw.resize(nM);

  // Slow-branch eigenmode ratios per mode (same formulas as beji_battjes.cpp):
  //   u = fu * eta,  p = fpp * eta,  w = fw * (quadrature-phase eta)
  std::vector<double> fu(nM), fpp(nM), fw(nM);
  for (int m = 0; m < nM; ++m) {
    const double Cp = om[m] / kk[m];
    fu[m] = Cp / h1;
    fpp[m] = Cp * Cp / h1 - g;
    fw[m] = -gamma * fpp[m] / (om[m] * h1);
  }
  // Zone-node tables (same selection criterion as addRelaxationZone).
  auto keyOf = [](double x) {
    unsigned long long k;
    static_assert(sizeof(k) == sizeof(double), "");
    std::memcpy(&k, &x, sizeof(k));
    return k;
  };
  std::unordered_map<unsigned long long, int> slot;
  std::vector<double> Ce, Se;
  {
    const auto xall = sys.xCoords();
    for (double x : xall)
      if (x > 0.0 && x < 1.2) {
        const int j = (int)slot.size();
        slot[keyOf(x)] = j;
        for (int m = 0; m < nM; ++m) {
          const double a = kk[m] * x + ph[m];
          Ce.push_back(aa[m] * std::cos(a));
          Se.push_back(aa[m] * std::sin(a));
        }
      }
  }

  // Generation-zone target at (x, t): ramped superposition of all modes.
  auto waveTarget = [=](double x, double t, double prim[4]) {
    const double r = std::fmin(1.0, t / 20.0); // 20 s smooth start-up
    const double ramp = r * r * (3.0 - 2.0 * r);
    if (t != cache->t) {
      for (int m = 0; m < nM; ++m) {
        cache->cw[m] = std::cos(om[m] * t);
        cache->sw[m] = std::sin(om[m] * t);
      }
      cache->t = t;
    }
    double eta = 0.0, u = 0.0, w = 0.0, p = 0.0;
    const auto it = slot.find(keyOf(x));
    if (it != slot.end()) {
      const double* ce = &Ce[(size_t)it->second * nM];
      const double* se = &Se[(size_t)it->second * nM];
      for (int m = 0; m < nM; ++m) {
        const double term = ce[m] * cache->cw[m] + se[m] * cache->sw[m];
        const double sterm = se[m] * cache->cw[m] - ce[m] * cache->sw[m];
        eta += term;
        u += fu[m] * term;
        p += fpp[m] * term;
        w += fw[m] * sterm;
      }
    } else { // fallback (should not happen): direct evaluation
      for (int m = 0; m < nM; ++m) {
        const double phm = kk[m] * x - om[m] * t + ph[m];
        const double c = std::cos(phm), s = std::sin(phm);
        eta += aa[m] * c;
        u += fu[m] * aa[m] * c;
        p += fpp[m] * aa[m] * c;
        w += fw[m] * aa[m] * s;
      }
    }
    prim[0] = h1 + ramp * eta;
    prim[1] = ramp * u;
    prim[2] = ramp * w;
    prim[3] = ramp * p;
  };
  auto restTarget = [=](double x, double, double prim[4]) {
    prim[0] = depth(x);
    prim[1] = prim[2] = prim[3] = 0.0;
  };

  sys.addRelaxationZone(0.0, 1.2, 3.0 * w0, waveTarget);
  sys.addRelaxationZone(6.5, 8.0, 3.0 * w0, restTarget);

  const int N = sys.nDof();
  const auto xn = sys.xCoords();
  std::vector<double> h(N), z(N, 0.0);
  for (int i = 0; i < N; ++i) h[i] = depth(xn[i]);
  auto U = sys.makeState(h, z, z, z);

  // Gauges: xs-0.50 .. xs+1.00 every 5 cm.
  std::vector<double> gx;
  for (double d = -0.50; d <= 1.001; d += 0.05) gx.push_back(xs + d);
  const int nG = (int)gx.size();
  std::vector<std::vector<double>> rec(nG);

  // --- time marching: spin-up (discarded), then record at 20 Hz ---------------
  const double tSpin = 60.0;
  const double fS = 20.0; // sampling rate [Hz]
  const double dtS = 1.0 / fS;
  const long nSpin = (long)(tSpin * fS);
  const long nRec = (long)(Trec * fS);

  char fname[160];
  std::snprintf(fname, sizeof fname, "%s_series.csv", pref);
  std::FILE* fser = std::fopen(fname, "w");
  if (!fser) { std::fprintf(stderr, "cannot open results/\n"); return 1; }
  std::fprintf(fser, "t,eta_m10,eta_p05,eta_p15,eta_p30\n");
  const double xsel[4] = {xs - 0.10, xs + 0.05, xs + 0.15, xs + 0.30};

  sgn::AdaptiveRKOptions opt;
  opt.atol = 1e-10;
  opt.rtol = 1e-7;

  double t = 0.0;
  for (long s = 0; s < nSpin + nRec; ++s) {
    const auto st = sgn::integrateAdaptive(sys.rhsFn(), U, t, t + dtS,
                                           sys.cflTimeStep(U, 0.5), opt);
    if (st.aborted) { std::fprintf(stderr, "ABORT t=%.2f\n", st.tReached); return 1; }
    t += dtS;
    if (s < nSpin) continue;

    for (int q = 0; q < nG; ++q) {
      double prim[4];
      sys.samplePrimitive(U, gx[q], prim);
      rec[q].push_back(prim[0] - depth(gx[q]));
    }
    std::fprintf(fser, "%.3f", t);
    for (double xq : xsel) {
      double prim[4];
      sys.samplePrimitive(U, xq, prim);
      std::fprintf(fser, ",%.6e", prim[0] - depth(xq));
    }
    std::fprintf(fser, "\n");
  }
  std::fclose(fser);

  // Statistics per gauge: central moments m2, m3, m4 of the recorded eta,
  // skewness m3/m2^1.5 and excess kurtosis m4/m2^2 - 3 (both 0 for a Gaussian).
  std::snprintf(fname, sizeof fname, "%s_stats.csv", pref);
  std::FILE* fst = std::fopen(fname, "w");
  std::fprintf(fst, "x_rel_cm,sigma_m,skewness,excess_kurtosis\n");
  std::printf("#\n#  x-xs[cm]   sigma[mm]   skew    exkurt\n");
  double skewMax = -1e30, xAtMax = 0.0;
  int qMax = 0;
  for (int q = 0; q < nG; ++q) {
    const auto& r = rec[q];
    const double n = (double)r.size();
    double m1 = 0.0;
    for (double v : r) m1 += v;
    m1 /= n;
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (double v : r) {
      const double d = v - m1;
      m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
    }
    m2 /= n; m3 /= n; m4 /= n;
    const double sk = m3 / std::pow(m2, 1.5);
    const double ku = m4 / (m2 * m2) - 3.0;
    const double xr = (gx[q] - xs) * 100.0;
    std::fprintf(fst, "%.1f,%.6e,%.4f,%.4f\n", xr, std::sqrt(m2), sk, ku);
    std::printf("#   %6.1f     %6.3f    %+.3f   %+.3f\n", xr,
                1e3 * std::sqrt(m2), sk, ku);
    if (sk > skewMax) { skewMax = sk; xAtMax = xr; qMax = q; }
  }
  std::fclose(fst);
  std::printf("#\n# max skewness %.3f at x-xs = %.1f cm "
              "(experiment: hot spot at +15 cm)\n", skewMax, xAtMax);

  // --- Gamma fit + rogue-wave probability at the hot spot -------------------
  // Paper eq. (3): zero-mean gamma with scale eta0, shape alpha. Moment fit:
  // skew = 2/sqrt(alpha), var = alpha eta0^2. Rogue defined (paper) as
  // P(eta > 4 sigma | eta > 0); Gaussian reference 6.3e-5; the lab measured
  // enhancement factors of x47 (alpha=6.5) to x65 (alpha=4.5).
  {
    const auto& r = rec[qMax];
    const double n = (double)r.size();
    double m1 = 0.0;
    for (double v : r) m1 += v;
    m1 /= n;
    double m2 = 0.0, m3 = 0.0;
    for (double v : r) {
      const double d = v - m1;
      m2 += d * d; m3 += d * d * d;
    }
    m2 /= n; m3 /= n;
    const double sig = std::sqrt(m2);
    const double sk = m3 / std::pow(m2, 1.5);
    const double alphaG = 4.0 / (sk * sk);
    const double eta0 = std::sqrt(m2 / alphaG);

    auto lpdf = [&](double e) { // log of paper eq. (3)
      const double z = alphaG + e / eta0;
      if (z <= 0.0) return -1e300;
      return -alphaG + (alphaG - 1.0) * std::log(z) - e / eta0 -
             std::log(eta0) - std::lgamma(alphaG);
    };
    auto integ = [&](double lo) { // trapezoid over the exponential tail
      const double de = eta0 / 400.0;
      double s = 0.0;
      for (double e = lo; e < lo + 50.0 * eta0; e += de)
        s += std::exp(lpdf(e + 0.5 * de)) * de;
      return s;
    };
    const double pPos = integ(0.0);
    const double pRogueG = integ(4.0 * sig) / pPos;
    long nPos = 0, nRog = 0;
    for (double v : r) {
      const double d = v - m1;
      if (d > 0.0) ++nPos;
      if (d > 4.0 * sig) ++nRog;
    }
    const double pEmp = (nPos > 0) ? (double)nRog / (double)nPos : 0.0;
    const double pGauss = 6.3e-5;

    std::printf("#\n# hot-spot gamma fit (x-xs=%.1f cm): alpha=%.2f eta0=%.4f mm"
                "  [lab: alpha ~ 4.5-6.5]\n", xAtMax, alphaG, 1e3 * eta0);
    std::printf("# rogue P(eta>4sig|eta>0): gamma-fit %.3e (x%.0f vs Gaussian)"
                " | empirical %.3e (%ld/%ld events)\n",
                pRogueG, pRogueG / pGauss, pEmp, nRog, nPos);

    // One aggregate row per run (append) -> the whole sweep in one table.
    std::FILE* fsw = std::fopen("results/bolles_sweep.csv", "a");
    if (fsw) {
      std::fseek(fsw, 0, SEEK_END);
      if (std::ftell(fsw) == 0)
        std::fprintf(fsw, "seed,sigma_eta,nManning,x_hotspot_cm,skew_max,"
                          "alpha,eta0_mm,rogue_gamma,rogue_factor,"
                          "rogue_emp,n_events\n");
      std::fprintf(fsw, "%u,%.4f,%.3f,%.1f,%.4f,%.3f,%.4f,%.4e,%.1f,%.4e,%ld\n",
                   seed, sigEta, nMan, xAtMax, skewMax, alphaG, 1e3 * eta0,
                   pRogueG, pRogueG / pGauss, pEmp, nRog);
      std::fclose(fsw);
    }

    // Histogram + fits for the money figure.
    std::snprintf(fname, sizeof fname, "%s_hist.csv", pref);
    std::FILE* fh = std::fopen(fname, "w");
    if (fh) {
      double lo = 1e300, hi = -1e300;
      for (double v : r) { lo = std::fmin(lo, v - m1); hi = std::fmax(hi, v - m1); }
      const int nb = 44;
      const double bw = (hi - lo) / nb;
      std::vector<long> cnt(nb, 0);
      for (double v : r) {
        int b = (int)((v - m1 - lo) / bw);
        if (b >= nb) b = nb - 1;
        if (b < 0) b = 0;
        ++cnt[b];
      }
      std::fprintf(fh, "eta_m,density,gamma_pdf,gauss_pdf\n");
      for (int b = 0; b < nb; ++b) {
        const double ec = lo + (b + 0.5) * bw;
        const double dens = cnt[b] / (n * bw);
        const double pg = std::exp(lpdf(ec));
        const double pn = std::exp(-0.5 * ec * ec / m2) /
                          std::sqrt(2.0 * kPi * m2);
        std::fprintf(fh, "%.6e,%.6e,%.6e,%.6e\n", ec, dens, pg, pn);
      }
      std::fclose(fh);
    }
  }

  std::printf("# outputs -> %s_{stats,series,hist}.csv\n", pref);
  return 0;
}
