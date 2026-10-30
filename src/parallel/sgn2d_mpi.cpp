// =============================================================================
// sgn2d_mpi.cpp -- implementation of the distributed 2D FR solver (MPI, CPU)
// =============================================================================
//
// rhs() follows the serial SGN2D::rhs with one extra phase:
//   1. face states of every owned element
//   2. HALO EXCHANGE: pack the leftmost / rightmost owned column's x-face states
//      into contiguous line buffers and swap them with the ring neighbours
//   3. volume work per owned element; an x-neighbour outside the slab is read
//      from the ghost line instead of from the local arrays
// =============================================================================

#include "sgn2d_mpi.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sgn {

using sgn2d::NVAR;
using sgn2d::H;

// -----------------------------------------------------------------------------
// Constructor: partition, then owned bathymetry (nodal, gradients, four faces).
// -----------------------------------------------------------------------------
SGN2DMpi::SGN2DMpi(int P, int NxGlobal, int Ny, double Lx, double Ly,
                   const sgn2d::Params& par, MPI_Comm comm,
                   std::function<double(double, double)> stillDepth,
                   PointSet pointSet)
    : P_(P), np_(P + 1), npp_((P + 1) * (P + 1)), NxG_(NxGlobal), Ny_(Ny),
      Lx_(Lx), Ly_(Ly), hx_(Lx / NxGlobal), hy_(Ly / Ny),
      Jx_(0.5 * Lx / NxGlobal), Jy_(0.5 * Ly / Ny), par_(par), comm_(comm),
      op_(P, pointSet) {
  // Step 1: rank, size and the column slab of this rank
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (NxGlobal < size)
    throw std::invalid_argument("SGN2DMpi: need NxGlobal >= nranks");
  part_ = makeSlabPartition1D(NxGlobal, rank, size);

  // Step 2: bathymetry arrays of the owned elements (zero = flat)
  const int Nl = nLocalDof();
  const int neL = nOwnedElem();
  Hnod_.assign(Nl, 0.0);
  Hx_.assign(Nl, 0.0);
  Hy_.assign(Nl, 0.0);
  HfaceXL_.assign((size_t)neL * np_, 0.0);
  HfaceXR_.assign((size_t)neL * np_, 0.0);
  HfaceYB_.assign((size_t)neL * np_, 0.0);
  HfaceYT_.assign((size_t)neL * np_, 0.0);
  if (!stillDepth) return;
  if (pointSet != PointSet::GaussLobatto)
    throw std::invalid_argument("SGN2DMpi: bathymetry requires GLL points");

  // Step 3: nodal H at the owned nodes
  std::vector<double> x, y;
  nodeCoordsLocal(x, y);
  for (int i = 0; i < Nl; ++i) Hnod_[i] = stillDepth(x[i], y[i]);

  // Step 4: nodal gradients, tensor FR differentiation along x- and y-lines
  const auto& D = op_.D();
  for (int e = 0; e < neL; ++e) {
    const double* He = &Hnod_[(size_t)e * npp_];
    for (int b = 0; b < np_; ++b)
      for (int a = 0; a < np_; ++a) {
        double dHx = 0.0, dHy = 0.0;
        for (int k = 0; k < np_; ++k) {
          dHx += D[(size_t)a * np_ + k] * He[k + np_ * b];   // along x
          dHy += D[(size_t)b * np_ + k] * He[a + np_ * k];   // along y
        }
        Hx_[(size_t)e * npp_ + a + np_ * b] = dHx / Jx_;
        Hy_[(size_t)e * npp_ + a + np_ * b] = dHy / Jy_;
      }
  }
  // Step 5: still depth on all four faces of every owned element
  const auto& xi = op_.nodes();
  for (int e = 0; e < neL; ++e) {
    const int lx = e % part_.w, ley = e / part_.w;   // local column, row
    const int gex = part_.a + lx;                    // global column
    // Periodic wrap: face coordinates are evaluated exactly as the serial
    // driver does (the rightmost/topmost interface uses the WRAPPED coordinate
    // x=0 / y=0, not Lx / Ly) so MPI == serial stays bitwise for ANY H.
    const int gexR = (gex + 1) % NxG_;
    const int leyT = (ley + 1) % Ny_;
    for (int t = 0; t < np_; ++t) {
      const double yb = ley * hy_ + 0.5 * hy_ * (xi[t] + 1.0);   // y of transverse node t
      const double xa = gex * hx_ + 0.5 * hx_ * (xi[t] + 1.0);   // x of transverse node t
      HfaceXL_[(size_t)e * np_ + t] = stillDepth(gex * hx_, yb);   // left
      HfaceXR_[(size_t)e * np_ + t] = stillDepth(gexR * hx_, yb);  // right
      HfaceYB_[(size_t)e * np_ + t] = stillDepth(xa, ley * hy_);   // bottom
      HfaceYT_[(size_t)e * np_ + t] = stillDepth(xa, leyT * hy_);  // top
    }
  }
}

// -----------------------------------------------------------------------------
// nodeCoordsLocal: global (x, y) of every owned node.
// -----------------------------------------------------------------------------
void SGN2DMpi::nodeCoordsLocal(std::vector<double>& x,
                               std::vector<double>& y) const {
  const int Nl = nLocalDof();
  x.assign(Nl, 0.0);
  y.assign(Nl, 0.0);
  const auto& xi = op_.nodes();
  for (int e = 0; e < nOwnedElem(); ++e) {
    const int lx = e % part_.w, ley = e / part_.w;
    const int gex = part_.a + lx;
    for (int b = 0; b < np_; ++b)
      for (int a = 0; a < np_; ++a) {
        const int g = e * npp_ + a + np_ * b;
        x[g] = gex * hx_ + 0.5 * hx_ * (xi[a] + 1.0);
        y[g] = ley * hy_ + 0.5 * hy_ * (xi[b] + 1.0);
      }
  }
}

// -----------------------------------------------------------------------------
// makeStateLocal: owned primitive fields -> owned var-major state.
// -----------------------------------------------------------------------------
State SGN2DMpi::makeStateLocal(const std::vector<double>& h,
                               const std::vector<double>& u,
                               const std::vector<double>& v,
                               const std::vector<double>& w,
                               const std::vector<double>& p) const {
  const int Nl = nLocalDof();
  State U(nLocalTotal());
  for (int i = 0; i < Nl; ++i) {
    double Un[NVAR];
    sgn2d::fromPrimitive(h[i], u[i], v[i], w[i], p[i], Un);
    for (int vv = 0; vv < NVAR; ++vv) U[(size_t)vv * Nl + i] = Un[vv];
  }
  return U;
}

// -----------------------------------------------------------------------------
// rhs: distributed right-hand side of the owned state.
// -----------------------------------------------------------------------------
void SGN2DMpi::rhs(const State& U, State& dUdt) const {
  const int np = np_;
  const int w = part_.w;         // owned columns
  const int neL = nOwnedElem();  // owned elements
  const int Nl = nLocalDof();    // owned nodes per variable
  if ((int)U.size() != nLocalTotal())
    throw std::invalid_argument("SGN2DMpi::rhs: size mismatch");
  dUdt.assign(nLocalTotal(), 0.0);

  const auto& D = op_.D();
  const auto& lL = op_.lL();
  const auto& lR = op_.lR();
  const auto& gLp = op_.gLprime();
  const auto& gRp = op_.gRprime();

  // flat index of variable var at node (a, b) of owned element e
  auto Uidx = [&](int e, int a, int b, int var) {
    return (size_t)var * Nl + (size_t)e * npp_ + a + np * b;
  };

  // 1) Owned face states (same loops and index layout as the serial driver).
  std::vector<double> xFaceL((size_t)neL * np * NVAR), xFaceR((size_t)neL * np * NVAR);
  std::vector<double> yFaceB((size_t)neL * np * NVAR), yFaceT((size_t)neL * np * NVAR);
  for (int e = 0; e < neL; ++e)
    for (int var = 0; var < NVAR; ++var) {
      for (int b = 0; b < np; ++b) {               // x-faces along x-lines
        double sL = 0.0, sR = 0.0;
        for (int a = 0; a < np; ++a) {
          const double val = U[Uidx(e, a, b, var)];
          sL += lL[a] * val; sR += lR[a] * val;
        }
        xFaceL[((size_t)e * np + b) * NVAR + var] = sL;
        xFaceR[((size_t)e * np + b) * NVAR + var] = sR;
      }
      for (int a = 0; a < np; ++a) {               // y-faces along y-lines
        double sB = 0.0, sT = 0.0;
        for (int b = 0; b < np; ++b) {
          const double val = U[Uidx(e, a, b, var)];
          sB += lL[b] * val; sT += lR[b] * val;
        }
        yFaceB[((size_t)e * np + a) * NVAR + var] = sB;
        yFaceT[((size_t)e * np + a) * NVAR + var] = sT;
      }
    }

  // 2) Halo exchange of the two x-boundary face-lines (Ny*np*NVAR each).
  //   sendL: left faces of the leftmost owned column, all rows
  //   sendR: right faces of the rightmost owned column, all rows
  //   ghostL / ghostR: the corresponding lines received from the neighbours
  //   line index [(ley*np + b)*NVAR + var]
  const size_t lineLen = (size_t)Ny_ * np * NVAR;
  std::vector<double> sendL(lineLen), sendR(lineLen), ghostL(lineLen), ghostR(lineLen);
  for (int ley = 0; ley < Ny_; ++ley)
    for (int b = 0; b < np; ++b)
      for (int var = 0; var < NVAR; ++var) {
        const int eLm = 0 + w * ley;        // leftmost owned column
        const int eRm = (w - 1) + w * ley;  // rightmost owned column
        sendL[((size_t)ley * np + b) * NVAR + var] =
            xFaceL[((size_t)eLm * np + b) * NVAR + var];
        sendR[((size_t)ley * np + b) * NVAR + var] =
            xFaceR[((size_t)eRm * np + b) * NVAR + var];
      }
  if (!part_.hasGhosts) {
    ghostL = sendR; // left neighbour is my own rightmost column (periodic)
    ghostR = sendL;
  } else {
    // ghostL = left neighbour's rightmost xFaceR -> send my sendR RIGHT, recv LEFT.
    MPI_Sendrecv(sendR.data(), (int)lineLen, MPI_DOUBLE, part_.rightRank, 0,
                 ghostL.data(), (int)lineLen, MPI_DOUBLE, part_.leftRank, 0,
                 comm_, MPI_STATUS_IGNORE);
    // ghostR = right neighbour's leftmost xFaceL -> send my sendL LEFT, recv RIGHT.
    MPI_Sendrecv(sendL.data(), (int)lineLen, MPI_DOUBLE, part_.leftRank, 1,
                 ghostR.data(), (int)lineLen, MPI_DOUBLE, part_.rightRank, 1,
                 comm_, MPI_STATUS_IGNORE);
  }

  // 3) Volume work (mirrors SGN2D::rhs; only the x-neighbour lookup differs).
  //    Scratch arrays, same meaning as in sgn2d.cpp, reused for every element.
  std::vector<double> Fnod(npp_ * NVAR), Gnod(npp_ * NVAR);
  std::vector<double> FxL(np * NVAR), FxR(np * NVAR), GyB(np * NVAR), GyT(np * NVAR);
  std::vector<double> FdXL(np * NVAR), FdXR(np * NVAR), GdYB(np * NVAR), GdYT(np * NVAR);
  std::vector<double> hSxL(np), hSxR(np), hSyB(np), hSyT(np);

  for (int e = 0; e < neL; ++e) {
    const int lx = e % w, ley = e / w;               // local column and row
    const int eB = lx + w * ((ley - 1 + Ny_) % Ny_); // bottom neighbour (periodic in y)
    const int eT = lx + w * ((ley + 1) % Ny_);       // top neighbour

    // 3a. nodal x- and y-fluxes
    for (int b = 0; b < np; ++b)
      for (int a = 0; a < np; ++a) {
        double Un[NVAR];
        for (int var = 0; var < NVAR; ++var) Un[var] = U[Uidx(e, a, b, var)];
        const double Hl = Hnod_[(size_t)e * npp_ + a + np * b];
        sgn2d::fluxX(Un, Hl, par_, &Fnod[(a + np * b) * NVAR]);
        sgn2d::fluxY(Un, Hl, par_, &Gnod[(a + np * b) * NVAR]);
      }

    // 3b. x faces: Rusanov flux, central depth and own face flux per row node b.
    //     The left / right neighbour is an owned column, or the ghost line when
    //     the element sits on the slab boundary.
    for (int b = 0; b < np; ++b) {
      const double* uL = (lx > 0)
          ? &xFaceR[((size_t)(e - 1) * np + b) * NVAR]
          : &ghostL[((size_t)ley * np + b) * NVAR];
      const double* uMe = &xFaceL[((size_t)e * np + b) * NVAR];
      const double* uMeR = &xFaceR[((size_t)e * np + b) * NVAR];
      const double* uR = (lx < w - 1)
          ? &xFaceL[((size_t)(e + 1) * np + b) * NVAR]
          : &ghostR[((size_t)ley * np + b) * NVAR];
      sgn2d::rusanov(uL, uMe, 0, HfaceXL_[(size_t)e * np + b], par_, &FxL[b * NVAR]);
      sgn2d::rusanov(uMeR, uR, 0, HfaceXR_[(size_t)e * np + b], par_, &FxR[b * NVAR]);
      hSxL[b] = 0.5 * (uL[H] + uMe[H]);
      hSxR[b] = 0.5 * (uMeR[H] + uR[H]);
      sgn2d::fluxX(uMe, HfaceXL_[(size_t)e * np + b], par_, &FdXL[b * NVAR]);
      sgn2d::fluxX(uMeR, HfaceXR_[(size_t)e * np + b], par_, &FdXR[b * NVAR]);
    }
    // 3c. y faces: always local (y is not decomposed).
    for (int a = 0; a < np; ++a) {
      const double* uB = &yFaceT[((size_t)eB * np + a) * NVAR];
      const double* uMe = &yFaceB[((size_t)e * np + a) * NVAR];
      const double* uMeT = &yFaceT[((size_t)e * np + a) * NVAR];
      const double* uT = &yFaceB[((size_t)eT * np + a) * NVAR];
      sgn2d::rusanov(uB, uMe, 1, HfaceYB_[(size_t)e * np + a], par_, &GyB[a * NVAR]);
      sgn2d::rusanov(uMeT, uT, 1, HfaceYT_[(size_t)e * np + a], par_, &GyT[a * NVAR]);
      hSyB[a] = 0.5 * (uB[H] + uMe[H]);
      hSyT[a] = 0.5 * (uMeT[H] + uT[H]);
      sgn2d::fluxY(uMe, HfaceYB_[(size_t)e * np + a], par_, &GdYB[a * NVAR]);
      sgn2d::fluxY(uMeT, HfaceYT_[(size_t)e * np + a], par_, &GdYT[a * NVAR]);
    }

    // 3d. per-node update (identical to sgn2d.cpp step 5)
    for (int b = 0; b < np; ++b)
      for (int a = 0; a < np; ++a) {
        const int loc = a + np * b;
        double Un[NVAR];
        for (int var = 0; var < NVAR; ++var) Un[var] = U[Uidx(e, a, b, var)];

        // FR-corrected divergence of F (x) and G (y)
        double dF[NVAR], dG[NVAR];
        for (int var = 0; var < NVAR; ++var) {
          double dfx = 0.0, dgy = 0.0;
          for (int k = 0; k < np; ++k) {
            dfx += D[(size_t)a * np + k] * Fnod[(k + np * b) * NVAR + var];
            dgy += D[(size_t)b * np + k] * Gnod[(a + np * k) * NVAR + var];
          }
          const double cx = (FxL[b * NVAR + var] - FdXL[b * NVAR + var]) * gLp[a] +
                            (FxR[b * NVAR + var] - FdXR[b * NVAR + var]) * gRp[a];
          const double cy = (GyB[a * NVAR + var] - GdYB[a * NVAR + var]) * gLp[b] +
                            (GyT[a * NVAR + var] - GdYT[a * NVAR + var]) * gRp[b];
          dF[var] = (dfx + cx) / Jx_;
          dG[var] = (dgy + cy) / Jy_;
        }

        // FR-corrected depth gradients with the central common depth
        double dhx = 0.0, dhy = 0.0;
        for (int k = 0; k < np; ++k) {
          dhx += D[(size_t)a * np + k] * U[Uidx(e, k, b, H)];
          dhy += D[(size_t)b * np + k] * U[Uidx(e, a, k, H)];
        }
        dhx += (hSxL[b] - xFaceL[((size_t)e * np + b) * NVAR + H]) * gLp[a] +
               (hSxR[b] - xFaceR[((size_t)e * np + b) * NVAR + H]) * gRp[a];
        dhy += (hSyB[a] - yFaceB[((size_t)e * np + a) * NVAR + H]) * gLp[b] +
               (hSyT[a] - yFaceT[((size_t)e * np + a) * NVAR + H]) * gRp[b];
        const double hx = dhx / Jx_, hy = dhy / Jy_;

        // pointwise NCPs and sources, then assemble
        double Bx[NVAR], By[NVAR], S[NVAR];
        sgn2d::ncpX(Un, hx, par_, Bx);
        sgn2d::ncpY(Un, hy, par_, By);
        sgn2d::source(Un, Hnod_[(size_t)e * npp_ + loc],
                      Hx_[(size_t)e * npp_ + loc], Hy_[(size_t)e * npp_ + loc],
                      par_, S);

        for (int var = 0; var < NVAR; ++var)
          dUdt[Uidx(e, a, b, var)] =
              -dF[var] - dG[var] - Bx[var] - By[var] + S[var];
      }
  }
}

// -----------------------------------------------------------------------------
// rk4Step: classical RK4 on the owned state (four distributed rhs calls).
// -----------------------------------------------------------------------------
void SGN2DMpi::rk4Step(State& U, double dt) const {
  const size_t n = U.size();
  State k1(n), k2(n), k3(n), k4(n), tmp(n);
  rhs(U, k1);
  for (size_t i = 0; i < n; ++i) tmp[i] = U[i] + 0.5 * dt * k1[i];
  rhs(tmp, k2);
  for (size_t i = 0; i < n; ++i) tmp[i] = U[i] + 0.5 * dt * k2[i];
  rhs(tmp, k3);
  for (size_t i = 0; i < n; ++i) tmp[i] = U[i] + dt * k3[i];
  rhs(tmp, k4);
  for (size_t i = 0; i < n; ++i)
    U[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

// -----------------------------------------------------------------------------
// integrateAdaptive: BS3(2) with a global error norm. Identical in structure to
// SGN1DMpi::integrateAdaptive (see the commented version in sgn1d_mpi.cpp).
// -----------------------------------------------------------------------------
AdaptiveRKStats SGN2DMpi::integrateAdaptive(State& U, double t0, double tEnd,
                                            double dt0,
                                            const AdaptiveRKOptions& opt) const {
  const int orderLow = 2;
  const double expo = 1.0 / (orderLow + 1.0);
  const double alpha = 0.7 * expo;   // PI exponents
  const double beta = 0.4 * expo;

  AdaptiveRKStats st;
  st.tReached = t0;
  const size_t n = U.size();         // owned degrees of freedom
  State k1(n), k2(n), k3(n), k4(n), tmp(n), uNew(n);

  double t = t0;
  double dt = std::min(dt0, opt.dtMax);
  double errPrev = 1.0;

  // global number of degrees of freedom, for the RMS error norm
  long nGlobal = 0;
  long nl = (long)n;
  MPI_Allreduce(&nl, &nGlobal, 1, MPI_LONG, MPI_SUM, comm_);

  rhs(U, k1);   // FSAL initial stage
  ++st.rhsEvals;

  while (t < tEnd - 1e-15 * std::max(1.0, std::abs(tEnd))) {
    if (t + dt > tEnd) dt = tEnd - t;

    // stages and third-order candidate
    for (size_t i = 0; i < n; ++i) tmp[i] = U[i] + 0.5 * dt * k1[i];
    rhs(tmp, k2);
    for (size_t i = 0; i < n; ++i) tmp[i] = U[i] + 0.75 * dt * k2[i];
    rhs(tmp, k3);
    for (size_t i = 0; i < n; ++i)
      uNew[i] = U[i] + dt * (2.0 / 9.0 * k1[i] + 1.0 / 3.0 * k2[i] +
                             4.0 / 9.0 * k3[i]);
    rhs(uNew, k4);
    st.rhsEvals += 3;

    // local error partials
    double localSum = 0.0, localMax = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double uEmb = U[i] + dt * (7.0 / 24.0 * k1[i] + 0.25 * k2[i] +
                                       1.0 / 3.0 * k3[i] + 0.125 * k4[i]);
      const double e = uNew[i] - uEmb;
      const double sc = opt.atol + opt.rtol * std::max(std::abs(U[i]),
                                                       std::abs(uNew[i]));
      const double r = e / sc;
      localSum += r * r;
      localMax = std::max(localMax, std::abs(r));
    }
    // combine across ranks (SUM of squares, MAX of the max-norm)
    double part[2] = {localSum, localMax}, glob[2] = {0, 0};
    MPI_Allreduce(&part[0], &glob[0], 1, MPI_DOUBLE, MPI_SUM, comm_);
    MPI_Allreduce(&part[1], &glob[1], 1, MPI_DOUBLE, MPI_MAX, comm_);
    const double errL2 = std::sqrt(glob[0] / (double)nGlobal);
    const double errNorm = std::max(errL2, 1e-300);

    if (errNorm <= 1.0) {           // accept
      t += dt;
      U.swap(uNew);
      std::swap(k1, k4);            // FSAL
      ++st.accepted;
      st.tReached = t;
      st.dtLast = dt;
      st.maxNormLast = glob[1];
      errPrev = errNorm;
    } else {                        // reject
      ++st.rejected;
    }

    // PI controller
    double fac = opt.safety * std::pow(errNorm, -alpha) *
                 std::pow(errPrev, beta);
    fac = std::clamp(fac, opt.facMin, opt.facMax);
    double dtNext = std::min(dt * fac, opt.dtMax);
    if (dtNext < opt.dtMin) { st.aborted = true; st.dtLast = dtNext; break; }
    dt = dtNext;
  }
  return st;
}

// -----------------------------------------------------------------------------
// maxWaveSpeed: local max over owned nodes and both directions, global MAX.
// -----------------------------------------------------------------------------
double SGN2DMpi::maxWaveSpeed(const State& U) const {
  const int Nl = nLocalDof();
  double lam = 0.0;
  for (int i = 0; i < Nl; ++i) {
    double Un[NVAR];
    for (int v = 0; v < NVAR; ++v) Un[v] = U[(size_t)v * Nl + i];
    lam = std::fmax(lam, std::fmax(sgn2d::maxWaveSpeed(Un, 0, par_),
                                   sgn2d::maxWaveSpeed(Un, 1, par_)));
  }
  double gl = 0.0;
  MPI_Allreduce(&lam, &gl, 1, MPI_DOUBLE, MPI_MAX, comm_);
  return gl;
}

// -----------------------------------------------------------------------------
// cflTimeStep: same formula as SGN2D with the global max wave speed.
// -----------------------------------------------------------------------------
double SGN2DMpi::cflTimeStep(const State& U, double cfl) const {
  const double lam = maxWaveSpeed(U);
  const double dxEff = std::fmin(hx_, hy_) / (2.0 * P_ + 1.0);
  return cfl * dxEff / (lam > 0.0 ? lam : 1e-300);
}

// -----------------------------------------------------------------------------
// mass: owned integral of h, summed over ranks.
// -----------------------------------------------------------------------------
double SGN2DMpi::mass(const State& U) const {
  const int Nl = nLocalDof();
  const auto& qw = op_.weights();
  double m = 0.0;
  for (int e = 0; e < nOwnedElem(); ++e)
    for (int b = 0; b < np_; ++b)
      for (int a = 0; a < np_; ++a)
        m += qw[a] * qw[b] * U[(size_t)H * Nl + (size_t)e * npp_ + a + np_ * b];
  m *= Jx_ * Jy_;
  double gm = 0.0;
  MPI_Allreduce(&m, &gm, 1, MPI_DOUBLE, MPI_SUM, comm_);
  return gm;
}

} // namespace sgn
