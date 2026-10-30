// =============================================================================
// sgn1d_mpi.cpp -- implementation of the distributed 1D FR solver (MPI, CPU)
// =============================================================================
//
// The arithmetic on every owned element is written to be byte-identical to
// SGN1D::rhs (src/solver/sgn1d.cpp); the only difference is where the state of
// the neighbour element across the slab boundary comes from (a ghost value
// received by MPI instead of an array read). Terminology:
//   owned element  element e in [0, w) of this rank (global id a + e)
//   ghostL/ghostR  face state of the element just outside the slab on the
//                  left / right, received from the neighbouring rank
// =============================================================================

#include "sgn1d_mpi.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sgn {

// -----------------------------------------------------------------------------
// Constructor: find this rank's slab and precompute the owned bathymetry.
// -----------------------------------------------------------------------------
SGN1DMpi::SGN1DMpi(int P, int nElemGlobal, double domainLength,
                   const Params& par, MPI_Comm comm,
                   std::function<double(double)> stillDepth, PointSet pointSet)
    : P_(P), neGlobal_(nElemGlobal), L_(domainLength),
      h_(domainLength / nElemGlobal), J_(domainLength / nElemGlobal / 2.0),
      par_(par), comm_(comm), op_(P, pointSet) {
  // Step 1: who am I (rank) and how many of us are there (size)
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (nElemGlobal < size)
    throw std::invalid_argument("SGN1DMpi: need nElemGlobal >= nranks");
  // Step 2: this rank's owned element range and ring neighbours
  part_ = makeSlabPartition1D(nElemGlobal, rank, size);

  // Step 3: bathymetry on the owned nodes (same formulas as SGN1D)
  const int np = P_ + 1;
  const int Nl = nLocalDof();
  Hnod_.assign(Nl, 0.0);
  Hx_.assign(Nl, 0.0);
  HfaceL_.assign(part_.w, 0.0);
  HfaceR_.assign(part_.w, 0.0);
  if (stillDepth) {
    if (pointSet != PointSet::GaussLobatto)
      throw std::invalid_argument("SGN1DMpi: bathymetry requires GLL points");
    // 3a. nodal H
    const auto x = xCoordsLocal();
    for (int i = 0; i < Nl; ++i) Hnod_[i] = stillDepth(x[i]);
    // 3b. nodal d_x H, element-local differentiation / Jacobian
    const auto& D = op_.D();
    for (int e = 0; e < part_.w; ++e)
      for (int j = 0; j < np; ++j) {
        double s = 0.0;
        for (int k = 0; k < np; ++k)
          s += D[(size_t)j * np + k] * Hnod_[e * np + k];
        Hx_[e * np + j] = s / J_;
      }
    // 3c. still depth at both faces of every owned element
    for (int e = 0; e < part_.w; ++e) {
      const int gElem = part_.ownedGlobal(e);
      // Periodic wrap: match the serial driver bitwise (the last interface is
      // evaluated at the wrapped coordinate x=0, not x=L).
      HfaceL_[e] = stillDepth(gElem * h_);
      HfaceR_[e] = stillDepth(((gElem + 1) % neGlobal_) * h_);
    }
  }
}

// -----------------------------------------------------------------------------
// xCoordsLocal: global physical x of every owned node.
// -----------------------------------------------------------------------------
std::vector<double> SGN1DMpi::xCoordsLocal() const {
  const int np = P_ + 1;
  const auto& xi = op_.nodes();
  std::vector<double> x(nLocalDof());
  for (int e = 0; e < part_.w; ++e) {
    const double xe = part_.ownedGlobal(e) * h_; // left edge (global coordinate)
    for (int j = 0; j < np; ++j)
      x[e * np + j] = xe + 0.5 * h_ * (xi[j] + 1.0);
  }
  return x;
}

// -----------------------------------------------------------------------------
// makeStateLocal: owned primitive fields -> owned var-major state.
// -----------------------------------------------------------------------------
State SGN1DMpi::makeStateLocal(const std::vector<double>& h,
                               const std::vector<double>& u,
                               const std::vector<double>& w,
                               const std::vector<double>& p) const {
  const int Nl = nLocalDof();
  if ((int)h.size() != Nl || (int)u.size() != Nl || (int)w.size() != Nl ||
      (int)p.size() != Nl)
    throw std::invalid_argument("makeStateLocal: size mismatch");
  State U(nLocalTotal());
  for (int i = 0; i < Nl; ++i) {
    double Un[NVAR];
    fromPrimitive(h[i], u[i], w[i], p[i], Un);
    for (int v = 0; v < NVAR; ++v) U[v * Nl + i] = Un[v];
  }
  return U;
}

// -----------------------------------------------------------------------------
// exchangeFaces: the halo exchange.
//   UfL, UfR  face states of all owned elements (index e*NVAR+v)
//   ghostL    OUT: right-face state of the element left of the slab
//   ghostR    OUT: left-face state of the element right of the slab
// -----------------------------------------------------------------------------
void SGN1DMpi::exchangeFaces(const double* UfL, const double* UfR,
                             double* ghostL, double* ghostR) const {
  const int w = part_.w;
  if (!part_.hasGhosts) {
    // size==1: the block's neighbours are its own opposite ends (periodic).
    for (int v = 0; v < NVAR; ++v) {
      ghostL[v] = UfR[(w - 1) * NVAR + v]; // left neighbour = own rightmost
      ghostR[v] = UfL[0 * NVAR + v];       // right neighbour = own leftmost
    }
    return;
  }
  // MPI_Sendrecv sends one buffer and receives another in a single call, so a
  // ring of ranks can all exchange simultaneously without deadlock.
  //
  // Exchange 1 (tag 0): every rank sends its rightmost element's right face to
  // the RIGHT neighbour, and receives from the LEFT neighbour -> ghostL.
  MPI_Sendrecv(const_cast<double*>(&UfR[(w - 1) * NVAR]), NVAR, MPI_DOUBLE,
               part_.rightRank, 0, ghostL, NVAR, MPI_DOUBLE, part_.leftRank, 0,
               comm_, MPI_STATUS_IGNORE);
  // Exchange 2 (tag 1): every rank sends its leftmost element's left face to
  // the LEFT neighbour, and receives from the RIGHT neighbour -> ghostR.
  MPI_Sendrecv(const_cast<double*>(&UfL[0]), NVAR, MPI_DOUBLE, part_.leftRank, 1,
               ghostR, NVAR, MPI_DOUBLE, part_.rightRank, 1, comm_,
               MPI_STATUS_IGNORE);
}

// -----------------------------------------------------------------------------
// rhs: distributed right-hand side of the owned state.
// -----------------------------------------------------------------------------
void SGN1DMpi::rhs(const State& U, State& dUdt) const {
  const int np = P_ + 1;
  const int w = part_.w;       // owned elements
  const int Nl = nLocalDof();  // owned nodes per variable
  if ((int)U.size() != nLocalTotal())
    throw std::invalid_argument("SGN1DMpi::rhs: size mismatch");
  dUdt.assign(nLocalTotal(), 0.0);

  // FR operators
  const auto& D = op_.D();
  const auto& lL = op_.lL();
  const auto& lR = op_.lR();
  const auto& gLp = op_.gLprime();
  const auto& gRp = op_.gRprime();

  // 1) Owned face states: interpolate each owned element to its two faces.
  std::vector<double> UfL(w * NVAR), UfR(w * NVAR);
  for (int e = 0; e < w; ++e)
    for (int v = 0; v < NVAR; ++v) {
      double sL = 0.0, sR = 0.0;
      const double* Uv = &U[v * Nl + e * np];
      for (int j = 0; j < np; ++j) { sL += lL[j] * Uv[j]; sR += lR[j] * Uv[j]; }
      UfL[e * NVAR + v] = sL;
      UfR[e * NVAR + v] = sR;
    }

  // 2) Halo exchange of the two boundary face states (the only communication).
  double ghostL[NVAR], ghostR[NVAR];
  exchangeFaces(UfL.data(), UfR.data(), ghostL, ghostR);

  // 3) Volume work per owned element (mirrors SGN1D::rhs exactly).
  std::vector<double> Fnod(np * NVAR);
  for (int e = 0; e < w; ++e) {
    // Neighbour face states on each side: an owned element when available,
    // otherwise the ghost received from the neighbouring rank.
    const double* lnUfR = (e > 0) ? &UfR[(e - 1) * NVAR] : ghostL;
    const double* rnUfL = (e < w - 1) ? &UfL[(e + 1) * NVAR] : ghostR;

    // Left/right interface Rusanov fluxes and central depths h*.
    double FstarL[NVAR], FstarR[NVAR];
    rusanov(lnUfR, &UfL[e * NVAR], HfaceL_[e], par_, FstarL);
    rusanov(&UfR[e * NVAR], rnUfL, HfaceR_[e], par_, FstarR);
    const double hStarL = 0.5 * (lnUfR[H] + UfL[e * NVAR + H]);
    const double hStarR = 0.5 * (UfR[e * NVAR + H] + rnUfL[H]);

    // Nodal flux at every node of the element.
    for (int j = 0; j < np; ++j) {
      double Un[NVAR];
      for (int v = 0; v < NVAR; ++v) Un[v] = U[v * Nl + e * np + j];
      fluxX(Un, Hnod_[e * np + j], par_, &Fnod[j * NVAR]);
    }
    // Discontinuous face fluxes (same face bathymetry as Rusanov).
    double FdiscL[NVAR], FdiscR[NVAR];
    fluxX(&UfL[e * NVAR], HfaceL_[e], par_, FdiscL);
    fluxX(&UfR[e * NVAR], HfaceR_[e], par_, FdiscR);
    const double hFaceL = UfL[e * NVAR + H];
    const double hFaceR = UfR[e * NVAR + H];

    for (int i = 0; i < np; ++i) {
      const int gloc = e * np + i;   // owned (local) node index
      // FR-corrected flux divergence
      double dF[NVAR];
      for (int v = 0; v < NVAR; ++v) {
        double dfdxi = 0.0;
        for (int k = 0; k < np; ++k)
          dfdxi += D[(size_t)i * np + k] * Fnod[k * NVAR + v];
        const double corr = (FstarL[v] - FdiscL[v]) * gLp[i] +
                            (FstarR[v] - FdiscR[v]) * gRp[i];
        dF[v] = (dfdxi + corr) / J_;
      }
      // FR-corrected depth gradient with the central common depth
      double dhdxi = 0.0;
      for (int k = 0; k < np; ++k)
        dhdxi += D[(size_t)i * np + k] * U[H * Nl + e * np + k];
      dhdxi += (hStarL - hFaceL) * gLp[i] + (hStarR - hFaceR) * gRp[i];
      const double hx = dhdxi / J_;

      // pointwise NCP and sources, then dU/dt = -dF/dx - B + S
      double Un[NVAR], B[NVAR], S[NVAR];
      for (int v = 0; v < NVAR; ++v) Un[v] = U[v * Nl + gloc];
      ncpX(Un, hx, par_, B);
      source(Un, Hnod_[gloc], Hx_[gloc], par_, S);
      for (int v = 0; v < NVAR; ++v) dUdt[v * Nl + gloc] = -dF[v] - B[v] + S[v];
    }
  }
}

// -----------------------------------------------------------------------------
// rk4Step: classical RK4 on the owned state (four distributed rhs calls).
// -----------------------------------------------------------------------------
void SGN1DMpi::rk4Step(State& U, double dt) const {
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
// integrateAdaptive: the adaptive BS3(2) of rk.cpp with a global error norm.
// The only differences from sgn::integrateAdaptive are marked [MPI]: the error
// is accumulated per rank and combined across ranks, so every rank sees the
// same norm and takes the same accept/reject decision and next dt.
// -----------------------------------------------------------------------------
AdaptiveRKStats SGN1DMpi::integrateAdaptive(State& U, double t0, double tEnd,
                                            double dt0,
                                            const AdaptiveRKOptions& opt) const {
  const int orderLow = 2;
  const double expo = 1.0 / (orderLow + 1.0);
  const double alpha = 0.7 * expo;   // PI exponents, as in rk.cpp
  const double beta = 0.4 * expo;

  AdaptiveRKStats st;
  st.tReached = t0;
  const size_t n = U.size();          // OWNED degrees of freedom on this rank
  State k1(n), k2(n), k3(n), k4(n), tmp(n), uNew(n);

  double t = t0;
  double dt = std::min(dt0, opt.dtMax);
  double errPrev = 1.0;

  // [MPI] total number of degrees of freedom over all ranks (for the RMS norm)
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

    // local contributions to the error norm (sum of squares and max)
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
    // [MPI] combine over ranks: sum of squares (SUM) and max (MAX). MAX is exact
    // and order-independent; SUM is not, which is why step sequences are an
    // observed rather than guaranteed invariant across rank counts.
    double part[2] = {localSum, localMax}, glob[2] = {0, 0};
    MPI_Allreduce(&part[0], &glob[0], 1, MPI_DOUBLE, MPI_SUM, comm_);
    MPI_Allreduce(&part[1], &glob[1], 1, MPI_DOUBLE, MPI_MAX, comm_);
    const double errL2 = std::sqrt(glob[0] / (double)nGlobal);
    const double errMax = glob[1];
    const double errNorm = std::max(errL2, 1e-300);

    if (errNorm <= 1.0) {           // accept
      t += dt;
      U.swap(uNew);
      std::swap(k1, k4);            // FSAL
      ++st.accepted;
      st.tReached = t;
      st.dtLast = dt;
      st.maxNormLast = errMax;
      errPrev = errNorm;
    } else {                        // reject and retry with a smaller dt
      ++st.rejected;
    }

    // PI controller (identical to rk.cpp)
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
// maxWaveSpeed: local max over owned nodes, then global max over ranks.
// -----------------------------------------------------------------------------
double SGN1DMpi::maxWaveSpeed(const State& U) const {
  const int Nl = nLocalDof();
  double lam = 0.0;
  for (int i = 0; i < Nl; ++i) {
    double Un[NVAR];
    for (int v = 0; v < NVAR; ++v) Un[v] = U[v * Nl + i];
    lam = std::fmax(lam, sgn::maxWaveSpeed(Un, par_));
  }
  double gl = 0.0;
  MPI_Allreduce(&lam, &gl, 1, MPI_DOUBLE, MPI_MAX, comm_);
  return gl;
}

// -----------------------------------------------------------------------------
// cflTimeStep: identical formula to SGN1D, with the global max wave speed.
// -----------------------------------------------------------------------------
double SGN1DMpi::cflTimeStep(const State& U, double cfl) const {
  const double lam = maxWaveSpeed(U);
  const double dxEff = h_ / (2.0 * P_ + 1.0);
  return cfl * dxEff / (lam > 0.0 ? lam : 1e-300);
}

// -----------------------------------------------------------------------------
// mass: owned integral of h, summed over ranks.
// -----------------------------------------------------------------------------
double SGN1DMpi::mass(const State& U) const {
  const int np = P_ + 1, Nl = nLocalDof();
  const auto& qw = op_.weights();
  double m = 0.0;
  for (int e = 0; e < part_.w; ++e)
    for (int j = 0; j < np; ++j) m += qw[j] * U[H * Nl + e * np + j];
  m *= J_;
  double gm = 0.0;
  MPI_Allreduce(&m, &gm, 1, MPI_DOUBLE, MPI_SUM, comm_);
  return gm;
}

// -----------------------------------------------------------------------------
// momentum: owned integral of hu, summed over ranks.
// -----------------------------------------------------------------------------
double SGN1DMpi::momentum(const State& U) const {
  const int np = P_ + 1, Nl = nLocalDof();
  const auto& qw = op_.weights();
  double m = 0.0;
  for (int e = 0; e < part_.w; ++e)
    for (int j = 0; j < np; ++j) m += qw[j] * U[HU * Nl + e * np + j];
  m *= J_;
  double gm = 0.0;
  MPI_Allreduce(&m, &gm, 1, MPI_DOUBLE, MPI_SUM, comm_);
  return gm;
}

} // namespace sgn
