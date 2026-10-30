// =============================================================================
// sgn2d.cpp -- implementation of the 2D CPU tensor-product FR solver
// =============================================================================
//
// See sgn2d.h for the data layout. rhs() works element by element:
//   1. (all elements) extrapolate the state to the four faces of every element
//   2. (per element) nodal x- and y-fluxes
//   3. (per element) Rusanov fluxes and central depths on its four face lines
//   4. (per element) the element's own discontinuous face fluxes
//   5. (per node)    FR-corrected divergence, FR-corrected depth gradients,
//                    non-conservative products and sources, assembled into
//                    dU/dt = -dF/dx - dG/dy - Bx - By + S
// Every interface flux is evaluated once from each side (twice in total); the
// two evaluations use identical inputs and so give identical results.
// =============================================================================

#include "sgn2d.h"

#include <cmath>
#include <stdexcept>

namespace sgn {

using sgn2d::NVAR;
using sgn2d::H;

// -----------------------------------------------------------------------------
// Constructor: mesh, operators, and bathymetry precomputation.
// -----------------------------------------------------------------------------
SGN2D::SGN2D(int P, int Nx, int Ny, double Lx, double Ly,
             const sgn2d::Params& par,
             std::function<double(double, double)> stillDepth,
             PointSet pointSet)
    : P_(P), np_(P + 1), npp_((P + 1) * (P + 1)), par_(par),
      mesh_(Nx, Ny, Lx, Ly), op_(P, pointSet) {
  if (Nx < 2 || Ny < 2) throw std::invalid_argument("SGN2D: Nx,Ny >= 2");

  // Step 1: bathymetry arrays, zero = flat bottom
  const int Nd = nDof();
  Hnod_.assign(Nd, 0.0);
  Hx_.assign(Nd, 0.0);
  Hy_.assign(Nd, 0.0);
  HfaceX_.assign(nElem() * np_, 0.0);
  HfaceY_.assign(nElem() * np_, 0.0);
  if (!stillDepth) return;                 // flat bottom: nothing else to do
  if (pointSet != PointSet::GaussLobatto)
    throw std::invalid_argument("SGN2D: bathymetry requires GLL points");

  // Step 2: still depth at every node
  std::vector<double> x, y;
  nodeCoords(x, y);
  for (int i = 0; i < Nd; ++i) Hnod_[i] = stillDepth(x[i], y[i]);

  // Step 3: nodal gradients by element-local tensor FR differentiation of H.
  //   d_x H at node (a,b): 1D derivative along the x-line through b
  //   d_y H at node (a,b): 1D derivative along the y-line through a
  const auto& D = op_.D();
  const double Jx = mesh_.Jx(), Jy = mesh_.Jy();
  for (int e = 0; e < nElem(); ++e) {
    const double* He = &Hnod_[e * npp_];   // nodal H of element e
    for (int b = 0; b < np_; ++b)
      for (int a = 0; a < np_; ++a) {
        double dHx = 0.0, dHy = 0.0;
        for (int k = 0; k < np_; ++k) {
          dHx += D[(size_t)a * np_ + k] * He[k + np_ * b]; // x-line
          dHy += D[(size_t)b * np_ + k] * He[a + np_ * k]; // y-line
        }
        Hx_[e * npp_ + a + np_ * b] = dHx / Jx;
        Hy_[e * npp_ + a + np_ * b] = dHy / Jy;
      }
  }
  // Step 4: face still-depths, sampled exactly on the element edge at the
  // transverse node positions (so they coincide with the edge nodes).
  const auto& xi = op_.nodes();
  for (int e = 0; e < nElem(); ++e) {
    const int ex = mesh_.ex(e), ey = mesh_.ey(e);
    for (int t = 0; t < np_; ++t) {
      const double yb = ey * mesh_.hy + 0.5 * mesh_.hy * (xi[t] + 1.0); // y of node t
      const double xa = ex * mesh_.hx + 0.5 * mesh_.hx * (xi[t] + 1.0); // x of node t
      HfaceX_[e * np_ + t] = stillDepth(ex * mesh_.hx, yb);       // left edge
      HfaceY_[e * np_ + t] = stillDepth(xa, ey * mesh_.hy);       // bottom edge
    }
  }
}

// -----------------------------------------------------------------------------
// nodeCoords: physical (x, y) of every node, affine map per direction.
// -----------------------------------------------------------------------------
void SGN2D::nodeCoords(std::vector<double>& x, std::vector<double>& y) const {
  x.assign(nDof(), 0.0);
  y.assign(nDof(), 0.0);
  const auto& xi = op_.nodes();
  for (int e = 0; e < nElem(); ++e) {
    const int ex = mesh_.ex(e), ey = mesh_.ey(e);   // element column and row
    for (int b = 0; b < np_; ++b)
      for (int a = 0; a < np_; ++a) {
        const int g = e * npp_ + a + np_ * b;
        x[g] = ex * mesh_.hx + 0.5 * mesh_.hx * (xi[a] + 1.0);
        y[g] = ey * mesh_.hy + 0.5 * mesh_.hy * (xi[b] + 1.0);
      }
  }
}

// -----------------------------------------------------------------------------
// makeState: nodal primitives -> var-major conserved state.
// -----------------------------------------------------------------------------
State SGN2D::makeState(const std::vector<double>& h,
                       const std::vector<double>& u,
                       const std::vector<double>& v,
                       const std::vector<double>& w,
                       const std::vector<double>& p) const {
  const int Nd = nDof();
  State U(nTotal());
  for (int i = 0; i < Nd; ++i) {
    double Un[NVAR];
    sgn2d::fromPrimitive(h[i], u[i], v[i], w[i], p[i], Un);
    for (int vv = 0; vv < NVAR; ++vv) U[vv * Nd + i] = Un[vv];
  }
  return U;
}

// -----------------------------------------------------------------------------
// rhs: semi-discrete right-hand side dU/dt = R(U).
// -----------------------------------------------------------------------------
void SGN2D::rhs(const State& U, State& dUdt) const {
  const int Nd = nDof();
  const int np = np_;
  if ((int)U.size() != nTotal())
    throw std::invalid_argument("SGN2D::rhs: size mismatch");
  dUdt.assign(nTotal(), 0.0);

  // 1D FR operators (applied along x-lines and y-lines)
  const auto& D = op_.D();
  const auto& lL = op_.lL();
  const auto& lR = op_.lR();
  const auto& gLp = op_.gLprime();
  const auto& gRp = op_.gRprime();
  const double Jx = mesh_.Jx(), Jy = mesh_.Jy();
  const int ne = nElem();

  // --- 1) Face-extrapolated states (per element, per transverse node, per var)
  //   xFaceL/R: index [ (e*np + b)*NVAR + var ]  state on the left/right face of
  //             element e at transverse (y) node b
  //   yFaceB/T: index [ (e*np + a)*NVAR + var ]  state on the bottom/top face at
  //             transverse (x) node a
  std::vector<double> xFaceL((size_t)ne * np * NVAR), xFaceR((size_t)ne * np * NVAR);
  std::vector<double> yFaceB((size_t)ne * np * NVAR), yFaceT((size_t)ne * np * NVAR);

  // position of variable var at node (a, b) of element e in the flat state
  auto Uidx = [&](int e, int a, int b, int var) {
    return (size_t)var * Nd + e * npp_ + a + np * b;
  };

  for (int e = 0; e < ne; ++e) {
    for (int var = 0; var < NVAR; ++var) {
      // x-faces: interpolate along a (x) for each transverse b.
      for (int b = 0; b < np; ++b) {
        double sL = 0.0, sR = 0.0;
        for (int a = 0; a < np; ++a) {
          const double val = U[Uidx(e, a, b, var)];
          sL += lL[a] * val; sR += lR[a] * val;
        }
        xFaceL[((size_t)e * np + b) * NVAR + var] = sL;
        xFaceR[((size_t)e * np + b) * NVAR + var] = sR;
      }
      // y-faces: interpolate along b (y) for each transverse a.
      for (int a = 0; a < np; ++a) {
        double sB = 0.0, sT = 0.0;
        for (int b = 0; b < np; ++b) {
          const double val = U[Uidx(e, a, b, var)];
          sB += lL[b] * val; sT += lR[b] * val;
        }
        yFaceB[((size_t)e * np + a) * NVAR + var] = sB;
        yFaceT[((size_t)e * np + a) * NVAR + var] = sT;
      }
    }
  }

  // --- 2..5) Volume work, element by element ------------------------------------
  // Fnod / Gnod: nodal x- and y-fluxes of the current element, index
  // [loc*NVAR + var] with loc = a + np*b (scratch reused per element).
  std::vector<double> Fnod(npp_ * NVAR), Gnod(npp_ * NVAR);
  for (int e = 0; e < ne; ++e) {
    // the four periodic neighbours of e
    const int eL = mesh_.neighbor(e, 3), eR = mesh_.neighbor(e, 1);
    const int eB = mesh_.neighbor(e, 0), eT = mesh_.neighbor(e, 2);

    // 2) Nodal fluxes at every node of the element.
    for (int b = 0; b < np; ++b)
      for (int a = 0; a < np; ++a) {
        double Un[NVAR];
        for (int var = 0; var < NVAR; ++var) Un[var] = U[Uidx(e, a, b, var)];
        const double Hl = Hnod_[e * npp_ + a + np * b];
        sgn2d::fluxX(Un, Hl, par_, &Fnod[(a + np * b) * NVAR]);
        sgn2d::fluxY(Un, Hl, par_, &Gnod[(a + np * b) * NVAR]);
      }

    // 3) Interface Rusanov fluxes + central depths on the four face lines.
    //   FxL/FxR[b*NVAR+var]  common flux on the left/right face at y-node b
    //   GyB/GyT[a*NVAR+var]  common flux on the bottom/top face at x-node a
    //   hSxL/hSxR[b], hSyB/hSyT[a]  central common depth h* on the same faces
    std::vector<double> FxL(np * NVAR), FxR(np * NVAR), GyB(np * NVAR), GyT(np * NVAR);
    std::vector<double> hSxL(np), hSxR(np), hSyB(np), hSyT(np);
    for (int b = 0; b < np; ++b) {
      const double* uL = &xFaceR[((size_t)eL * np + b) * NVAR]; // left nbr right face
      const double* uMe = &xFaceL[((size_t)e * np + b) * NVAR]; // own left face
      const double* uR = &xFaceL[((size_t)eR * np + b) * NVAR]; // right nbr left face
      const double* uMeR = &xFaceR[((size_t)e * np + b) * NVAR]; // own right face
      const double HxL = HfaceX_[e * np + b];        // left edge of e
      const double HxR = HfaceX_[eR * np + b];       // left edge of right nbr = right edge of e
      sgn2d::rusanov(uL, uMe, 0, HxL, par_, &FxL[b * NVAR]);
      sgn2d::rusanov(uMeR, uR, 0, HxR, par_, &FxR[b * NVAR]);
      hSxL[b] = 0.5 * (uL[H] + uMe[H]);
      hSxR[b] = 0.5 * (uMeR[H] + uR[H]);
    }
    for (int a = 0; a < np; ++a) {
      const double* uB = &yFaceT[((size_t)eB * np + a) * NVAR]; // bottom nbr top face
      const double* uMe = &yFaceB[((size_t)e * np + a) * NVAR]; // own bottom face
      const double* uT = &yFaceB[((size_t)eT * np + a) * NVAR]; // top nbr bottom face
      const double* uMeT = &yFaceT[((size_t)e * np + a) * NVAR]; // own top face
      const double HyB = HfaceY_[e * np + a];        // bottom edge of e
      const double HyT = HfaceY_[eT * np + a];       // bottom edge of top nbr = top edge of e
      sgn2d::rusanov(uB, uMe, 1, HyB, par_, &GyB[a * NVAR]);
      sgn2d::rusanov(uMeT, uT, 1, HyT, par_, &GyT[a * NVAR]);
      hSyB[a] = 0.5 * (uB[H] + uMe[H]);
      hSyT[a] = 0.5 * (uMeT[H] + uT[H]);
    }

    // 4) The element's own discontinuous face fluxes (same face bathymetry as
    //    the Rusanov flux, so the correction vanishes for a continuous field).
    std::vector<double> FdXL(np * NVAR), FdXR(np * NVAR), GdYB(np * NVAR), GdYT(np * NVAR);
    for (int b = 0; b < np; ++b) {
      sgn2d::fluxX(&xFaceL[((size_t)e * np + b) * NVAR], HfaceX_[e * np + b], par_, &FdXL[b * NVAR]);
      sgn2d::fluxX(&xFaceR[((size_t)e * np + b) * NVAR], HfaceX_[eR * np + b], par_, &FdXR[b * NVAR]);
    }
    for (int a = 0; a < np; ++a) {
      sgn2d::fluxY(&yFaceB[((size_t)e * np + a) * NVAR], HfaceY_[e * np + a], par_, &GdYB[a * NVAR]);
      sgn2d::fluxY(&yFaceT[((size_t)e * np + a) * NVAR], HfaceY_[eT * np + a], par_, &GdYT[a * NVAR]);
    }

    // 5) Per-node update.
    for (int b = 0; b < np; ++b)
      for (int a = 0; a < np; ++a) {
        const int loc = a + np * b;
        double Un[NVAR];
        for (int var = 0; var < NVAR; ++var) Un[var] = U[Uidx(e, a, b, var)];

        // 5a. FR-corrected divergence: x-line derivative of F plus its face
        //     correction, and y-line derivative of G plus its face correction
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
          dF[var] = (dfx + cx) / Jx;
          dG[var] = (dgy + cy) / Jy;
        }

        // 5b. FR-corrected depth gradients (central common h at faces).
        double dhx = 0.0, dhy = 0.0;
        for (int k = 0; k < np; ++k) {
          dhx += D[(size_t)a * np + k] * U[Uidx(e, k, b, H)];
          dhy += D[(size_t)b * np + k] * U[Uidx(e, a, k, H)];
        }
        dhx += (hSxL[b] - xFaceL[((size_t)e * np + b) * NVAR + H]) * gLp[a] +
               (hSxR[b] - xFaceR[((size_t)e * np + b) * NVAR + H]) * gRp[a];
        dhy += (hSyB[a] - yFaceB[((size_t)e * np + a) * NVAR + H]) * gLp[b] +
               (hSyT[a] - yFaceT[((size_t)e * np + a) * NVAR + H]) * gRp[b];
        const double hx = dhx / Jx, hy = dhy / Jy;

        // 5c. pointwise non-conservative products and algebraic sources
        double Bx[NVAR], By[NVAR], S[NVAR];
        sgn2d::ncpX(Un, hx, par_, Bx);
        sgn2d::ncpY(Un, hy, par_, By);
        sgn2d::source(Un, Hnod_[e * npp_ + loc], Hx_[e * npp_ + loc],
                      Hy_[e * npp_ + loc], par_, S);

        // 5d. assemble dU/dt = -dF/dx - dG/dy - Bx - By + S
        for (int var = 0; var < NVAR; ++var)
          dUdt[Uidx(e, a, b, var)] =
              -dF[var] - dG[var] - Bx[var] - By[var] + S[var];
      }
  }
}

// -----------------------------------------------------------------------------
// maxWaveSpeed: largest spectral radius over all nodes and both directions.
// -----------------------------------------------------------------------------
double SGN2D::maxWaveSpeed(const State& U) const {
  const int Nd = nDof();
  double lam = 0.0;
  for (int i = 0; i < Nd; ++i) {
    double Un[NVAR];
    for (int v = 0; v < NVAR; ++v) Un[v] = U[(size_t)v * Nd + i];
    lam = std::fmax(lam, std::fmax(sgn2d::maxWaveSpeed(Un, 0, par_),
                                   sgn2d::maxWaveSpeed(Un, 1, par_)));
  }
  return lam;
}

// -----------------------------------------------------------------------------
// cflTimeStep: dt = cfl * dxEff / lambda, dxEff from the smaller element size.
// -----------------------------------------------------------------------------
double SGN2D::cflTimeStep(const State& U, double cfl) const {
  const double lam = maxWaveSpeed(U);
  const double dxEff = std::fmin(mesh_.hx, mesh_.hy) / (2.0 * P_ + 1.0);
  return cfl * dxEff / (lam > 0.0 ? lam : 1e-300);
}

// -----------------------------------------------------------------------------
// mass: integral of h, tensor-product GLL quadrature times the area Jacobian.
// -----------------------------------------------------------------------------
double SGN2D::mass(const State& U) const {
  const int Nd = nDof();
  const auto& qw = op_.weights();
  const double JJ = mesh_.Jx() * mesh_.Jy();   // reference -> physical area
  double m = 0.0;
  for (int e = 0; e < nElem(); ++e)
    for (int b = 0; b < np_; ++b)
      for (int a = 0; a < np_; ++a)
        m += qw[a] * qw[b] * U[(size_t)H * Nd + e * npp_ + a + np_ * b];
  return m * JJ;
}

// -----------------------------------------------------------------------------
// energy: integral of the mechanical energy density (diagnostic).
// -----------------------------------------------------------------------------
double SGN2D::energy(const State& U) const {
  const int Nd = nDof();
  const auto& qw = op_.weights();
  const double JJ = mesh_.Jx() * mesh_.Jy();
  double E = 0.0;
  for (int e = 0; e < nElem(); ++e)
    for (int b = 0; b < np_; ++b)
      for (int a = 0; a < np_; ++a) {
        const int loc = a + np_ * b;
        double Un[NVAR];
        for (int v = 0; v < NVAR; ++v) Un[v] = U[(size_t)v * Nd + e * npp_ + loc];
        E += qw[a] * qw[b] * sgn2d::energyDensity(Un, Hnod_[e * npp_ + loc], par_);
      }
  return E * JJ;
}

} // namespace sgn
