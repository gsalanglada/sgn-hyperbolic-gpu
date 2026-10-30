// =============================================================================
// sgn1d.cpp -- implementation of the 1D CPU Flux Reconstruction solver
// =============================================================================
//
// See sgn1d.h for the discretization and the data layout. The central method is
// rhs(), which evaluates the semi-discrete right-hand side in four phases:
//   1. extrapolate every element's state to its two faces
//   2. at every interface: Rusanov flux and the common depth h* for the NCP
//   3. inside every element: FR-corrected flux divergence, FR-corrected depth
//      gradient, non-conservative product and algebraic sources, assembled into
//      dU/dt = -dF/dx - B + S at each node
//   4. relaxation zones (wave generation / absorption), if any
// =============================================================================

#include "sgn1d.h"

#include <cmath>
#include <stdexcept>

namespace sgn {

// -----------------------------------------------------------------------------
// Constructor: store the discretization parameters, build the FR operators of
// degree P and precompute everything that depends on the bathymetry.
// -----------------------------------------------------------------------------
SGN1D::SGN1D(int P, int nElem, double domainLength, const Params& par,
             std::function<double(double)> stillDepth, PointSet pointSet)
    : P_(P), nElem_(nElem), L_(domainLength),
      h_(domainLength / nElem), J_(domainLength / nElem / 2.0),
      par_(par), op_(P, pointSet) {

  // Step 1: at least 2 elements, so that every interface has two distinct sides
  if (nElem < 2) throw std::invalid_argument("SGN1D: nElem >= 2");

  // Step 2: allocate the bathymetry arrays (all zero = flat bottom, H = 0)
  const int N = nDof(); // number of nodes per variable
  Hnod_.assign(N, 0.0);
  Hx_.assign(N, 0.0);
  Hface_.assign(nElem_, 0.0);

  // Step 3: if a bathymetry H(x) was provided, sample and differentiate it
  if (stillDepth) {
    // 3a. faces must be nodes, so both sides of a face see the same H
    if (pointSet != PointSet::GaussLobatto)
      throw std::invalid_argument(
          "SGN1D: bathymetry requires GLL points (faces must be nodes)");

    // 3b. still depth at every node
    const auto x = xCoords();
    for (int i = 0; i < N; ++i) Hnod_[i] = stillDepth(x[i]);

    // 3c. nodal gradient Hx = d_x H, element by element: apply the
    //     differentiation matrix to the element's nodal H (no interface
    //     correction: H is smooth and continuous) and divide by the Jacobian
    const int np = P_ + 1; // nodes per element
    const auto& D = op_.D();
    for (int e = 0; e < nElem_; ++e) {
      for (int j = 0; j < np; ++j) {
        double s = 0.0;
        // (D H)_j = sum_k D[j][k] H_k   (derivative in reference coordinate xi)
        for (int k = 0; k < np; ++k)
          s += D[(size_t)j * np + k] * Hnod_[e * np + k];
        // d/dx = (d/dxi) / J
        Hx_[e * np + j] = s / J_;
      }
    }

    // 3d. still depth at every interface I (located at x = I*h)
    for (int I = 0; I < nElem_; ++I) Hface_[I] = stillDepth(I * h_);
  }
}

// -----------------------------------------------------------------------------
// xCoords: physical x of every node, x[e*(P+1)+j].
// -----------------------------------------------------------------------------
std::vector<double> SGN1D::xCoords() const {
  const int np = P_ + 1;
  const auto& xi = op_.nodes(); // reference nodes in [-1, 1]
  std::vector<double> x(nDof());

  for (int e = 0; e < nElem_; ++e) {
    const double xe = e * h_; // left edge of element e

    // affine map from [-1, 1] to [xe, xe + h]
    for (int j = 0; j < np; ++j)
      x[e * np + j] = xe + 0.5 * h_ * (xi[j] + 1.0);
  }
  return x;
}

// -----------------------------------------------------------------------------
// makeState: pack nodal primitive fields (h, u, w, p) into the var-major state.
// -----------------------------------------------------------------------------
State SGN1D::makeState(const std::vector<double>& h,
                       const std::vector<double>& u,
                       const std::vector<double>& w,
                       const std::vector<double>& p) const {
  const int N = nDof();

  // Step 1: every input field must have one value per node
  if ((int)h.size() != N || (int)u.size() != N || (int)w.size() != N ||
      (int)p.size() != N)
    throw std::invalid_argument("makeState: size mismatch");

  // Step 2: allocate the full state (NVAR * N entries)
  State U(nTotal());

  // Step 3: convert node by node to conserved variables and store variable-major
  for (int i = 0; i < N; ++i) {
    double Un[NVAR];                           // conserved state at node i
    fromPrimitive(h[i], u[i], w[i], p[i], Un);
    for (int v = 0; v < NVAR; ++v) U[v * N + i] = Un[v];
  }
  return U;
}

// -----------------------------------------------------------------------------
// addRelaxationZone: register a wave generation or absorption zone on [x0, x1].
// Only the nodes inside the zone are stored, with their relaxation coefficient.
// -----------------------------------------------------------------------------
void SGN1D::addRelaxationZone(double x0, double x1, double strength,
                              ZoneTarget target) {
  // Step 1: validate the zone
  if (!(x1 > x0) || strength <= 0.0 || !target)
    throw std::invalid_argument("addRelaxationZone: bad zone");

  RelaxZone z;
  const auto x = xCoords();
  constexpr double pi = 3.14159265358979323846;

  // Step 2: collect the nodes strictly inside the zone
  for (int i = 0; i < nDof(); ++i) {
    if (x[i] > x0 && x[i] < x1) {
      // Step 3: smooth sin^2 profile, 0 at both edges and 1 at the centre, so
      // the zone itself does not reflect incoming waves
      const double s = (x[i] - x0) / (x1 - x0);   // position within the zone, 0..1
      const double sp = std::sin(pi * s);

      // Step 4: store node index, relaxation coefficient and coordinate
      z.node.push_back(i);
      z.sig.push_back(strength * sp * sp);
      z.x.push_back(x[i]);
    }
  }
  z.target = std::move(target);
  zones_.push_back(std::move(z));
}

// -----------------------------------------------------------------------------
// rhs: semi-discrete right-hand side dU/dt = R(U) (the core of the solver).
// -----------------------------------------------------------------------------
void SGN1D::rhs(const State& U, State& dUdt, double t) const {
  const int np = P_ + 1;   // nodes per element
  const int N = nDof();    // nodes per variable

  // Step 0: validate the input and zero the output
  if ((int)U.size() != nTotal())
    throw std::invalid_argument("SGN1D::rhs: size mismatch");
  dUdt.assign(nTotal(), 0.0);

  // FR operators of the reference element (see fr_operators.h)
  const auto& D = op_.D();          // differentiation matrix
  const auto& lL = op_.lL();        // interpolation row to the left face
  const auto& lR = op_.lR();        // interpolation row to the right face
  const auto& gLp = op_.gLprime();  // left correction-function derivative
  const auto& gRp = op_.gRprime();  // right correction-function derivative

  // --- PHASE 1) Face-extrapolated states ---------------------------------------
  // UfL[e*NVAR+v] / UfR[e*NVAR+v]: variable v of element e evaluated at its left
  // / right face (xi = -1 / +1), from the element's own polynomial.
  std::vector<double> UfL(nElem_ * NVAR), UfR(nElem_ * NVAR);
  for (int e = 0; e < nElem_; ++e) {
    for (int v = 0; v < NVAR; ++v) {
      double sL = 0.0, sR = 0.0;
      const double* Uv = &U[v * N + e * np]; // nodal values of variable v in element e

      // interpolate to the two faces
      for (int j = 0; j < np; ++j) {
        sL += lL[j] * Uv[j];
        sR += lR[j] * Uv[j];
      }
      UfL[e * NVAR + v] = sL;
      UfR[e * NVAR + v] = sR;
    }
  }

  // --- PHASE 2) Interface quantities -------------------------------------------
  // For interface I (x = I*h):
  //   Fstar[I*NVAR+v]  common (Rusanov) flux of variable v
  //   hStar[I]         common depth used by the non-conservative product
  std::vector<double> Fstar(nElem_ * NVAR);
  std::vector<double> hStar(nElem_);
  for (int I = 0; I < nElem_; ++I) {
    // 2a. the two elements sharing interface I (periodic wrap-around)
    const int eL = (I - 1 + nElem_) % nElem_;
    const double* UL = &UfR[eL * NVAR]; // right face of the left element
    const double* UR = &UfL[I * NVAR];  // left face of the right element

    // 2b. Rusanov flux with the single-valued face bathymetry
    rusanov(UL, UR, Hface_[I], par_, &Fstar[I * NVAR]);

    // 2c. common depth for the non-conservative product
    if (ncp_ == NcpInterface::Central) {
      hStar[I] = 0.5 * (UL[H] + UR[H]);             // production: central average
    } else {
      // test-only variant: take h from the upwind side of the mean velocity
      const double uBar = 0.5 * (UL[HU] / UL[H] + UR[HU] / UR[H]);
      hStar[I] = (uBar >= 0.0) ? UL[H] : UR[H];
    }
  }

  // --- PHASE 3) Per-element volume work ----------------------------------------
  // Fnod[j*NVAR+v]: physical flux of variable v at local node j of the current
  // element (reused scratch array).
  std::vector<double> Fnod(np * NVAR);
  for (int e = 0; e < nElem_; ++e) {
    const int IL = e;                   // interface at the left face of e
    const int IR = (e + 1) % nElem_;    // interface at the right face of e

    // 3a. physical flux at every node of the element
    for (int j = 0; j < np; ++j) {
      double Un[NVAR];
      for (int v = 0; v < NVAR; ++v) Un[v] = U[v * N + e * np + j];
      fluxX(Un, Hnod_[e * np + j], par_, &Fnod[j * NVAR]);
    }

    // 3b. the element's own (discontinuous) flux at its two faces
    double FdiscL[NVAR], FdiscR[NVAR];
    fluxX(&UfL[e * NVAR], Hface_[IL], par_, FdiscL);
    fluxX(&UfR[e * NVAR], Hface_[IR], par_, FdiscR);

    // the element's own depth at its two faces (for the depth-gradient correction)
    const double hFaceL = UfL[e * NVAR + H];
    const double hFaceR = UfR[e * NVAR + H];

    for (int i = 0; i < np; ++i) {
      const int g = e * np + i; // global node index

      // 3c. FR-corrected flux divergence dF/dx at node i:
      //     D*Fnod  +  (F* - F_disc) lifted by the correction functions, / J
      double dF[NVAR];
      for (int v = 0; v < NVAR; ++v) {
        double dfdxi = 0.0;
        // interior derivative in the reference coordinate
        for (int k = 0; k < np; ++k)
          dfdxi += D[(size_t)i * np + k] * Fnod[k * NVAR + v];

        // correction: jump between common and element face flux at both faces
        const double corr = (Fstar[IL * NVAR + v] - FdiscL[v]) * gLp[i] +
                            (Fstar[IR * NVAR + v] - FdiscR[v]) * gRp[i];
        dF[v] = (dfdxi + corr) / J_;
      }

      // 3d. FR-corrected depth gradient d_x h at node i, same construction with
      //     the common depth h* instead of a flux
      double dhdxi = 0.0;
      for (int k = 0; k < np; ++k)
        dhdxi += D[(size_t)i * np + k] * U[H * N + e * np + k];
      dhdxi += (hStar[IL] - hFaceL) * gLp[i] + (hStar[IR] - hFaceR) * gRp[i];
      const double hx = dhdxi / J_;

      // 3e. pointwise terms at node i: non-conservative product B and sources S
      double Un[NVAR], B[NVAR], S[NVAR];
      for (int v = 0; v < NVAR; ++v) Un[v] = U[v * N + g];
      ncpX(Un, hx, par_, B);
      source(Un, Hnod_[g], Hx_[g], par_, S);

      // 3f. assemble the right-hand side: dU/dt = -dF/dx - B d_x U + S
      for (int v = 0; v < NVAR; ++v)
        dUdt[v * N + g] = -dF[v] - B[v] + S[v];
    }
  }

  // --- PHASE 4) Relaxation zones -----------------------------------------------
  // Inside each zone add -sigma (U - U*): the solution is pulled towards the
  // target state, which either injects a prescribed wave (generation) or
  // drives the field to rest (absorption).
  for (const auto& z : zones_) {
    for (size_t m = 0; m < z.node.size(); ++m) {
      const int i = z.node[m]; // global node index
      double prim[4], Ustar[NVAR];

      // target state at this node and time
      z.target(z.x[m], t, prim);
      fromPrimitive(prim[0], prim[1], prim[2], prim[3], Ustar);

      // relaxation source
      for (int v = 0; v < NVAR; ++v)
        dUdt[v * N + i] -= z.sig[m] * (U[v * N + i] - Ustar[v]);
    }
  }
}

// -----------------------------------------------------------------------------
// samplePrimitive: evaluate the solution at an arbitrary point xq (wave gauge).
// -----------------------------------------------------------------------------
void SGN1D::samplePrimitive(const State& U, double xq, double prim[4]) const {
  const int np = P_ + 1;
  const int N = nDof();

  // Step 1: periodic wrap of xq into [0, L)
  double xw = std::fmod(xq, L_);
  if (xw < 0.0) xw += L_;

  // Step 2: element containing xw
  int e = (int)(xw / h_);
  if (e >= nElem_) e = nElem_ - 1;   // guard for xw == L within round-off

  // Step 3: reference coordinate of xw inside element e
  const double xi = 2.0 * (xw - e * h_) / h_ - 1.0;

  // Step 4: Lagrange basis values l_j(xi) (product form; lb holds up to P = 15)
  const auto& xn = op_.nodes();
  double lb[16];
  for (int j = 0; j < np; ++j) {
    double prod = 1.0;
    for (int k = 0; k < np; ++k)
      if (k != j) prod *= (xi - xn[k]) / (xn[j] - xn[k]);
    lb[j] = prod; // weight of node j at xi
  }

  // Step 5: interpolate every conserved variable
  double Uc[NVAR] = {0.0, 0.0, 0.0, 0.0};
  for (int v = 0; v < NVAR; ++v)
    for (int j = 0; j < np; ++j) Uc[v] += lb[j] * U[v * N + e * np + j];

  // Step 6: conserved -> primitive (h, u, w, p)
  prim[0] = Uc[H];
  prim[1] = Uc[HU] / Uc[H];
  prim[2] = Uc[HW] / Uc[H];
  prim[3] = Uc[HP] / Uc[H];
}

// -----------------------------------------------------------------------------
// maxWaveSpeed: largest |u| + Ce over all nodes.
// -----------------------------------------------------------------------------
double SGN1D::maxWaveSpeed(const State& U) const {
  const int N = nDof();
  double lam = 0.0;

  for (int i = 0; i < N; ++i) {
    double Un[NVAR];
    // conserved state at node i
    for (int v = 0; v < NVAR; ++v) Un[v] = U[v * N + i];
    // pointwise spectral radius from sgn_flux.h
    lam = std::fmax(lam, sgn::maxWaveSpeed(Un, par_));
  }
  return lam;
}

// -----------------------------------------------------------------------------
// cflTimeStep: stable explicit time step estimate.
// -----------------------------------------------------------------------------
double SGN1D::cflTimeStep(const State& U, double cfl) const {
  const double lam = maxWaveSpeed(U); // fastest characteristic speed
  // effective node spacing of a degree-P element (usual FR/DG scaling)
  const double dxEff = h_ / (2.0 * P_ + 1.0);
  // dt = CFL * dx_eff / lambda  (guard against a state at rest with lambda = 0)
  return cfl * dxEff / (lam > 0.0 ? lam : 1e-300);
}

// -----------------------------------------------------------------------------
// mass: integral of h over the domain (GLL quadrature, element by element).
// -----------------------------------------------------------------------------
double SGN1D::mass(const State& U) const {
  const int np = P_ + 1;
  const int N = nDof();
  const auto& qw = op_.weights(); // reference quadrature weights
  double m = 0.0;

  for (int e = 0; e < nElem_; ++e)
    for (int j = 0; j < np; ++j) m += qw[j] * U[H * N + e * np + j];

  return m * J_; // reference -> physical element length
}

// -----------------------------------------------------------------------------
// momentum: integral of hu over the domain.
// -----------------------------------------------------------------------------
double SGN1D::momentum(const State& U) const {
  const int np = P_ + 1;
  const int N = nDof();
  const auto& qw = op_.weights();
  double m = 0.0;

  for (int e = 0; e < nElem_; ++e)
    for (int j = 0; j < np; ++j) m += qw[j] * U[HU * N + e * np + j];

  return m * J_;
}

// -----------------------------------------------------------------------------
// energy: integral of the mechanical energy density (diagnostic).
// -----------------------------------------------------------------------------
double SGN1D::energy(const State& U) const {
  const int np = P_ + 1;
  const int N = nDof();
  const auto& qw = op_.weights();
  double Etot = 0.0;

  for (int e = 0; e < nElem_; ++e)
    for (int j = 0; j < np; ++j) {
      const int g = e * np + j;
      double Un[NVAR];
      for (int v = 0; v < NVAR; ++v) Un[v] = U[v * N + g];
      Etot += qw[j] * energyDensity(Un, Hnod_[g], par_);
    }

  return Etot * J_;
}

} // namespace sgn
