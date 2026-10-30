#ifndef SGN_SGN1D_H
#define SGN_SGN1D_H

// =============================================================================
// sgn1d.h -- 1D Flux Reconstruction solver (CPU reference implementation)
// =============================================================================
//
// WHAT THIS FILE IS
//   The one-dimensional, periodic, multi-element FR discretization of the
//   hyperbolic SGN system with optional smooth bathymetry. It turns the PDE into
//   the semi-discrete system dU/dt = R(U) (method rhs()), which the integrator of
//   rk.h then advances in time. It is the CPU reference ("oracle"): the CUDA port
//   (src/cuda/sgn1d_cuda.*) and the MPI port (src/parallel/sgn1d_mpi.*) mirror it
//   operation by operation and are tested bitwise against it.
//
//     d_t U + d_x F(U;H) + B(U) d_x U = S(U;H,Hx),   U = (h, hu, hw, hp)
//
// DISCRETIZATION (paper Section 3 and Appendix A)
//   - Conservative flux (sgn_flux.h, pre-balanced): Rusanov at the interfaces
//     with a single-valued face bathymetry Hface, FR-corrected divergence. A lake
//     at rest over arbitrary smooth H gives an exactly zero right-hand side.
//   - Non-conservative term -c^2 u d_x h: FR-corrected derivative of h using the
//     CENTRAL common value h* = (hL + hR)/2 at interfaces, then pointwise.
//   - Bathymetry sources: algebraic, with the nodal gradient Hx computed once at
//     setup by element-local FR differentiation of the nodal H.
//   - Gauss-Lobatto solution points are REQUIRED with bathymetry: element faces
//     are nodes, so the face bathymetry seen from both sides is identical.
//
// DATA LAYOUT
//   nElem elements of width h_ on [0, L), P+1 nodes per element.
//   Global node index  g = e*(P+1) + j   (element e, local node j).
//   State U is variable-major ("structure of arrays"):
//     U[v * nDof() + g]  = variable v (H, HU, HW, HP) at global node g.
//   Interface I lies at x = I*h_, the LEFT face of element I; its left neighbour
//   is element I-1 (periodic wrap-around), its right neighbour element I.
// =============================================================================

#include "../core/fr_operators.h"
#include "../physics/sgn_flux.h"
#include "rk.h"

#include <functional>
#include <vector>

namespace sgn {

class SGN1D {
public:
  // P            polynomial degree (P+1 nodes per element)
  // nElem        number of elements (>= 2, periodic)
  // domainLength L [m]
  // par          physical constants (g, c, gamma, Manning)
  // stillDepth   H(x) > 0, smooth (bathymetry). Empty function -> H == 0
  //              everywhere, which reduces every H term to zero and recovers the
  //              flat-bottom equations exactly (legacy behaviour; only
  //              eta = h - H shifts).
  // pointSet     solution-point family (GLL required with bathymetry)
  SGN1D(int P, int nElem, double domainLength, const Params& par,
        std::function<double(double)> stillDepth = {},
        PointSet pointSet = PointSet::GaussLobatto);

  int order() const { return P_; }                       // polynomial degree P
  int nElem() const { return nElem_; }                   // number of elements
  int nDof() const { return nElem_ * (P_ + 1); }         // nodes per variable
  int nTotal() const { return NVAR * nDof(); }           // full state size
  const Params& params() const { return par_; }

  // Physical x-coordinate of every node (gnode = e*(P+1)+j).
  std::vector<double> xCoords() const;

  // Assemble the var-major state from primitive nodal fields h, u, w, p
  // (each of size nDof()).
  State makeState(const std::vector<double>& h, const std::vector<double>& u,
                  const std::vector<double>& w,
                  const std::vector<double>& p) const;

  // Interface state used for the non-conservative product -c^2 u d_x h.
  //   Central: h* = (h- + h+)/2, the path-consistent choice for the straight-line
  //            path. PRODUCTION DEFAULT.
  //   Upwind:  h* taken from the upwind side according to the sign of the mean
  //            interface velocity.
  //   The Upwind option exists only to test the hypothesis that the central
  //   state causes the order loss at even P. The measurements (apps/ncp_tradeoff.cpp,
  //   paper Appendix B) show it does NOT: both variants give the same errors to
  //   four significant figures and both preserve the lake at rest exactly (with
  //   GLL faces, h- = h+ = H at rest, so every choice of h* coincides).
  enum class NcpInterface { Central, Upwind };
  void setNcpInterface(NcpInterface m) { ncp_ = m; }
  NcpInterface ncpInterface() const { return ncp_; }

  // Right-hand side R(U) of the semi-discrete system, written into dUdt.
  // Periodic in x. `t` only matters when relaxation zones with time-dependent
  // targets are present.
  void rhs(const State& U, State& dUdt, double t = 0.0) const;

  // Wrap rhs() as an RhsFn callback for the RK integrators of rk.h.
  RhsFn rhsFn() const {
    return [this](double t, const State& U, State& dU) { rhs(U, dU, t); };
  }

  // --- Wave generation / absorption via relaxation zones --------------------
  // Adds the source  -sigma(x) (U - U*(x,t))  inside [x0,x1], where
  // sigma(x) = strength * sin^2(pi (x-x0)/(x1-x0)) is a smooth bump (0 at both
  // zone edges). target fills prim = (h,u,w,p) at (x,t): a prescribed incident
  // wave (generation zone) or the rest state (absorption zone). Combined with the
  // periodic mesh, a generation zone plus a downstream absorption zone emulate
  // an open flume without wall boundary conditions. CPU driver only.
  using ZoneTarget = std::function<void(double x, double t, double prim[4])>;
  void addRelaxationZone(double x0, double x1, double strength,
                         ZoneTarget target);

  // Max |eigenvalue| over all nodes: max(|u| + Ce). For the CFL time step.
  double maxWaveSpeed(const State& U) const;

  // CFL-based time-step guess: dt = cfl * dxEff / maxWaveSpeed, with the
  // effective node spacing dxEff = elemWidth / (2P+1) (usual FR/DG scaling).
  double cflTimeStep(const State& U, double cfl) const;

  // Diagnostics (quadrature over the whole domain).
  double mass(const State& U) const;      // integral of h
  double momentum(const State& U) const;  // integral of hu
  double energy(const State& U) const;    // integral of the energy density (nodal H)

  // Point sampling (wave gauges): evaluate the element's FR polynomial at an
  // arbitrary x (periodic wrap) and return primitives prim = (h,u,w,p). Exact
  // for the element's polynomial (tested).
  void samplePrimitive(const State& U, double x, double prim[4]) const;

  // --- Accessors for the CUDA and MPI ports (they mirror, not duplicate, the setup)
  const FROperators1D& ops() const { return op_; }
  double jacobian() const { return J_; }
  double elemWidth() const { return h_; }
  const std::vector<double>& stillDepthNodal() const { return Hnod_; }
  const std::vector<double>& stillDepthGrad() const { return Hx_; }
  const std::vector<double>& stillDepthFace() const { return Hface_; }

private:
  int P_;      // polynomial degree
  int nElem_;  // number of elements
  double L_;   // domain length [m]
  double h_;   // element width [m]
  double J_;   // Jacobian dx/dxi = h/2
  Params par_; // physical constants
  FROperators1D op_;                           // 1D FR operators of degree P
  NcpInterface ncp_ = NcpInterface::Central;   // production default
  std::vector<double> Hnod_;   // nodal still depth H(x_i)      (nDof)
  std::vector<double> Hx_;     // nodal d_x H                    (nDof)
  std::vector<double> Hface_;  // still depth at interface I     (nElem)

  struct RelaxZone {          // sparse: only the nodes inside the zone
    std::vector<int> node;    // global node index
    std::vector<double> sig;  // strength * sin^2 profile at that node
    std::vector<double> x;    // node coordinate (for the target)
    ZoneTarget target;        // prescribed state U* inside the zone
  };
  std::vector<RelaxZone> zones_;
};

} // namespace sgn

#endif // SGN_SGN1D_H
