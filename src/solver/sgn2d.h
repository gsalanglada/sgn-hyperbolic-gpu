#ifndef SGN_SGN2D_H
#define SGN_SGN2D_H

// =============================================================================
// sgn2d.h -- 2D tensor-product Flux Reconstruction solver (CPU reference)
// =============================================================================
//
// WHAT THIS FILE IS
//   The two-dimensional, periodic, tensor-product FR discretization of the
//   hyperbolic SGN system with optional bathymetry H(x,y). It is the CPU oracle
//   for the 2D GPU solvers (src/cuda/sgn2d_cuda.*, src/cuda/sgn2d_cuda_mpi.*) and
//   the 2D MPI CPU solver (src/parallel/sgn2d_mpi.*), which reproduce its
//   right-hand side bitwise per element.
//
//     d_t U + d_x F + d_y G + Bx d_x U + By d_y U = S,   U = (h, hu, hv, hw, hp)
//
// TENSOR-PRODUCT CONSTRUCTION
//   Each element carries (P+1) x (P+1) nodes on a tensor grid of the 1D
//   solution points. The x- and y-operators are the SAME validated 1D
//   FROperators1D applied along x-lines (fixed b) and y-lines (fixed a).
//
// DATA LAYOUT
//   np  = P+1 nodes per direction, npp = np*np nodes per element.
//   Node (a, b) of element e: a = index along x, b = index along y.
//   Local node index    loc   = a + np*b
//   Global node index   gnode = e*npp + loc,  e = ex + Nx*ey (see mesh2d.h)
//   State U is variable-major: U[v * nDof() + gnode].
//
// WELL-BALANCING
//   Pre-balanced diagonal momentum flux g(h²-H²)/2 + hp with the bathymetry
//   gradient as a source (same construction as 1D) -> lake at rest exact.
//   GLL points required with bathymetry (faces are nodes -> single-valued Hface).
//
// DIFFERENCES FROM THE 1D SOLVER
//   Always uses the central h* for the non-conservative product (no switch), has
//   no relaxation zones, and its rhs() ignores time.
// =============================================================================

#include "../core/fr_operators.h"
#include "../core/mesh2d.h"
#include "../physics/sgn_flux2d.h"
#include "rk.h"

#include <functional>
#include <vector>

namespace sgn {

class SGN2D {
public:
  // P          polynomial degree
  // Nx, Ny     elements in x and y (each >= 2, periodic)
  // Lx, Ly     domain size [m]
  // par        physical constants
  // stillDepth H(x,y) > 0 (bathymetry). Empty -> H == 0 (flat, eta = h).
  // pointSet   solution-point family (GLL required with bathymetry)
  SGN2D(int P, int Nx, int Ny, double Lx, double Ly, const sgn2d::Params& par,
        std::function<double(double, double)> stillDepth = {},
        PointSet pointSet = PointSet::GaussLobatto);

  int order() const { return P_; }                              // degree P
  int nElem() const { return mesh_.nElem(); }                   // Nx*Ny
  int nodesPerElem() const { return (P_ + 1) * (P_ + 1); }      // npp
  int nDof() const { return nElem() * nodesPerElem(); }         // nodes per variable
  int nTotal() const { return sgn2d::NVAR * nDof(); }           // full state size
  const sgn2d::Params& params() const { return par_; }
  const CartesianMesh2D& mesh() const { return mesh_; }

  // (x,y) of every node; gnode = e*np*np + (a + np*b).
  void nodeCoords(std::vector<double>& x, std::vector<double>& y) const;

  // Pack nodal primitive fields (h, u, v, w, p) into the var-major state.
  State makeState(const std::vector<double>& h, const std::vector<double>& u,
                  const std::vector<double>& v, const std::vector<double>& w,
                  const std::vector<double>& p) const;

  // Right-hand side R(U) of the semi-discrete system.
  void rhs(const State& U, State& dUdt) const;
  // rhs() wrapped as an RhsFn for rk.h (the time argument is ignored).
  RhsFn rhsFn() const {
    return [this](double, const State& U, State& dU) { rhs(U, dU); };
  }

  // Largest characteristic speed over all nodes and both directions.
  double maxWaveSpeed(const State& U) const;
  // dt = cfl * min(hx,hy)/(2P+1) / maxWaveSpeed.
  double cflTimeStep(const State& U, double cfl) const;

  double mass(const State& U) const;   // integral of h
  double energy(const State& U) const; // integral of the energy density

  // --- Accessors for the CUDA / MPI ports (they mirror, not duplicate, the setup)
  const FROperators1D& ops() const { return op_; }
  const std::vector<double>& stillDepthNodal() const { return Hnod_; }
  const std::vector<double>& stillDepthGradX() const { return Hx_; }
  const std::vector<double>& stillDepthGradY() const { return Hy_; }
  const std::vector<double>& stillDepthFaceX() const { return HfaceX_; }
  const std::vector<double>& stillDepthFaceY() const { return HfaceY_; }

private:
  int P_, np_, npp_;        // degree, nodes per direction, nodes per element
  sgn2d::Params par_;       // physical constants
  CartesianMesh2D mesh_;    // element geometry and neighbours
  FROperators1D op_;        // 1D operators, applied per direction

  std::vector<double> Hnod_;    // nodal H(x,y)                 (nDof)
  std::vector<double> Hx_, Hy_; // nodal gradients d_x H, d_y H (nDof)
  // Face still-depths (single-valued along each face).
  //   HfaceX[e*np + b] = H at the LEFT edge of element e, transverse node b
  //   HfaceY[e*np + a] = H at the BOTTOM edge of element e, node a
  // (the right / top edge of e is the left / bottom edge of its neighbour)
  std::vector<double> HfaceX_, HfaceY_;
};

} // namespace sgn

#endif // SGN_SGN2D_H
