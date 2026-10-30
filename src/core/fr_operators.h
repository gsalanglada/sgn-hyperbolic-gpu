#ifndef SGN_FR_OPERATORS_H
#define SGN_FR_OPERATORS_H

// =============================================================================
// fr_operators.h -- 1D Flux Reconstruction operators on the reference element
// =============================================================================
//
// WHAT THIS FILE IS
//   Everything the spatial discretization needs to know about ONE element of
//   degree P, independently of physics and of the mesh. Each element is mapped
//   to the reference interval xi in [-1, 1], carrying P+1 solution points. A
//   field is stored by its values at those points (a "nodal" representation of
//   the degree-P interpolating polynomial). This class precomputes, once:
//
//     nodes()     the P+1 solution-point coordinates xi_j
//     D()         differentiation matrix: derivative of the interpolant at the
//                 nodes, from the nodal values
//     lL(), lR()  interpolation rows: value of the interpolant at xi = -1 / +1
//     gLprime(),
//     gRprime()   derivatives of the FR correction functions at the nodes
//     weights()   quadrature weights (integrals over the element)
//
// HOW THE FR DERIVATIVE WORKS (the core idea of flux reconstruction)
//   Inside an element the nodal polynomial f is discontinuous with its
//   neighbours. The numerical interface values f*_L, f*_R (e.g. a Riemann flux)
//   are made to hold at the faces by adding correction functions gL, gR that are
//   1 at their own face and 0 at the other:
//
//     (df/dxi)_i = sum_j D_ij f_j + (f*_L - f(-1)) gL'(xi_i) + (f*_R - f(+1)) gR'(xi_i)
//
//   Dividing by the Jacobian J = dx_element / 2 gives d/dx in physical space.
//   With Huynh's g_DG correction functions this is equivalent to nodal DG.
//
//   Multi-dimensional operators are NOT built here: 2D uses the tensor product
//   of these 1D operators, applied direction by direction.
//
//   Direct port of the Gate-1-validated FROperators1D from the NSK-FR-CUDA
//   engine (the reused numerical foundation).
// =============================================================================

#include <vector>
#include <cstddef>

namespace sgn {

// Solution-point family.
//   GaussLobatto  includes the endpoints -1 and +1, so element faces coincide
//                 with solution points: face values need no interpolation and
//                 the face bathymetry is single-valued. Used in production.
//   GaussLegendre interior points only; slightly more accurate. Tests only.
enum class PointSet { GaussLobatto, GaussLegendre };

// Correction-function family. Only DG (Huynh's g_DG, which recovers nodal
// discontinuous Galerkin) is implemented.
enum class CorrectionType { DG };

class FROperators1D {
public:
  // Build all operators for polynomial degree P (P+1 solution points).
  FROperators1D(int P,
                PointSet pointSet = PointSet::GaussLobatto,
                CorrectionType correction = CorrectionType::DG);

  int order() const { return P_; }       // polynomial degree P
  int nPoints() const { return P_ + 1; } // solution points per element (per direction)

  // Solution-point coordinates on [-1, 1], ascending.
  const std::vector<double>& nodes() const { return x_; }

  // Differentiation matrix, row-major (P+1)x(P+1): (D f)_i = sum_j D[i*n+j] f_j
  // is the derivative at node x_i of the interpolating polynomial through
  // (x_j, f_j). Exact for polynomials of degree <= P.
  const std::vector<double>& D() const { return D_; }

  // Derivatives of the left/right FR correction functions, evaluated at the
  // solution points. They lift the face jump (common value minus interior
  // value) into the element interior.
  const std::vector<double>& gLprime() const { return gLp_; }
  const std::vector<double>& gRprime() const { return gRp_; }

  // Interpolation of nodal values to the left (-1) and right (+1) faces:
  // f(-1) = sum_j lL[j] f_j,  f(+1) = sum_j lR[j] f_j.
  // With Gauss-Lobatto points these are unit vectors (first / last node).
  const std::vector<double>& lL() const { return lL_; }
  const std::vector<double>& lR() const { return lR_; }

  // Quadrature weights on [-1,1] for the node set (GLL or Gauss). Sum to 2.
  // Physical integrals over an element are sum_j qw_j f_j * J.
  const std::vector<double>& weights() const { return qw_; }

  // Convenience: discontinuous derivative D*f at the solution points
  // (no interface correction).
  std::vector<double> differentiate(const std::vector<double>& f) const;

  // Full FR derivative of a scalar field on one element, given the common
  // (numerical) interface values fL_common at face -1 and fR_common at face +1:
  //   dF/dxi = D f + (fL_common - f(-1)) gLprime + (fR_common - f(+1)) gRprime
  // With a continuous field and fL_common=f(-1), fR_common=f(+1) this reduces to
  // the exact derivative for polynomials of degree <= P. (The production solvers
  // inline this formula over all elements instead of calling it per element.)
  std::vector<double> frDerivative(const std::vector<double>& f,
                                   double fL_common,
                                   double fR_common) const;

private:
  int P_;                    // polynomial degree
  std::vector<double> x_;    // solution-point coordinates, size P+1
  std::vector<double> w_;    // barycentric weights of the nodes, size P+1
  std::vector<double> D_;    // differentiation matrix, row-major, size (P+1)^2
  std::vector<double> gLp_;  // gL'(x_i), size P+1
  std::vector<double> gRp_;  // gR'(x_i), size P+1
  std::vector<double> lL_;   // interpolation row to xi = -1, size P+1
  std::vector<double> lR_;   // interpolation row to xi = +1, size P+1
  std::vector<double> qw_;   // quadrature weights, size P+1
};

// --- Legendre helpers (exposed for tests) ----------------------------------
double legendreP(int n, double x);      // Legendre polynomial L_n(x)
double legendrePderiv(int n, double x); // its derivative L_n'(x)

} // namespace sgn

#endif // SGN_FR_OPERATORS_H
