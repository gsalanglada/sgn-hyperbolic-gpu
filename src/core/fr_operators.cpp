// =============================================================================
// fr_operators.cpp -- construction of the 1D Flux Reconstruction operators
// =============================================================================
//
// Computes, once per polynomial degree P, the node set, differentiation matrix,
// face interpolation rows, correction-function derivatives and quadrature
// weights declared in fr_operators.h. Everything here runs on the host at setup
// time; the solvers copy the resulting small arrays (at most (P+1)^2 entries)
// and, on GPUs, upload them to the device.
//
// Direct port of the NSK-FR-CUDA Gate-1 operator (the reused numerical core).
// =============================================================================

#include "fr_operators.h"

#include <cmath>
#include <stdexcept>

namespace sgn {

namespace { constexpr double kPi = 3.14159265358979323846; }

// ---------------------------------------------------------------------------
// Legendre polynomials via the standard three-term recurrence.
//   (n+1) L_{n+1} = (2n+1) x L_n - n L_{n-1},   L_0 = 1, L_1 = x
// ---------------------------------------------------------------------------
double legendreP(int n, double x) {
  if (n == 0) return 1.0;
  if (n == 1) return x;
  double Lm1 = 1.0, L = x;  // L_{k-1} and L_k, starting at k = 1
  for (int k = 1; k < n; ++k) {
    double Lp1 = ((2.0 * k + 1.0) * x * L - k * Lm1) / (k + 1.0);  // L_{k+1}
    Lm1 = L;
    L = Lp1;
  }
  return L;
}

// Derivative of the Legendre polynomial L_n at x.
double legendrePderiv(int n, double x) {
  if (n == 0) return 0.0;
  // L_n'(x) = n / (x^2 - 1) * (x L_n - L_{n-1}), guarded at the endpoints.
  const double denom = x * x - 1.0;
  if (std::abs(denom) < 1e-14) {
    // At x = +-1 the formula is 0/0; use the closed form
    // L_n'(+-1) = (+-1)^{n-1} n (n+1) / 2
    const double s = (x > 0.0) ? 1.0 : ((n - 1) % 2 == 0 ? 1.0 : -1.0);
    return s * 0.5 * n * (n + 1.0);
  }
  return n * (x * legendreP(n, x) - legendreP(n - 1, x)) / denom;
}

// ---------------------------------------------------------------------------
// Node computation by Newton iteration (file-local helpers).
// ---------------------------------------------------------------------------
namespace {

// Gauss-Legendre-Lobatto nodes: -1, +1, and the P-1 roots of L_P'(x).
std::vector<double> gllNodes(int P) {
  std::vector<double> x(P + 1);
  if (P == 0) { x[0] = 0.0; return x; }
  x[0] = -1.0;  // left face
  x[P] = 1.0;   // right face
  for (int i = 1; i < P; ++i) {
    // Chebyshev-Gauss-Lobatto point as initial guess, then Newton on L_P'.
    double xi = -std::cos(kPi * i / static_cast<double>(P));
    for (int it = 0; it < 100; ++it) {
      const double f = legendrePderiv(P, xi);   // function whose root we want
      // its derivative, from the Legendre ODE:
      // L_P''(x) = (2 x L_P' - P(P+1) L_P) / (1 - x^2)
      const double fp =
          (2.0 * xi * legendrePderiv(P, xi) - P * (P + 1.0) * legendreP(P, xi)) /
          (1.0 - xi * xi);
      const double dx = f / fp;                 // Newton step
      xi -= dx;
      if (std::abs(dx) < 1e-15) break;          // converged to machine precision
    }
    x[i] = xi;
  }
  return x;
}

// Gauss-Legendre nodes: the P+1 roots of L_{P+1}(x) (no endpoints).
std::vector<double> gaussNodes(int P) {
  const int n = P + 1;
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) {
    double xi = -std::cos(kPi * (i + 0.75) / (n + 0.5)); // asymptotic guess
    for (int it = 0; it < 100; ++it) {
      const double f = legendreP(n, xi);
      const double fp = legendrePderiv(n, xi);
      const double dx = f / fp;                 // Newton step on L_{P+1}
      xi -= dx;
      if (std::abs(dx) < 1e-15) break;
    }
    x[i] = xi;
  }
  return x;
}

// Barycentric weights w_j = 1 / prod_{k!=j}(x_j - x_k). They give a numerically
// stable way to evaluate and differentiate the Lagrange interpolant.
std::vector<double> barycentricWeights(const std::vector<double>& x) {
  const int n = static_cast<int>(x.size());
  std::vector<double> w(n, 1.0);
  for (int j = 0; j < n; ++j) {
    double prod = 1.0;
    for (int k = 0; k < n; ++k)
      if (k != j) prod *= (x[j] - x[k]);
    w[j] = 1.0 / prod;
  }
  return w;
}

// Value of the j-th Lagrange basis polynomial at xq, barycentric form.
// If xq coincides with a node the result is the Kronecker delta, returned
// exactly (avoids 0/0).
double lagrangeAt(const std::vector<double>& x, const std::vector<double>& w,
                  int j, double xq) {
  const int n = static_cast<int>(x.size());
  for (int k = 0; k < n; ++k)
    if (std::abs(xq - x[k]) < 1e-14) return (k == j) ? 1.0 : 0.0;
  double num = w[j] / (xq - x[j]);
  double den = 0.0;
  for (int k = 0; k < n; ++k) den += w[k] / (xq - x[k]);
  return num / den;
}

} // namespace

// ---------------------------------------------------------------------------
// FROperators1D constructor: builds every operator for degree P.
// ---------------------------------------------------------------------------
FROperators1D::FROperators1D(int P, PointSet pointSet, CorrectionType correction)
    : P_(P) {
  if (P < 0) throw std::invalid_argument("FROperators1D: P must be >= 0");

  // Step 1: solution points and their barycentric weights
  x_ = (pointSet == PointSet::GaussLobatto) ? gllNodes(P) : gaussNodes(P);
  w_ = barycentricWeights(x_);

  const int n = P + 1;  // number of solution points

  // Step 2: differentiation matrix (Berrut & Trefethen barycentric formula):
  //   D[i][j] = (w_j / w_i) / (x_i - x_j),  i != j
  //   D[i][i] = -sum_{j != i} D[i][j]   (rows sum to zero: derivative of a
  //                                      constant is exactly 0 in floating point)
  D_.assign(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    double diag = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const double dij = (w_[j] / w_[i]) / (x_[i] - x_[j]);
      D_[static_cast<size_t>(i) * n + j] = dij;
      diag -= dij;
    }
    D_[static_cast<size_t>(i) * n + i] = diag;
  }

  // Step 3: face interpolation rows: f(-1)=sum lL_j f_j, f(+1)=sum lR_j f_j.
  lL_.assign(n, 0.0);
  lR_.assign(n, 0.0);
  for (int j = 0; j < n; ++j) {
    lL_[j] = lagrangeAt(x_, w_, j, -1.0);
    lR_[j] = lagrangeAt(x_, w_, j, +1.0);
  }

  // Step 4: correction-function derivatives at the solution points (Huynh g_DG):
  //   gL(x) = (-1)^P / 2 * (L_P - L_{P+1})   -> gL(-1)=1, gL(+1)=0
  //   gR(x) =        1 / 2 * (L_P + L_{P+1})  -> gR(+1)=1, gR(-1)=0
  gLp_.assign(n, 0.0);
  gRp_.assign(n, 0.0);
  if (correction != CorrectionType::DG)
    throw std::invalid_argument("FROperators1D: only DG correction implemented");
  const double signP = (P % 2 == 0) ? 1.0 : -1.0;  // (-1)^P
  for (int i = 0; i < n; ++i) {
    const double dLP  = legendrePderiv(P, x_[i]);
    const double dLP1 = legendrePderiv(P + 1, x_[i]);
    gLp_[i] = 0.5 * signP * (dLP - dLP1);
    gRp_[i] = 0.5 * (dLP + dLP1);
  }

  // Step 5: quadrature weights on [-1,1] (sum to 2).
  qw_.assign(n, 0.0);
  if (P == 0) {
    qw_[0] = 2.0;
  } else if (pointSet == PointSet::GaussLobatto) {
    // Gauss-Lobatto: w_i = 2 / (P(P+1) [L_P(x_i)]^2), exact for degree 2P-1
    for (int i = 0; i < n; ++i) {
      const double LP = legendreP(P, x_[i]);
      qw_[i] = 2.0 / (P * (P + 1.0) * LP * LP);
    }
  } else {
    // Gauss: w_i = 2 / ((1 - x_i^2) [L'_{P+1}(x_i)]^2), exact for degree 2P+1
    for (int i = 0; i < n; ++i) {
      const double dLP1 = legendrePderiv(P + 1, x_[i]);
      qw_[i] = 2.0 / ((1.0 - x_[i] * x_[i]) * dLP1 * dLP1);
    }
  }
}

// Discontinuous derivative (D f)_i of nodal values f on one element.
std::vector<double>
FROperators1D::differentiate(const std::vector<double>& f) const {
  const int n = P_ + 1;
  if (static_cast<int>(f.size()) != n)
    throw std::invalid_argument("differentiate: size mismatch");
  std::vector<double> df(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) s += D_[static_cast<size_t>(i) * n + j] * f[j];
    df[i] = s;
  }
  return df;
}

// FR derivative on one element: interior derivative plus the correction terms
// that enforce the common interface values (see fr_operators.h).
std::vector<double> FROperators1D::frDerivative(const std::vector<double>& f,
                                                double fL_common,
                                                double fR_common) const {
  const int n = P_ + 1;
  std::vector<double> df = differentiate(f);
  // Step 1: face values of the element's own (discontinuous) interpolant.
  double fLm = 0.0, fRp = 0.0;
  for (int j = 0; j < n; ++j) {
    fLm += lL_[j] * f[j];
    fRp += lR_[j] * f[j];
  }
  // Step 2: jumps between the common interface values and the interior values.
  const double jumpL = fL_common - fLm;
  const double jumpR = fR_common - fRp;
  // Step 3: lift the jumps into the interior with the correction functions.
  for (int i = 0; i < n; ++i) df[i] += jumpL * gLp_[i] + jumpR * gRp_[i];
  return df;
}

} // namespace sgn
