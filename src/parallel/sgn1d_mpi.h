#ifndef SGN_SGN1D_MPI_H
#define SGN_SGN1D_MPI_H

// =============================================================================
// sgn1d_mpi.h -- distributed 1D FR solver (MPI, CPU)
// =============================================================================
//
// WHAT THIS FILE IS
//   The multi-rank version of SGN1D: the same discretization, split across MPI
//   processes with the slab partition of slab1d.h. It validated the parallel
//   algorithm on the CPU oracle before the GPU port; the 2D multi-GPU solver
//   (src/cuda/sgn2d_cuda_mpi.*) uses the same communication pattern.
//
// MPI IN ONE PARAGRAPH
//   The program is launched as R identical processes ("ranks", e.g. srun -n R).
//   Each rank holds ONLY the elements it owns and computes on them. Data a rank
//   does not own but needs -- here the face state of the element just across its
//   slab boundary, the "halo" or "ghost" data -- is received from the neighbour
//   rank with a point-to-point message. Global quantities (the error norm of the
//   adaptive controller, the max wave speed, the total mass) are combined across
//   ranks with a collective MPI_Allreduce.
//
// COMMUNICATION PER STEP
//   - every right-hand-side evaluation: two MPI_Sendrecv calls, each moving ONE
//     face state (NVAR doubles) to one ring neighbour and receiving one from the
//     other;
//   - every adaptive step: two MPI_Allreduce calls on scalars (sum of squared
//     scaled errors, and their max).
//
// REPRODUCIBILITY
//   The per-owned-element arithmetic is byte-identical to the serial SGN1D, so
//   MPI(n ranks) == serial is BITWISE on owned elements (tested). The sum
//   Allreduce is the only order-dependent operation.
// =============================================================================

#include "../core/fr_operators.h"
#include "../physics/sgn_flux.h"
#include "../solver/rk.h"
#include "slab1d.h"

#include <functional>
#include <vector>
#include <mpi.h>

namespace sgn {

class SGN1DMpi {
public:
  // Global problem (P, nElemGlobal, L, par, optional bathymetry H(x)) split
  // across the ranks of `comm`. Every rank constructs the object with the same
  // global arguments and keeps only its own slab. GLL points required when
  // bathymetry is present.
  SGN1DMpi(int P, int nElemGlobal, double domainLength, const Params& par,
           MPI_Comm comm, std::function<double(double)> stillDepth = {},
           PointSet pointSet = PointSet::GaussLobatto);

  int order() const { return P_; }
  int nOwnedElem() const { return part_.w; }                 // owned elements
  int nLocalDof() const { return part_.w * (P_ + 1); }     // owned nodes / var
  int nLocalTotal() const { return NVAR * nLocalDof(); }   // owned state size
  const SlabPartition1D& partition() const { return part_; }
  const Params& params() const { return par_; }

  // Physical x-coordinate of every OWNED node.
  std::vector<double> xCoordsLocal() const;

  // Assemble the owned var-major state from primitive nodal fields (owned).
  State makeStateLocal(const std::vector<double>& h,
                       const std::vector<double>& u,
                       const std::vector<double>& w,
                       const std::vector<double>& p) const;

  // Distributed RHS of the owned state (does the halo exchange internally, so
  // EVERY rank must call it the same number of times).
  void rhs(const State& U, State& dUdt) const;

  // One fixed-step RK4 step of the owned state (no reduction -> stays bitwise
  // to a serial RK4 on owned elements). For the reproducibility test.
  void rk4Step(State& U, double dt) const;

  // Adaptive BS3(2) with a GLOBAL error norm (MPI_Allreduce). Same controller
  // and constants as sgn::integrateAdaptive, so all ranks take identical
  // accept/reject decisions and the same dt.
  AdaptiveRKStats integrateAdaptive(State& U, double t0, double tEnd, double dt0,
                                    const AdaptiveRKOptions& opt = {}) const;

  // Global max wave speed / CFL step (Allreduce max over ranks).
  double maxWaveSpeed(const State& U) const;
  double cflTimeStep(const State& U, double cfl) const;

  // Global diagnostics (Allreduce sum over ranks).
  double mass(const State& U) const;
  double momentum(const State& U) const;

private:
  // Exchange the two boundary face-state vectors with ring neighbours. Fills
  // ghostL (left neighbour's right face) and ghostR (right neighbour's left
  // face), each NVAR. size==1 -> self-wrap, no MPI.
  void exchangeFaces(const double* UfL, const double* UfR,
                     double* ghostL, double* ghostR) const;

  int P_;          // polynomial degree
  int neGlobal_;   // global element count
  double L_;       // domain length [m]
  double h_;       // element width
  double J_;       // dx/dxi
  Params par_;     // physical constants
  MPI_Comm comm_;  // communicator of the participating ranks
  SlabPartition1D part_;   // this rank's slab and neighbours
  FROperators1D op_;       // 1D FR operators

  std::vector<double> Hnod_;    // owned nodal H            (nLocalDof)
  std::vector<double> Hx_;      // owned nodal d_x H        (nLocalDof)
  std::vector<double> HfaceL_;  // still depth at each owned element's left face
  std::vector<double> HfaceR_;  // ... right face           (w each)
};

} // namespace sgn

#endif // SGN_SGN1D_MPI_H
