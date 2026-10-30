#ifndef SGN_SGN2D_MPI_H
#define SGN_SGN2D_MPI_H

// =============================================================================
// sgn2d_mpi.h -- distributed 2D FR solver (MPI, CPU)
// =============================================================================
//
// WHAT THIS FILE IS
//   The multi-rank version of SGN2D, and the CPU reference of the multi-GPU
//   solver (src/cuda/sgn2d_cuda_mpi.*), which reuses its setup through the
//   accessors below. See sgn1d_mpi.h for a one-paragraph introduction to MPI.
//
// DECOMPOSITION
//   1D slab decomposition in x over element COLUMNS (reuses the validated
//   SlabPartition1D over Nx): rank r owns global columns [a, b), w = b - a of
//   them, each spanning all Ny rows. The y direction stays local and periodic
//   on every rank, so only x faces cross rank boundaries.
//
// COMMUNICATION
//   Per right-hand-side evaluation each rank exchanges ONE x-boundary face-line
//   per side: the face states of its leftmost / rightmost column at every row
//   and transverse node, Ny*(P+1)*NVAR doubles. Per adaptive step: one pair of
//   scalar MPI_Allreduce calls (sum and max of the error), as in 1D.
//
// LOCAL INDEXING
//   Local element id: e = lx + w*ley  (lx in [0,w) local column, ley in [0,Ny)
//   row), i.e. a local CartesianMesh2D(w, Ny) with the same node layout as SGN2D:
//   local node (a, b) of element e is at e*npp + a + np*b.
//   Global column of local column lx: part_.a + lx.
//
// REPRODUCIBILITY
//   The per-owned-element arithmetic is byte-identical to the serial SGN2D, so
//   MPI(n) == serial is BITWISE on owned elements (tested).
// =============================================================================

#include "../core/fr_operators.h"
#include "../physics/sgn_flux2d.h"
#include "../solver/rk.h"
#include "slab1d.h"

#include <functional>
#include <vector>
#include <mpi.h>

namespace sgn {

class SGN2DMpi {
public:
  // Same global arguments on every rank; each rank keeps only its slab.
  //   NxGlobal  global element columns (split across ranks), Ny rows (local)
  SGN2DMpi(int P, int NxGlobal, int Ny, double Lx, double Ly,
           const sgn2d::Params& par, MPI_Comm comm,
           std::function<double(double, double)> stillDepth = {},
           PointSet pointSet = PointSet::GaussLobatto);

  int order() const { return P_; }
  int nOwnedElem() const { return part_.w * Ny_; }          // owned elements
  int nLocalDof() const { return nOwnedElem() * npp_; }     // owned nodes / var
  int nLocalTotal() const { return sgn2d::NVAR * nLocalDof(); } // owned state size
  const SlabPartition1D& partition() const { return part_; }

  // global (x, y) of every owned node
  void nodeCoordsLocal(std::vector<double>& x, std::vector<double>& y) const;

  // owned primitive fields -> owned var-major state
  State makeStateLocal(const std::vector<double>& h,
                       const std::vector<double>& u,
                       const std::vector<double>& v,
                       const std::vector<double>& w,
                       const std::vector<double>& p) const;

  // distributed right-hand side (halo exchange inside; call on every rank)
  void rhs(const State& U, State& dUdt) const;
  // fixed-step RK4 on the owned state (bitwise reproducibility test)
  void rk4Step(State& U, double dt) const;
  // adaptive BS3(2) with a global error norm, identical controller to rk.cpp
  AdaptiveRKStats integrateAdaptive(State& U, double t0, double tEnd,
                                    double dt0,
                                    const AdaptiveRKOptions& opt = {}) const;

  double maxWaveSpeed(const State& U) const;   // global (Allreduce max)
  double cflTimeStep(const State& U, double cfl) const;
  double mass(const State& U) const;           // global (Allreduce sum)

  // --- Accessors for the CUDA-aware multi-GPU port (it uploads these arrays) ---
  const FROperators1D& ops() const { return op_; }
  const sgn2d::Params& params() const { return par_; }
  int Ny() const { return Ny_; }
  double Jx() const { return Jx_; }
  double Jy() const { return Jy_; }
  const std::vector<double>& stillDepthNodal() const { return Hnod_; }
  const std::vector<double>& stillDepthGradX() const { return Hx_; }
  const std::vector<double>& stillDepthGradY() const { return Hy_; }
  const std::vector<double>& faceXL() const { return HfaceXL_; }
  const std::vector<double>& faceXR() const { return HfaceXR_; }
  const std::vector<double>& faceYB() const { return HfaceYB_; }
  const std::vector<double>& faceYT() const { return HfaceYT_; }

private:
  int P_, np_, npp_;                    // degree, nodes per direction, nodes per element
  int NxG_, Ny_;                        // global columns, rows
  double Lx_, Ly_, hx_, hy_, Jx_, Jy_;  // domain, element size, Jacobians
  sgn2d::Params par_;                   // physical constants
  MPI_Comm comm_;                       // communicator
  SlabPartition1D part_;   // over element columns in x
  FROperators1D op_;       // 1D FR operators

  std::vector<double> Hnod_, Hx_, Hy_;              // owned nodes: H, d_x H, d_y H
  // Face still depths of every owned element, one value per transverse node.
  // Unlike the serial solver, all four faces are stored per element, so no
  // neighbour lookup is needed across rank boundaries.
  std::vector<double> HfaceXL_, HfaceXR_;           // per elem * np (transverse b)
  std::vector<double> HfaceYB_, HfaceYT_;           // per elem * np (transverse a)
};

} // namespace sgn

#endif // SGN_SGN2D_MPI_H
