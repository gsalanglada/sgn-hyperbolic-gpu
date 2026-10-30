#ifndef SGN_SGN2D_CUDA_MPI_H
#define SGN_SGN2D_CUDA_MPI_H

// =============================================================================
// sgn2d_cuda_mpi.h -- distributed multi-GPU 2D FR solver (PRODUCTION SOLVER)
// =============================================================================
//
// WHAT THIS FILE IS
//   The solver used for every scaling and application run of the paper
//   (apps/sgn_bench.cu, apps/shoal_focus_mpi.cu, apps/tsunami_margin_mpi.cu). It
//   merges the single-GPU kernels (sgn2d_cuda.*) with the CPU-MPI slab exchange
//   (src/parallel/sgn2d_mpi.*). One MPI rank == one GPU.
//   CUDA vocabulary: see the primer at the top of sgn1d_cuda.cu.
//   MPI vocabulary:  see the header of src/parallel/sgn1d_mpi.h.
//
// WHAT HAPPENS PER RIGHT-HAND-SIDE EVALUATION (see rhsDevice)
//   1. kernels compute nodal fluxes and the face states of every owned element
//   2. a kernel packs the two x-boundary face-lines into contiguous device
//      buffers
//   3. the host waits for the GPU (so the buffers are complete), then exchanges
//      them with MPI directly from DEVICE pointers ("CUDA-aware MPI": OpenMPI
//      moves GPU memory itself, with no copy through host memory; on MN5 run
//      with UCX_TLS=^cuda_ipc)
//   4. kernels read the received ghost lines in place and finish the RHS
//   Interior arithmetic mirrors the CPU-MPI driver element by element, which is
//   itself bitwise the serial SGN2D; the interface fluxes are evaluated per
//   element (both edges) exactly like the CPU-MPI code, so multi-GPU ==
//   single-GPU holds bitwise.
//
// ADAPTIVE BS3(2)
//   Device-resident. Per step each rank reduces its error partials per block on
//   the device, sums the blocks on the host, and ONE MPI_Allreduce pair (sum and
//   max) yields the global norm, so every rank takes the same decision.
//
// CONSTRUCTION
//   Mirrors a host SGN2DMpi (already tested): geometry, operators and
//   bathymetry arrays are uploaded from it; `comm` must be the same communicator
//   the host partition was built with.
// =============================================================================

#include "../parallel/sgn2d_mpi.h"

namespace sgn {

class SGN2DCudaMpi {
public:
  // Select this rank's GPU, allocate all device buffers, upload host setup.
  SGN2DCudaMpi(const SGN2DMpi& host, MPI_Comm comm);
  ~SGN2DCudaMpi();                                  // frees all device memory
  SGN2DCudaMpi(const SGN2DCudaMpi&) = delete;       // owns raw device pointers
  SGN2DCudaMpi& operator=(const SGN2DCudaMpi&) = delete;

  int nLocalDof() const { return Nl_; }                    // owned nodes per variable
  int nLocalTotal() const { return sgn2d::NVAR * Nl_; }    // owned state size

  void upload(const State& U);       // owned local state, host -> device
  void download(State& U) const;     // owned local state, device -> host
  void rhsOnce(State& dU);           // one distributed device RHS + download

  // Device-resident adaptive integration with a global error norm. Must be
  // called on every rank (it communicates).
  AdaptiveRKStats integrateAdaptive(double t0, double tEnd, double dt0,
                                    const AdaptiveRKOptions& opt = {});

private:
  // dU = R(U) on this rank's GPU, including the halo exchange
  void rhsDevice(const double* U, double* dU);

  // degree, nodes per direction, nodes per element; owned columns, rows;
  // owned elements (w*Ny), owned nodes per variable
  int P_ = 0, np_ = 0, npp_ = 0, w_ = 0, Ny_ = 0, neL_ = 0, Nl_ = 0;
  double Jx_ = 0.0, Jy_ = 0.0;     // Jacobians
  sgn2d::Params par_;              // physical constants
  MPI_Comm comm_;                  // communicator
  SlabPartition1D part_;           // this rank's column slab and ring neighbours

  // Operator tables + bathymetry (device).
  double *d_D = nullptr, *d_lL = nullptr, *d_lR = nullptr;   // 1D FR operators
  double *d_gLp = nullptr, *d_gRp = nullptr;
  double *d_Hnod = nullptr, *d_Hx = nullptr, *d_Hy = nullptr;   // nodal H and gradients
  double *d_HfXL = nullptr, *d_HfXR = nullptr, *d_HfYB = nullptr, *d_HfYT = nullptr; // face H, 4 faces

  // State + RK stages (each NVAR*Nl).
  double *d_U = nullptr, *d_k1 = nullptr, *d_k2 = nullptr, *d_k3 = nullptr,
         *d_k4 = nullptr, *d_tmp = nullptr, *d_uNew = nullptr;

  // RHS work buffers.
  double *d_Fnod = nullptr, *d_Gnod = nullptr;          // nodal x/y fluxes, NVAR*Nl
  double *d_xFaceL = nullptr, *d_xFaceR = nullptr;      // face states, neL*np*NVAR
  double *d_yFaceB = nullptr, *d_yFaceT = nullptr;
  double *d_FdXL = nullptr, *d_FdXR = nullptr, *d_GdYB = nullptr, *d_GdYT = nullptr; // element face fluxes
  double *d_FxL = nullptr, *d_FxR = nullptr, *d_GyB = nullptr, *d_GyT = nullptr;     // Rusanov fluxes, both edges
  double *d_hSxL = nullptr, *d_hSxR = nullptr, *d_hSyB = nullptr, *d_hSyT = nullptr; // central depths h*

  // Halo buffers (device): packed x-boundary face-lines, len Ny*np*NVAR.
  double *d_sendL = nullptr, *d_sendR = nullptr;     // own leftmost / rightmost column faces
  double *d_ghostL = nullptr, *d_ghostR = nullptr;   // received from left / right rank

  int nBlocksErr_ = 0;                               // blocks of a full-state launch
  double *d_psum = nullptr, *d_pmax = nullptr;       // per-block error partials
};

} // namespace sgn

#endif // SGN_SGN2D_CUDA_MPI_H
