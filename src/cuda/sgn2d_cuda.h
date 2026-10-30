#ifndef SGN_SGN2D_CUDA_H
#define SGN_SGN2D_CUDA_H

// =============================================================================
// sgn2d_cuda.h -- single-GPU port of the 2D tensor-product FR solver
// =============================================================================
//
// WHAT THIS FILE IS
//   The GPU version of SGN2D (src/solver/sgn2d.*) on one device. It mirrors
//   SGN2D::rhs() kernel by kernel and calls exactly the same SGN_HD physics
//   (sgn_flux2d.h). Same device-resident adaptive BS3(2) + PI controller as the
//   1D port; only the per-block error-norm partials cross the PCIe bus per step.
//   For the CUDA vocabulary (kernel, thread, block, launch) see the primer at
//   the top of sgn1d_cuda.cu. The distributed production version of the same
//   kernels is sgn2d_cuda_mpi.*.
//
// INTERFACE OWNERSHIP
//   The CPU solver evaluates every interface flux twice (once from each
//   element). Here each interface flux is computed ONCE and read by both
//   neighbours: element e "owns" the interfaces on its LEFT edge (x) and its
//   BOTTOM edge (y). The right / top interface of e is therefore stored at its
//   right / top neighbour. The arithmetic is identical to the CPU's per-element
//   evaluation, since both sides call rusanov with the same arguments.
// =============================================================================

#include "../solver/sgn2d.h"

namespace sgn {

class SGN2DCuda {
public:
  // Allocate device memory and upload operators, bathymetry and connectivity.
  explicit SGN2DCuda(const SGN2D& host);
  ~SGN2DCuda();                               // frees all device memory
  SGN2DCuda(const SGN2DCuda&) = delete;       // owns raw device pointers
  SGN2DCuda& operator=(const SGN2DCuda&) = delete;

  int nDof() const { return Nd_; }                   // nodes per variable
  int nTotal() const { return sgn2d::NVAR * Nd_; }   // full state size

  void upload(const State& U);      // host -> device state
  void download(State& U) const;    // device -> host state
  void rhsOnce(State& dU);   // one device RHS of the uploaded state + download

  // device-resident adaptive integration (same controller as rk.cpp)
  AdaptiveRKStats integrateAdaptive(double t0, double tEnd, double dt0,
                                    const AdaptiveRKOptions& opt = {});

private:
  // launch the five RHS kernels: dU = R(U), both device pointers
  void rhsDevice(const double* U, double* dU);

  int P_ = 0, np_ = 0, npp_ = 0, ne_ = 0, Nd_ = 0;  // degree, nodes/dir, nodes/elem, elements, nodes/var
  double Jx_ = 0.0, Jy_ = 0.0;                      // Jacobians
  sgn2d::Params par_;                               // physical constants

  // Operator tables, bathymetry, connectivity (device).
  double *d_D = nullptr, *d_lL = nullptr, *d_lR = nullptr;   // 1D FR operators
  double *d_gLp = nullptr, *d_gRp = nullptr;
  double *d_Hnod = nullptr, *d_Hx = nullptr, *d_Hy = nullptr;   // nodal H, gradients
  double *d_HfaceX = nullptr, *d_HfaceY = nullptr;             // left / bottom edge H, ne*np
  int *d_nbrL = nullptr, *d_nbrR = nullptr, *d_nbrB = nullptr, *d_nbrT = nullptr; // neighbour ids, ne

  // State + RK stages (device), each NVAR*Nd.
  double *d_U = nullptr, *d_k1 = nullptr, *d_k2 = nullptr, *d_k3 = nullptr,
         *d_k4 = nullptr, *d_tmp = nullptr, *d_uNew = nullptr;

  // RHS work buffers.
  double *d_Fnod = nullptr, *d_Gnod = nullptr;          // NVAR*Nd each: nodal x/y fluxes
  double *d_xFaceL = nullptr, *d_xFaceR = nullptr;      // ne*np*NVAR: x-face states
  double *d_yFaceB = nullptr, *d_yFaceT = nullptr;      //             y-face states
  double *d_FdXL = nullptr, *d_FdXR = nullptr;          // element's own (disc) face fluxes
  double *d_GdYB = nullptr, *d_GdYT = nullptr;
  double *d_FstarX = nullptr, *d_FstarY = nullptr;      // interface fluxes (left / bottom edge of e)
  double *d_hStarX = nullptr, *d_hStarY = nullptr;      // central face depths, ne*np

  int nBlocksErr_ = 0;                              // blocks of a full-state launch
  double *d_psum = nullptr, *d_pmax = nullptr;      // per-block error partials
};

} // namespace sgn

#endif // SGN_SGN2D_CUDA_H
