#ifndef SGN_SGN1D_CUDA_H
#define SGN_SGN1D_CUDA_H

// =============================================================================
// sgn1d_cuda.h -- single-GPU port of the 1D FR solver
// =============================================================================
//
// WHAT THIS FILE IS
//   A GPU version of SGN1D (src/solver/sgn1d.*): the same right-hand side, split
//   into four CUDA kernels, and the same adaptive integrator, all running on one
//   device. It calls EXACTLY the same SGN_HD physics functions
//   (src/physics/sgn_flux.h), so the GPU cannot drift from the CPU oracle: any
//   difference is summation-order round-off only (tested).
//
//   This is the 1D stepping stone of the port. The production solver is the 2D
//   multi-GPU one (sgn2d_cuda_mpi.*). The general explanation of CUDA terms
//   (kernel, thread, block, launch, shared memory) is at the top of
//   sgn1d_cuda.cu.
//
// TIME INTEGRATION
//   The same embedded Bogacki-Shampine 3(2) + PI controller as src/solver/rk.cpp,
//   with the state and all stages resident on the device ("device-resident"):
//   per step only the per-block partials of the error norm are copied back to
//   the host.
//
// SETUP
//   Construction copies the (already validated) setup of a host SGN1D: FR
//   operator tables, bathymetry arrays, geometry. One SGN1DCuda == one device.
//   Naming: members prefixed d_ point to DEVICE (GPU) memory and must never be
//   dereferenced on the host.
// =============================================================================

#include "../solver/sgn1d.h"

namespace sgn {

class SGN1DCuda {
public:
  // Allocate device memory and upload the operators and bathymetry of `host`.
  explicit SGN1DCuda(const SGN1D& host);
  ~SGN1DCuda();                              // frees all device memory
  SGN1DCuda(const SGN1DCuda&) = delete;      // owns raw device pointers: not copyable
  SGN1DCuda& operator=(const SGN1DCuda&) = delete;

  int nDof() const { return N_; }            // nodes per variable
  int nTotal() const { return NVAR * N_; }   // full state size

  void upload(const State& U);        // host -> device state
  void download(State& U) const;      // device state -> host

  // Compute dU = RHS(U) once, entirely on device, and download it (used by the
  // CPU/GPU parity test).
  void rhsOnce(State& dU);

  // Device-resident adaptive integration of the uploaded state (same
  // controller/constants as sgn::integrateAdaptive). State stays on device;
  // call download() afterwards.
  AdaptiveRKStats integrateAdaptive(double t0, double tEnd, double dt0,
                                    const AdaptiveRKOptions& opt = {});

private:
  // Launch the four RHS kernels: dU = R(U), both device pointers.
  void rhsDevice(const double* U, double* dU);

  int P_ = 0, ne_ = 0, N_ = 0, np_ = 0;  // degree, elements, nodes/var, nodes/element
  double J_ = 0.0;                       // Jacobian dx/dxi
  Params par_;                           // physical constants (copied into kernels)

  // Operator tables and bathymetry (device).
  double* d_D = nullptr;      // differentiation matrix, np*np
  double* d_lL = nullptr;     // left-face interpolation row, np
  double* d_lR = nullptr;     // right-face interpolation row, np
  double* d_gLp = nullptr;    // left correction derivative, np
  double* d_gRp = nullptr;    // right correction derivative, np
  double* d_Hnod = nullptr;   // nodal still depth, N
  double* d_Hx = nullptr;     // nodal d_x H, N
  double* d_Hface = nullptr;  // still depth at each interface, ne

  // State and RK stage buffers (device), each NVAR*N.
  double* d_U = nullptr;      // current state
  double* d_k1 = nullptr;     // stage derivatives k1..k4
  double* d_k2 = nullptr;
  double* d_k3 = nullptr;
  double* d_k4 = nullptr;
  double* d_tmp = nullptr;    // stage evaluation state
  double* d_uNew = nullptr;   // third-order candidate

  // RHS work buffers (device).
  double* d_Fnod = nullptr;   // NVAR*N nodal flux
  double* d_UfL = nullptr;    // ne*NVAR face states (left face of each element)
  double* d_UfR = nullptr;    // ne*NVAR face states (right face)
  double* d_FdL = nullptr;    // ne*NVAR discontinuous face fluxes (left)
  double* d_FdR = nullptr;    // ne*NVAR discontinuous face fluxes (right)
  double* d_Fstar = nullptr;  // ne*NVAR interface (Rusanov) fluxes
  double* d_hStar = nullptr;  // ne central face depths

  // Error-norm reduction partials: one entry per thread block of kErrNorm.
  int nBlocksErr_ = 0;        // number of blocks for a full-state launch
  double* d_psum = nullptr;   // per-block sum of squared scaled errors
  double* d_pmax = nullptr;   // per-block max scaled error
};

} // namespace sgn

#endif // SGN_SGN1D_CUDA_H
