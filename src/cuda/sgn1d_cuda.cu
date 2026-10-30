// =============================================================================
// sgn1d_cuda.cu -- single-GPU kernels and integrator for the 1D FR solver
// =============================================================================
//
// CUDA PRIMER (terms used in every .cu file of this project)
//
//   kernel      A function marked __global__. It is not called like a C++
//               function: the host LAUNCHES it on the GPU, where it runs as many
//               parallel threads, each executing the same code on different
//               data.
//
//   launch      kName<<<nBlocks, threadsPerBlock>>>(args...)
//               starts nBlocks * threadsPerBlock threads. Here threadsPerBlock
//               is always kThreads = 128, and nBlocks(n) = ceil(n / 128), so at
//               least n threads run; the last block may contain a few extra
//               threads beyond n.
//
//   thread index
//               Inside a kernel each thread computes its own global index
//                 g = blockIdx.x * blockDim.x + threadIdx.x
//               (block number * threads per block + position inside the block),
//               and returns immediately if g is past the end (the padding
//               threads of the last block). The meaning of g is chosen per
//               kernel: here one thread per NODE, per ELEMENT, per INTERFACE or
//               per ARRAY ENTRY -- see the comment above each kernel.
//
//   no races    Threads run in no guaranteed order and do not talk to each
//               other. Every kernel below is written so that each thread READS
//               shared inputs freely but WRITES only the output slots belonging
//               to its own index, so the result does not depend on scheduling.
//
//   asynchrony  A launch returns to the host immediately; the GPU works in the
//               background. cudaMemcpy to the host (and cudaDeviceSynchronize)
//               waits for all queued work to finish.
//
//   shared memory / __syncthreads
//               Used only by the reduction kernel kErrNorm: a small scratch array
//               visible to all threads of ONE block (fast on-chip memory, size
//               given as the third launch parameter), and a barrier where every
//               thread of the block waits until all of them arrive. Together
//               they allow a tree reduction inside each block.
//
//   device memory
//               Pointers named d_* point to GPU memory (cudaMalloc). Physics
//               functions from sgn_flux.h run inside kernels because they are
//               SGN_HD (see src/physics/hd.h).
//
// WHAT THIS FILE DOES
//   Single-GPU port of SGN1D::rhs and sgn::integrateAdaptive. The physics comes
//   from sgn_flux.h (SGN_HD), compiled here as device code -- one source of
//   truth, no drift possible.
// =============================================================================

#include "sgn1d_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

// Abort with file/line on any failed CUDA runtime call (allocation, copy, launch).
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(err_), __FILE__, __LINE__);            \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

namespace sgn {

namespace {

// Threads per block for every launch. Must be a power of two, because the tree
// reduction in kErrNorm halves the active range at each level.
constexpr int kThreads = 128;

// --- Kernel 1: nodal conservative flux, one thread per node ----------------
//   Thread g = global node (0..N-1).  Reads U at node g, writes Fnod at node g.
//   Mirrors phase 3a of SGN1D::rhs, but for all elements at once.
__global__ void kNodalFlux(const double* U, const double* Hnod, Params par,
                           int N, double* Fnod) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) return;                      // padding thread of the last block
  double Un[NVAR], F[NVAR];                // thread-local copies (per thread)
  for (int v = 0; v < NVAR; ++v) Un[v] = U[v * N + g];
  fluxX(Un, Hnod[g], par, F);              // SGN_HD physics, same as the CPU
  for (int v = 0; v < NVAR; ++v) Fnod[v * N + g] = F[v];
}

// --- Kernel 2: face extrapolation + discontinuous face fluxes, per element -
//   Thread e = element (0..ne-1). Reads the np nodes of element e, writes the
//   element's two face states UfL/UfR and its own face fluxes FdL/FdR.
//   Mirrors phase 1 and step 3b of SGN1D::rhs.
__global__ void kFaces(const double* U, const double* lL, const double* lR,
                       const double* Hface, Params par, int ne, int np, int N,
                       double* UfL, double* UfR, double* FdL, double* FdR) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= ne) return;
  for (int v = 0; v < NVAR; ++v) {
    double sL = 0.0, sR = 0.0;
    const double* Uv = U + v * N + e * np;   // nodal values of variable v in element e
    for (int j = 0; j < np; ++j) {
      sL += lL[j] * Uv[j];
      sR += lR[j] * Uv[j];
    }
    UfL[e * NVAR + v] = sL;
    UfR[e * NVAR + v] = sR;
  }
  const int IL = e;                             // interface at the left face
  const int IR = (e + 1 == ne) ? 0 : e + 1;     // interface at the right face (periodic)
  fluxX(&UfL[e * NVAR], Hface[IL], par, &FdL[e * NVAR]);
  fluxX(&UfR[e * NVAR], Hface[IR], par, &FdR[e * NVAR]);
}

// --- Kernel 3: interface Rusanov flux + central depth, per interface -------
//   Thread I = interface (0..ne-1), at the left face of element I. Reads the
//   face states of elements I-1 and I (computed by kFaces), writes Fstar and
//   hStar of interface I. Mirrors phase 2 of SGN1D::rhs (central h* only).
__global__ void kInterfaces(const double* UfL, const double* UfR,
                            const double* Hface, Params par, int ne,
                            double* Fstar, double* hStar) {
  const int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= ne) return;
  const int eL = (I == 0) ? ne - 1 : I - 1;   // element left of interface I (periodic)
  const double* UL = &UfR[eL * NVAR]; // left element's right face
  const double* UR = &UfL[I * NVAR];  // right element's left face
  rusanov(UL, UR, Hface[I], par, &Fstar[I * NVAR]);
  hStar[I] = 0.5 * (UL[H] + UR[H]);
}

// --- Kernel 4: volume update, one thread per node ---------------------------
//   Thread g = global node. Recovers its element e and local node i, reads the
//   nodal fluxes of element e and the interface data of its two faces, and
//   writes dU at node g. Mirrors steps 3c-3f of SGN1D::rhs.
__global__ void kVolume(const double* U, const double* Fnod,
                        const double* UfL, const double* UfR,
                        const double* FdL, const double* FdR,
                        const double* Fstar, const double* hStar,
                        const double* D, const double* gLp, const double* gRp,
                        const double* Hnod, const double* Hx, Params par,
                        int ne, int np, int N, double Jinv, double* dU) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) return;
  const int e = g / np;           // element containing node g
  const int i = g - e * np;       // local node index inside element e
  const int IL = e;                             // left interface of e
  const int IR = (e + 1 == ne) ? 0 : e + 1;     // right interface of e

  // FR-corrected flux divergence, per variable.
  double dF[NVAR];
  for (int v = 0; v < NVAR; ++v) {
    double dfdxi = 0.0;
    const double* Fv = Fnod + v * N + e * np;   // nodal flux of variable v in element e
    for (int k = 0; k < np; ++k) dfdxi += D[i * np + k] * Fv[k];
    const double corr = (Fstar[IL * NVAR + v] - FdL[e * NVAR + v]) * gLp[i] +
                        (Fstar[IR * NVAR + v] - FdR[e * NVAR + v]) * gRp[i];
    dF[v] = (dfdxi + corr) * Jinv;
  }

  // FR-corrected depth gradient for the non-conservative term.
  double dhdxi = 0.0;
  {
    const double* hv = U + H * N + e * np;      // nodal h of element e
    for (int k = 0; k < np; ++k) dhdxi += D[i * np + k] * hv[k];
    dhdxi += (hStar[IL] - UfL[e * NVAR + H]) * gLp[i] +
             (hStar[IR] - UfR[e * NVAR + H]) * gRp[i];
  }
  const double hx = dhdxi * Jinv;

  // pointwise NCP and sources, then dU/dt = -dF/dx - B + S
  double Un[NVAR], B[NVAR], S[NVAR];
  for (int v = 0; v < NVAR; ++v) Un[v] = U[v * N + g];
  ncpX(Un, hx, par, B);
  source(Un, Hnod[g], Hx[g], par, S);

  for (int v = 0; v < NVAR; ++v) dU[v * N + g] = -dF[v] - B[v] + S[v];
}

// --- RK stage combination kernels -------------------------------------------
//   One thread per state entry i (all variables, all nodes).
//   kComb1: out = u + a1 k1                   (stage evaluation points)
__global__ void kComb1(double* out, const double* u, double a1,
                       const double* k1, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = u[i] + a1 * k1[i];
}

//   kComb3: out = u + a1 k1 + a2 k2 + a3 k3   (third-order candidate)
__global__ void kComb3(double* out, const double* u, double a1,
                       const double* k1, double a2, const double* k2,
                       double a3, const double* k3, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = u[i] + a1 * k1[i] + a2 * k2[i] + a3 * k3[i];
}

// --- Embedded-error norm: per-block partial sum(r^2) and max|r| -------------
//   One thread per state entry i. Each thread computes the scaled error r of
//   its entry (same formula as rk.cpp). Then, inside each block of kThreads
//   threads, the values are combined by a tree reduction in shared memory:
//     level 1: thread t adds thread t+64's value   (t < 64)
//     level 2: thread t adds thread t+32's value   (t < 32)
//     ...     until thread 0 holds the block's sum and max.
//   __syncthreads() makes all threads of the block finish one level before the
//   next begins. Thread 0 then writes the block partials psum[block],
//   pmax[block]; the host combines the nb partials.
//   Padding threads (i >= n) contribute 0 to both, which is neutral for a sum
//   and for a max of non-negative values.
__global__ void kErrNorm(const double* uNew, const double* u,
                         const double* k1, const double* k2, const double* k3,
                         const double* k4, double dt, double atol, double rtol,
                         int n, double* psum, double* pmax) {
  // per-block shared scratch of 2*kThreads doubles: first half sums, second max
  extern __shared__ double sh[];
  double* ssum = sh;
  double* smax = sh + blockDim.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s = 0.0, m = 0.0;
  if (i < n) {
    const double uEmb = u[i] + dt * (7.0 / 24.0 * k1[i] + 0.25 * k2[i] +
                                     1.0 / 3.0 * k3[i] + 0.125 * k4[i]);
    const double e = uNew[i] - uEmb;                        // embedded error
    const double sc = atol + rtol * fmax(fabs(u[i]), fabs(uNew[i])); // tolerance scale
    const double r = e / sc;
    s = r * r;
    m = fabs(r);
  }
  ssum[threadIdx.x] = s;      // each thread writes only its own slot
  smax[threadIdx.x] = m;
  __syncthreads();            // wait until every thread of the block has written
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      // fold the upper half of the active range into the lower half
      ssum[threadIdx.x] += ssum[threadIdx.x + off];
      smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + off]);
    }
    __syncthreads();          // finish this level on all threads before the next
  }
  if (threadIdx.x == 0) {
    psum[blockIdx.x] = ssum[0];   // block partials, one slot per block
    pmax[blockIdx.x] = smax[0];
  }
}

// number of blocks needed to cover n items with kThreads threads per block
inline int nBlocks(int n) { return (n + kThreads - 1) / kThreads; }

// allocate n doubles of device memory
double* devAlloc(size_t n) {
  double* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(double)));
  return p;
}

// copy n doubles host -> device
void devUpload(double* d, const double* h, size_t n) {
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(double), cudaMemcpyHostToDevice));
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor: allocate every device buffer once and upload the static data.
// ---------------------------------------------------------------------------
SGN1DCuda::SGN1DCuda(const SGN1D& host)
    : P_(host.order()), ne_(host.nElem()), N_(host.nDof()), np_(host.order() + 1),
      J_(host.jacobian()), par_(host.params()) {
  const auto& op = host.ops();

  // FR operator tables (tiny: at most np*np doubles)
  d_D = devAlloc((size_t)np_ * np_);
  d_lL = devAlloc(np_);
  d_lR = devAlloc(np_);
  d_gLp = devAlloc(np_);
  d_gRp = devAlloc(np_);
  devUpload(d_D, op.D().data(), (size_t)np_ * np_);
  devUpload(d_lL, op.lL().data(), np_);
  devUpload(d_lR, op.lR().data(), np_);
  devUpload(d_gLp, op.gLprime().data(), np_);
  devUpload(d_gRp, op.gRprime().data(), np_);

  // bathymetry precomputed by the host solver
  d_Hnod = devAlloc(N_);
  d_Hx = devAlloc(N_);
  d_Hface = devAlloc(ne_);
  devUpload(d_Hnod, host.stillDepthNodal().data(), N_);
  devUpload(d_Hx, host.stillDepthGrad().data(), N_);
  devUpload(d_Hface, host.stillDepthFace().data(), ne_);

  // state and RK stages
  const size_t nTot = (size_t)NVAR * N_;
  d_U = devAlloc(nTot);
  d_k1 = devAlloc(nTot);
  d_k2 = devAlloc(nTot);
  d_k3 = devAlloc(nTot);
  d_k4 = devAlloc(nTot);
  d_tmp = devAlloc(nTot);
  d_uNew = devAlloc(nTot);

  // RHS work buffers
  d_Fnod = devAlloc(nTot);
  d_UfL = devAlloc((size_t)ne_ * NVAR);
  d_UfR = devAlloc((size_t)ne_ * NVAR);
  d_FdL = devAlloc((size_t)ne_ * NVAR);
  d_FdR = devAlloc((size_t)ne_ * NVAR);
  d_Fstar = devAlloc((size_t)ne_ * NVAR);
  d_hStar = devAlloc(ne_);

  // one error partial per block of a full-state launch
  nBlocksErr_ = nBlocks((int)nTot);
  d_psum = devAlloc(nBlocksErr_);
  d_pmax = devAlloc(nBlocksErr_);
}

// Destructor: release every device allocation.
SGN1DCuda::~SGN1DCuda() {
  for (double* p : {d_D, d_lL, d_lR, d_gLp, d_gRp, d_Hnod, d_Hx, d_Hface,
                    d_U, d_k1, d_k2, d_k3, d_k4, d_tmp, d_uNew, d_Fnod,
                    d_UfL, d_UfR, d_FdL, d_FdR, d_Fstar, d_hStar,
                    d_psum, d_pmax})
    if (p) cudaFree(p);
}

// host state -> device state
void SGN1DCuda::upload(const State& U) {
  devUpload(d_U, U.data(), (size_t)NVAR * N_);
}

// device state -> host state (blocks until queued kernels finish)
void SGN1DCuda::download(State& U) const {
  U.resize((size_t)NVAR * N_);
  CUDA_CHECK(cudaMemcpy(U.data(), d_U, (size_t)NVAR * N_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// rhsDevice: dU = R(U) with four launches, in dependency order. Each launch uses
// outputs of the previous ones; they are queued on the same device stream, so
// they execute in order.
// ---------------------------------------------------------------------------
void SGN1DCuda::rhsDevice(const double* U, double* dU) {
  const double Jinv = 1.0 / J_;
  // 1. nodal fluxes (one thread per node)
  kNodalFlux<<<nBlocks(N_), kThreads>>>(U, d_Hnod, par_, N_, d_Fnod);
  // 2. face states and element face fluxes (one thread per element)
  kFaces<<<nBlocks(ne_), kThreads>>>(U, d_lL, d_lR, d_Hface, par_, ne_, np_,
                                     N_, d_UfL, d_UfR, d_FdL, d_FdR);
  // 3. Rusanov fluxes and central depths (one thread per interface)
  kInterfaces<<<nBlocks(ne_), kThreads>>>(d_UfL, d_UfR, d_Hface, par_, ne_,
                                          d_Fstar, d_hStar);
  // 4. assemble dU at every node (one thread per node)
  kVolume<<<nBlocks(N_), kThreads>>>(U, d_Fnod, d_UfL, d_UfR, d_FdL, d_FdR,
                                     d_Fstar, d_hStar, d_D, d_gLp, d_gRp,
                                     d_Hnod, d_Hx, par_, ne_, np_, N_, Jinv,
                                     dU);
  CUDA_CHECK(cudaGetLastError());   // report a failed launch
}

// One RHS evaluation of the uploaded state, downloaded (parity test).
void SGN1DCuda::rhsOnce(State& dU) {
  rhsDevice(d_U, d_k1);
  dU.resize((size_t)NVAR * N_);
  CUDA_CHECK(cudaMemcpy(dU.data(), d_k1, (size_t)NVAR * N_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// integrateAdaptive: device-resident BS3(2). The algorithm is line-for-line the
// one of rk.cpp (see the comments there); the state, stages and linear
// combinations live on the GPU, and only the nb per-block error partials are
// copied to the host each step to decide accept/reject.
// ---------------------------------------------------------------------------
AdaptiveRKStats SGN1DCuda::integrateAdaptive(double t0, double tEnd, double dt0,
                                             const AdaptiveRKOptions& opt) {
  // Same Bogacki-Shampine 3(2) + PI controller as sgn::integrateAdaptive
  // (src/solver/rk.cpp) -- keep the constants in lockstep.
  const int orderLow = 2;
  const double expo = 1.0 / (orderLow + 1.0);
  const double alpha = 0.7 * expo;
  const double beta = 0.4 * expo;

  AdaptiveRKStats st;
  st.tReached = t0;
  const int n = NVAR * N_;                               // state size
  const int nb = nBlocks(n);                             // blocks per full-state launch
  const size_t shBytes = 2 * kThreads * sizeof(double);  // kErrNorm shared scratch

  std::vector<double> psum(nBlocksErr_), pmax(nBlocksErr_);  // host copies of partials

  double t = t0;
  double dt = std::min(dt0, opt.dtMax);
  double errPrev = 1.0;

  rhsDevice(d_U, d_k1); // FSAL head
  ++st.rhsEvals;

  while (t < tEnd - 1e-15 * std::max(1.0, std::abs(tEnd))) {
    if (t + dt > tEnd) dt = tEnd - t;

    // stages and third-order candidate, all on device
    kComb1<<<nb, kThreads>>>(d_tmp, d_U, 0.5 * dt, d_k1, n);
    rhsDevice(d_tmp, d_k2);
    kComb1<<<nb, kThreads>>>(d_tmp, d_U, 0.75 * dt, d_k2, n);
    rhsDevice(d_tmp, d_k3);
    kComb3<<<nb, kThreads>>>(d_uNew, d_U, dt * 2.0 / 9.0, d_k1,
                             dt * 1.0 / 3.0, d_k2, dt * 4.0 / 9.0, d_k3, n);
    rhsDevice(d_uNew, d_k4);
    st.rhsEvals += 3;

    // error partials on device, then two small copies to the host
    // (these copies are the synchronization point of the step)
    kErrNorm<<<nb, kThreads, shBytes>>>(d_uNew, d_U, d_k1, d_k2, d_k3, d_k4,
                                        dt, opt.atol, opt.rtol, n, d_psum,
                                        d_pmax);
    CUDA_CHECK(cudaMemcpy(psum.data(), d_psum, nb * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pmax.data(), d_pmax, nb * sizeof(double),
                          cudaMemcpyDeviceToHost));
    // combine the block partials on the host, in block order
    double errL2 = 0.0, errMax = 0.0;
    for (int b = 0; b < nb; ++b) {
      errL2 += psum[b];
      errMax = std::max(errMax, pmax[b]);
    }
    errL2 = std::sqrt(errL2 / (double)n);
    const double errNorm = std::max(errL2, 1e-300);

    if (errNorm <= 1.0) {
      // accept: swap device POINTERS (no data copy)
      t += dt;
      std::swap(d_U, d_uNew);
      std::swap(d_k1, d_k4); // FSAL
      ++st.accepted;
      st.tReached = t;
      st.dtLast = dt;
      st.maxNormLast = errMax;
      errPrev = errNorm;
    } else {
      ++st.rejected;
    }

    // PI controller (identical to rk.cpp)
    double fac = opt.safety * std::pow(errNorm, -alpha) *
                 std::pow(errPrev, beta);
    fac = std::clamp(fac, opt.facMin, opt.facMax);
    double dtNext = dt * fac;
    dtNext = std::min(dtNext, opt.dtMax);
    if (dtNext < opt.dtMin) {
      st.aborted = true;
      st.dtLast = dtNext;
      break;
    }
    dt = dtNext;
  }

  return st;
}

} // namespace sgn
