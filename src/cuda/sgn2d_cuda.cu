// =============================================================================
// sgn2d_cuda.cu -- single-GPU kernels and integrator for the 2D FR solver
// =============================================================================
//
// For the CUDA vocabulary (kernel, launch, thread index, blocks, shared memory)
// read the primer at the top of sgn1d_cuda.cu first.
//
// RHS = five launches, in dependency order:
//   1. kNodalFlux2D   one thread per node       nodal F and G
//   2. kFaces2D       one thread per element    face states + element face fluxes
//   3. kInterfacesX   one thread per (element, y-node) on the element's left edge
//   4. kInterfacesY   one thread per (element, x-node) on the element's bottom edge
//   5. kVolume2D      one thread per node       assemble dU
// Index conventions are those of sgn2d.h: node (a, b) of element e is at
// e*npp + a + np*b; face arrays are indexed (e*np + transverse node)*NVAR + var.
// =============================================================================

#include "sgn2d_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

// Abort with file/line on any failed CUDA runtime call.
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

using sgn2d::NVAR;
using sgn2d::H;

// threads per block (power of two, required by the kErrNorm tree reduction)
constexpr int kThreads = 128;

// --- Kernel 1: nodal fluxes F and G, one thread per node --------------------
//   Thread g = global node. Reads U at g, writes Fnod and Gnod at g.
__global__ void kNodalFlux2D(const double* U, const double* Hnod,
                             sgn2d::Params par, int Nd, double* Fnod,
                             double* Gnod) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= Nd) return;                     // padding thread
  double Un[NVAR], F[NVAR], G[NVAR];
  for (int v = 0; v < NVAR; ++v) Un[v] = U[(size_t)v * Nd + g];
  sgn2d::fluxX(Un, Hnod[g], par, F);
  sgn2d::fluxY(Un, Hnod[g], par, G);
  for (int v = 0; v < NVAR; ++v) {
    Fnod[(size_t)v * Nd + g] = F[v];
    Gnod[(size_t)v * Nd + g] = G[v];
  }
}

// --- Kernel 2: face extrapolation + discontinuous face fluxes, per element --
//   Thread e = element. Reads the npp nodes of e, writes its four face-state
//   lines (x: left/right at every y-node b; y: bottom/top at every x-node a) and
//   the element's own flux on each face. The right / top face bathymetry is the
//   left / bottom edge value of the right / top neighbour (eR, eT).
//   One thread per element, each reading its whole element block twice (once
//   per direction): this is the kernel the paper's profiling identifies as the
//   memory-latency hotspot (the multi-GPU version has the same structure).
__global__ void kFaces2D(const double* U, const double* lL, const double* lR,
                         const double* HfaceX, const double* HfaceY,
                         const int* nbrR, const int* nbrT, sgn2d::Params par,
                         int ne, int np, int Nd, double* xFaceL,
                         double* xFaceR, double* yFaceB, double* yFaceT,
                         double* FdXL, double* FdXR, double* GdYB,
                         double* GdYT) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= ne) return;
  const int npp = np * np;
  const int eR = nbrR[e], eT = nbrT[e];   // right and top neighbours

  for (int v = 0; v < NVAR; ++v) {
    const double* Uv = U + (size_t)v * Nd + (size_t)e * npp;  // element e, variable v
    // x-faces: interpolate along each x-line (fixed b)
    for (int b = 0; b < np; ++b) {
      double sL = 0.0, sR = 0.0;
      for (int a = 0; a < np; ++a) {
        const double val = Uv[a + np * b];
        sL += lL[a] * val; sR += lR[a] * val;
      }
      xFaceL[((size_t)e * np + b) * NVAR + v] = sL;
      xFaceR[((size_t)e * np + b) * NVAR + v] = sR;
    }
    // y-faces: interpolate along each y-line (fixed a)
    for (int a = 0; a < np; ++a) {
      double sB = 0.0, sT = 0.0;
      for (int b = 0; b < np; ++b) {
        const double val = Uv[a + np * b];
        sB += lL[b] * val; sT += lR[b] * val;
      }
      yFaceB[((size_t)e * np + a) * NVAR + v] = sB;
      yFaceT[((size_t)e * np + a) * NVAR + v] = sT;
    }
  }
  // Discontinuous face fluxes with the SAME face bathymetry as Rusanov.
  for (int b = 0; b < np; ++b) {
    sgn2d::fluxX(&xFaceL[((size_t)e * np + b) * NVAR], HfaceX[e * np + b], par,
                 &FdXL[((size_t)e * np + b) * NVAR]);
    sgn2d::fluxX(&xFaceR[((size_t)e * np + b) * NVAR], HfaceX[eR * np + b], par,
                 &FdXR[((size_t)e * np + b) * NVAR]);
  }
  for (int a = 0; a < np; ++a) {
    sgn2d::fluxY(&yFaceB[((size_t)e * np + a) * NVAR], HfaceY[e * np + a], par,
                 &GdYB[((size_t)e * np + a) * NVAR]);
    sgn2d::fluxY(&yFaceT[((size_t)e * np + a) * NVAR], HfaceY[eT * np + a], par,
                 &GdYT[((size_t)e * np + a) * NVAR]);
  }
}

// --- Kernel 3a/3b: interface Rusanov + central depths -----------------------
// x-interface owned by element e at its LEFT edge, transverse node b.
//   Thread idx = e*np + b (one thread per face node of every left edge).
//   Reads the right face of the left neighbour and the left face of e, writes
//   FstarX and hStarX at (e, b).
__global__ void kInterfacesX(const double* xFaceL, const double* xFaceR,
                             const double* HfaceX, const int* nbrL,
                             sgn2d::Params par, int ne, int np,
                             double* FstarX, double* hStarX) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ne * np) return;
  const int e = idx / np, b = idx - e * np;   // owning element and y-node
  const int eL = nbrL[e];
  const double* UL = &xFaceR[((size_t)eL * np + b) * NVAR];   // left side of the interface
  const double* UR = &xFaceL[((size_t)e * np + b) * NVAR];    // right side
  sgn2d::rusanov(UL, UR, 0, HfaceX[e * np + b], par,
                 &FstarX[((size_t)e * np + b) * NVAR]);
  hStarX[e * np + b] = 0.5 * (UL[H] + UR[H]);
}

// y-interface owned by element e at its BOTTOM edge, transverse node a.
//   Thread idx = e*np + a. Mirror of kInterfacesX in y.
__global__ void kInterfacesY(const double* yFaceB, const double* yFaceT,
                             const double* HfaceY, const int* nbrB,
                             sgn2d::Params par, int ne, int np,
                             double* FstarY, double* hStarY) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ne * np) return;
  const int e = idx / np, a = idx - e * np;   // owning element and x-node
  const int eB = nbrB[e];
  const double* UL = &yFaceT[((size_t)eB * np + a) * NVAR];   // below the interface
  const double* UR = &yFaceB[((size_t)e * np + a) * NVAR];    // above
  sgn2d::rusanov(UL, UR, 1, HfaceY[e * np + a], par,
                 &FstarY[((size_t)e * np + a) * NVAR]);
  hStarY[e * np + a] = 0.5 * (UL[H] + UR[H]);
}

// --- Kernel 4: volume update, one thread per node ----------------------------
//   Thread g = global node -> element e, local node (a, b). Reads the element's
//   nodal fluxes, the four face lines and the interface data (left/bottom stored
//   at e, right/top stored at the neighbours eR/eT), writes dU at g.
//   Mirrors step 5 of SGN2D::rhs.
__global__ void kVolume2D(const double* U, const double* Fnod,
                          const double* Gnod, const double* xFaceL,
                          const double* xFaceR, const double* yFaceB,
                          const double* yFaceT, const double* FdXL,
                          const double* FdXR, const double* GdYB,
                          const double* GdYT, const double* FstarX,
                          const double* FstarY, const double* hStarX,
                          const double* hStarY, const double* D,
                          const double* gLp, const double* gRp,
                          const double* Hnod, const double* Hx,
                          const double* Hy, const int* nbrR, const int* nbrT,
                          sgn2d::Params par, int ne, int np, int Nd,
                          double JxInv, double JyInv, double* dU) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= Nd) return;
  const int npp = np * np;
  const int e = g / npp;                    // element of node g
  const int loc = g - e * npp;              // local node index
  const int a = loc % np, b = loc / np;     // x- and y-node within the element
  const int eR = nbrR[e], eT = nbrT[e];     // neighbours holding the right/top interfaces

  // FR-corrected divergence of F along x and G along y
  double dF[NVAR], dG[NVAR];
  for (int v = 0; v < NVAR; ++v) {
    double dfx = 0.0, dgy = 0.0;
    const double* Fv = Fnod + (size_t)v * Nd + (size_t)e * npp;
    const double* Gv = Gnod + (size_t)v * Nd + (size_t)e * npp;
    for (int k = 0; k < np; ++k) {
      dfx += D[a * np + k] * Fv[k + np * b];
      dgy += D[b * np + k] * Gv[a + np * k];
    }
    const double cx =
        (FstarX[((size_t)e * np + b) * NVAR + v] -       // left interface of e
         FdXL[((size_t)e * np + b) * NVAR + v]) * gLp[a] +
        (FstarX[((size_t)eR * np + b) * NVAR + v] -      // right interface = left of eR
         FdXR[((size_t)e * np + b) * NVAR + v]) * gRp[a];
    const double cy =
        (FstarY[((size_t)e * np + a) * NVAR + v] -       // bottom interface of e
         GdYB[((size_t)e * np + a) * NVAR + v]) * gLp[b] +
        (FstarY[((size_t)eT * np + a) * NVAR + v] -      // top interface = bottom of eT
         GdYT[((size_t)e * np + a) * NVAR + v]) * gRp[b];
    dF[v] = (dfx + cx) * JxInv;
    dG[v] = (dgy + cy) * JyInv;
  }

  // FR-corrected depth gradients.
  double dhx = 0.0, dhy = 0.0;
  {
    const double* hv = U + (size_t)H * Nd + (size_t)e * npp;   // nodal h of element e
    for (int k = 0; k < np; ++k) {
      dhx += D[a * np + k] * hv[k + np * b];
      dhy += D[b * np + k] * hv[a + np * k];
    }
    dhx += (hStarX[e * np + b] -
            xFaceL[((size_t)e * np + b) * NVAR + H]) * gLp[a] +
           (hStarX[eR * np + b] -
            xFaceR[((size_t)e * np + b) * NVAR + H]) * gRp[a];
    dhy += (hStarY[e * np + a] -
            yFaceB[((size_t)e * np + a) * NVAR + H]) * gLp[b] +
           (hStarY[eT * np + a] -
            yFaceT[((size_t)e * np + a) * NVAR + H]) * gRp[b];
  }
  const double hx = dhx * JxInv, hy = dhy * JyInv;

  // pointwise NCPs and sources, then assemble dU
  double Un[NVAR], Bx[NVAR], By[NVAR], S[NVAR];
  for (int v = 0; v < NVAR; ++v) Un[v] = U[(size_t)v * Nd + g];
  sgn2d::ncpX(Un, hx, par, Bx);
  sgn2d::ncpY(Un, hy, par, By);
  sgn2d::source(Un, Hnod[g], Hx[g], Hy[g], par, S);

  for (int v = 0; v < NVAR; ++v)
    dU[(size_t)v * Nd + g] = -dF[v] - dG[v] - Bx[v] - By[v] + S[v];
}

// --- RK stage combination + error norm (same as the 1D port) ----------------
// One thread per state entry. See sgn1d_cuda.cu for the detailed comments.
__global__ void kComb1(double* out, const double* u, double a1,
                       const double* k1, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = u[i] + a1 * k1[i];                 // out = u + a1 k1
}

__global__ void kComb3(double* out, const double* u, double a1,
                       const double* k1, double a2, const double* k2,
                       double a3, const double* k3, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = u[i] + a1 * k1[i] + a2 * k2[i] + a3 * k3[i];
}

// Scaled embedded error per entry, tree-reduced to one (sum, max) pair per block
// in block shared memory; thread 0 of each block writes psum/pmax[block].
__global__ void kErrNorm(const double* uNew, const double* u,
                         const double* k1, const double* k2, const double* k3,
                         const double* k4, double dt, double atol, double rtol,
                         int n, double* psum, double* pmax) {
  extern __shared__ double sh[];     // 2*kThreads doubles per block
  double* ssum = sh;                 // first half: running sums
  double* smax = sh + blockDim.x;    // second half: running maxima
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s = 0.0, m = 0.0;           // padding threads contribute 0
  if (i < n) {
    const double uEmb = u[i] + dt * (7.0 / 24.0 * k1[i] + 0.25 * k2[i] +
                                     1.0 / 3.0 * k3[i] + 0.125 * k4[i]);
    const double e = uNew[i] - uEmb;
    const double sc = atol + rtol * fmax(fabs(u[i]), fabs(uNew[i]));
    const double r = e / sc;
    s = r * r;
    m = fabs(r);
  }
  ssum[threadIdx.x] = s;
  smax[threadIdx.x] = m;
  __syncthreads();                   // all threads of the block have written
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) {         // fold the upper half into the lower half
      ssum[threadIdx.x] += ssum[threadIdx.x + off];
      smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + off]);
    }
    __syncthreads();                 // finish the level before the next
  }
  if (threadIdx.x == 0) {
    psum[blockIdx.x] = ssum[0];
    pmax[blockIdx.x] = smax[0];
  }
}

// blocks needed to cover n items
inline int nBlocks(int n) { return (n + kThreads - 1) / kThreads; }

// device allocation / upload helpers (double and int arrays)
double* devAlloc(size_t n) {
  double* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(double)));
  return p;
}
int* devAllocInt(size_t n) {
  int* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(int)));
  return p;
}
void devUpload(double* d, const double* h, size_t n) {
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(double), cudaMemcpyHostToDevice));
}
void devUploadInt(int* d, const int* h, size_t n) {
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice));
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor: allocate all device buffers once, upload static data.
// ---------------------------------------------------------------------------
SGN2DCuda::SGN2DCuda(const SGN2D& host)
    : P_(host.order()), np_(host.order() + 1),
      npp_((host.order() + 1) * (host.order() + 1)), ne_(host.nElem()),
      Nd_(host.nDof()), Jx_(host.mesh().Jx()), Jy_(host.mesh().Jy()),
      par_(host.params()) {
  const auto& op = host.ops();

  // 1D FR operator tables
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

  // bathymetry precomputed on the host
  d_Hnod = devAlloc(Nd_);
  d_Hx = devAlloc(Nd_);
  d_Hy = devAlloc(Nd_);
  d_HfaceX = devAlloc((size_t)ne_ * np_);
  d_HfaceY = devAlloc((size_t)ne_ * np_);
  devUpload(d_Hnod, host.stillDepthNodal().data(), Nd_);
  devUpload(d_Hx, host.stillDepthGradX().data(), Nd_);
  devUpload(d_Hy, host.stillDepthGradY().data(), Nd_);
  devUpload(d_HfaceX, host.stillDepthFaceX().data(), (size_t)ne_ * np_);
  devUpload(d_HfaceY, host.stillDepthFaceY().data(), (size_t)ne_ * np_);

  // Connectivity: neighbour ids of every element, so kernels can find the
  // neighbour without the host mesh object (face 3=left, 1=right, 0=bottom, 2=top).
  {
    std::vector<int> nL(ne_), nR(ne_), nB(ne_), nT(ne_);
    for (int e = 0; e < ne_; ++e) {
      nL[e] = host.mesh().neighbor(e, 3);
      nR[e] = host.mesh().neighbor(e, 1);
      nB[e] = host.mesh().neighbor(e, 0);
      nT[e] = host.mesh().neighbor(e, 2);
    }
    d_nbrL = devAllocInt(ne_);
    d_nbrR = devAllocInt(ne_);
    d_nbrB = devAllocInt(ne_);
    d_nbrT = devAllocInt(ne_);
    devUploadInt(d_nbrL, nL.data(), ne_);
    devUploadInt(d_nbrR, nR.data(), ne_);
    devUploadInt(d_nbrB, nB.data(), ne_);
    devUploadInt(d_nbrT, nT.data(), ne_);
  }

  // state and RK stages
  const size_t nTot = (size_t)NVAR * Nd_;
  d_U = devAlloc(nTot);
  d_k1 = devAlloc(nTot);
  d_k2 = devAlloc(nTot);
  d_k3 = devAlloc(nTot);
  d_k4 = devAlloc(nTot);
  d_tmp = devAlloc(nTot);
  d_uNew = devAlloc(nTot);

  // RHS work buffers
  d_Fnod = devAlloc(nTot);
  d_Gnod = devAlloc(nTot);
  const size_t nFace = (size_t)ne_ * np_ * NVAR;   // one face line per element
  d_xFaceL = devAlloc(nFace);
  d_xFaceR = devAlloc(nFace);
  d_yFaceB = devAlloc(nFace);
  d_yFaceT = devAlloc(nFace);
  d_FdXL = devAlloc(nFace);
  d_FdXR = devAlloc(nFace);
  d_GdYB = devAlloc(nFace);
  d_GdYT = devAlloc(nFace);
  d_FstarX = devAlloc(nFace);
  d_FstarY = devAlloc(nFace);
  d_hStarX = devAlloc((size_t)ne_ * np_);
  d_hStarY = devAlloc((size_t)ne_ * np_);

  // error partials: one per block of a full-state launch
  nBlocksErr_ = nBlocks((int)nTot);
  d_psum = devAlloc(nBlocksErr_);
  d_pmax = devAlloc(nBlocksErr_);
}

// Destructor: release every device allocation.
SGN2DCuda::~SGN2DCuda() {
  for (double* p : {d_D, d_lL, d_lR, d_gLp, d_gRp, d_Hnod, d_Hx, d_Hy,
                    d_HfaceX, d_HfaceY, d_U, d_k1, d_k2, d_k3, d_k4, d_tmp,
                    d_uNew, d_Fnod, d_Gnod, d_xFaceL, d_xFaceR, d_yFaceB,
                    d_yFaceT, d_FdXL, d_FdXR, d_GdYB, d_GdYT, d_FstarX,
                    d_FstarY, d_hStarX, d_hStarY, d_psum, d_pmax})
    if (p) cudaFree(p);
  for (int* p : {d_nbrL, d_nbrR, d_nbrB, d_nbrT})
    if (p) cudaFree(p);
}

// host -> device state
void SGN2DCuda::upload(const State& U) {
  devUpload(d_U, U.data(), (size_t)NVAR * Nd_);
}

// device -> host state (waits for queued kernels)
void SGN2DCuda::download(State& U) const {
  U.resize((size_t)NVAR * Nd_);
  CUDA_CHECK(cudaMemcpy(U.data(), d_U, (size_t)NVAR * Nd_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// rhsDevice: dU = R(U), five launches in dependency order (see file header).
// ---------------------------------------------------------------------------
void SGN2DCuda::rhsDevice(const double* U, double* dU) {
  // 1. nodal fluxes, one thread per node
  kNodalFlux2D<<<nBlocks(Nd_), kThreads>>>(U, d_Hnod, par_, Nd_, d_Fnod,
                                           d_Gnod);
  // 2. face states + element face fluxes, one thread per element
  kFaces2D<<<nBlocks(ne_), kThreads>>>(U, d_lL, d_lR, d_HfaceX, d_HfaceY,
                                       d_nbrR, d_nbrT, par_, ne_, np_, Nd_,
                                       d_xFaceL, d_xFaceR, d_yFaceB, d_yFaceT,
                                       d_FdXL, d_FdXR, d_GdYB, d_GdYT);
  // 3. x interfaces, one thread per left-edge face node
  kInterfacesX<<<nBlocks(ne_ * np_), kThreads>>>(d_xFaceL, d_xFaceR, d_HfaceX,
                                                 d_nbrL, par_, ne_, np_,
                                                 d_FstarX, d_hStarX);
  // 4. y interfaces, one thread per bottom-edge face node
  kInterfacesY<<<nBlocks(ne_ * np_), kThreads>>>(d_yFaceB, d_yFaceT, d_HfaceY,
                                                 d_nbrB, par_, ne_, np_,
                                                 d_FstarY, d_hStarY);
  // 5. assemble dU, one thread per node
  kVolume2D<<<nBlocks(Nd_), kThreads>>>(
      U, d_Fnod, d_Gnod, d_xFaceL, d_xFaceR, d_yFaceB, d_yFaceT, d_FdXL,
      d_FdXR, d_GdYB, d_GdYT, d_FstarX, d_FstarY, d_hStarX, d_hStarY, d_D,
      d_gLp, d_gRp, d_Hnod, d_Hx, d_Hy, d_nbrR, d_nbrT, par_, ne_, np_, Nd_,
      1.0 / Jx_, 1.0 / Jy_, dU);
  CUDA_CHECK(cudaGetLastError());   // report a failed launch
}

// One RHS of the uploaded state, downloaded (parity test).
void SGN2DCuda::rhsOnce(State& dU) {
  rhsDevice(d_U, d_k1);
  dU.resize((size_t)NVAR * Nd_);
  CUDA_CHECK(cudaMemcpy(dU.data(), d_k1, (size_t)NVAR * Nd_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// integrateAdaptive: device-resident BS3(2), line-for-line the algorithm of
// rk.cpp; see sgn1d_cuda.cu for the step-by-step comments.
// ---------------------------------------------------------------------------
AdaptiveRKStats SGN2DCuda::integrateAdaptive(double t0, double tEnd,
                                             double dt0,
                                             const AdaptiveRKOptions& opt) {
  const int orderLow = 2;
  const double expo = 1.0 / (orderLow + 1.0);
  const double alpha = 0.7 * expo;   // PI exponents
  const double beta = 0.4 * expo;

  AdaptiveRKStats st;
  st.tReached = t0;
  const int n = NVAR * Nd_;                               // state size
  const int nb = nBlocks(n);                              // blocks per launch
  const size_t shBytes = 2 * kThreads * sizeof(double);   // kErrNorm shared scratch
  std::vector<double> psum(nBlocksErr_), pmax(nBlocksErr_);

  double t = t0;
  double dt = std::min(dt0, opt.dtMax);
  double errPrev = 1.0;

  rhsDevice(d_U, d_k1);   // FSAL head
  ++st.rhsEvals;

  while (t < tEnd - 1e-15 * std::max(1.0, std::abs(tEnd))) {
    if (t + dt > tEnd) dt = tEnd - t;

    // stages and candidate, on device
    kComb1<<<nb, kThreads>>>(d_tmp, d_U, 0.5 * dt, d_k1, n);
    rhsDevice(d_tmp, d_k2);
    kComb1<<<nb, kThreads>>>(d_tmp, d_U, 0.75 * dt, d_k2, n);
    rhsDevice(d_tmp, d_k3);
    kComb3<<<nb, kThreads>>>(d_uNew, d_U, dt * 2.0 / 9.0, d_k1,
                             dt * 1.0 / 3.0, d_k2, dt * 4.0 / 9.0, d_k3, n);
    rhsDevice(d_uNew, d_k4);
    st.rhsEvals += 3;

    // error partials on device, copied to the host (synchronization point)
    kErrNorm<<<nb, kThreads, shBytes>>>(d_uNew, d_U, d_k1, d_k2, d_k3, d_k4,
                                        dt, opt.atol, opt.rtol, n, d_psum,
                                        d_pmax);
    CUDA_CHECK(cudaMemcpy(psum.data(), d_psum, nb * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pmax.data(), d_pmax, nb * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double errL2 = 0.0, errMax = 0.0;
    for (int bb = 0; bb < nb; ++bb) {   // combine block partials in block order
      errL2 += psum[bb];
      errMax = std::max(errMax, pmax[bb]);
    }
    errL2 = std::sqrt(errL2 / (double)n);
    const double errNorm = std::max(errL2, 1e-300);

    if (errNorm <= 1.0) {             // accept: pointer swaps, no copies
      t += dt;
      std::swap(d_U, d_uNew);
      std::swap(d_k1, d_k4);          // FSAL
      ++st.accepted;
      st.tReached = t;
      st.dtLast = dt;
      st.maxNormLast = errMax;
      errPrev = errNorm;
    } else {
      ++st.rejected;
    }

    // PI controller
    double fac = opt.safety * std::pow(errNorm, -alpha) *
                 std::pow(errPrev, beta);
    fac = std::clamp(fac, opt.facMin, opt.facMax);
    double dtNext = std::min(dt * fac, opt.dtMax);
    if (dtNext < opt.dtMin) { st.aborted = true; st.dtLast = dtNext; break; }
    dt = dtNext;
  }
  return st;
}

} // namespace sgn
