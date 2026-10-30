// =============================================================================
// sgn2d_cuda_mpi.cu -- kernels, halo exchange and integrator of the multi-GPU
//                      2D FR solver (production)
// =============================================================================
//
// Read first: the CUDA primer at the top of sgn1d_cuda.cu, and the MPI summary
// in src/parallel/sgn1d_mpi.h.
//
// EXECUTION MODEL
//   srun -n R starts R identical processes; each one picks one GPU of its node
//   (constructor), owns a slab of w element columns and runs the kernels below
//   on it. Ranks only talk to their left/right ring neighbour (halo lines, once
//   per RHS) and to everybody through the per-step error-norm Allreduce.
//
// RHS = five kernel launches plus one exchange, in this order (rhsDevice):
//   1. kNodalFlux    one thread per node       nodal F and G
//   2. kFaces        one thread per element    face states + element face fluxes
//   3. kPack         one thread per (row, node) of the slab boundary: packs the
//                    leftmost / rightmost column x-faces into send buffers
//   -- device synchronization, then two blocking MPI_Sendrecv on device pointers
//   4. kInterfaces   one thread per (element, transverse node): Rusanov fluxes
//                    and central depths on all four edges, reading ghost lines
//                    at the slab boundary
//   5. kVolume       one thread per node       assemble dU
//   The exchange is blocking and not overlapped with interior work; profiling
//   shows it is a small fraction of the step at the meshes of the paper.
//
// INDEXING (as in src/parallel/sgn2d_mpi.h)
//   owned element e = lx + w*ley, node (a, b) at e*npp + a + np*b,
//   face arrays [(e*np + t)*NVAR + var], halo lines [(ley*np + t)*NVAR + var].
// =============================================================================

#include "sgn2d_cuda_mpi.h"

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

// --- nodal fluxes, one thread per node --------------------------------------
//   Thread g = owned global-local node (0..Nl-1). Writes Fnod, Gnod at g.
__global__ void kNodalFlux(const double* U, const double* Hnod,
                           sgn2d::Params par, int Nl, double* Fnod,
                           double* Gnod) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= Nl) return;                  // padding thread
  double Un[NVAR], F[NVAR], G[NVAR];
  for (int v = 0; v < NVAR; ++v) Un[v] = U[(size_t)v * Nl + g];
  sgn2d::fluxX(Un, Hnod[g], par, F);
  sgn2d::fluxY(Un, Hnod[g], par, G);
  for (int v = 0; v < NVAR; ++v) {
    Fnod[(size_t)v * Nl + g] = F[v];
    Gnod[(size_t)v * Nl + g] = G[v];
  }
}

// --- face extrapolation + discontinuous face fluxes, per element ------------
//   Thread e = owned element. Reads the npp nodes of e (twice: once per
//   direction), writes its four face-state lines and its own flux on each face,
//   using the per-element face bathymetry of all four edges (HfXL..HfYT).
//
//   PERFORMANCE NOTE: this kernel dominates the step (45-75% of RHS time in the
//   paper's Nsight profile). One thread per element means neighbouring threads
//   read element blocks (P+1)^2 doubles apart, so a warp touches many separate
//   cache lines per load; its cost per element depends on whether the working
//   set fits in the GPU's L2 cache. That is the mechanism behind the measured
//   superlinear strong scaling. Changing the thread layout would change the
//   performance numbers of the paper.
__global__ void kFaces(const double* U, const double* lL, const double* lR,
                       const double* HfXL, const double* HfXR,
                       const double* HfYB, const double* HfYT,
                       sgn2d::Params par, int neL, int np, int Nl,
                       double* xFaceL, double* xFaceR, double* yFaceB,
                       double* yFaceT, double* FdXL, double* FdXR,
                       double* GdYB, double* GdYT) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= neL) return;
  const int npp = np * np;

  for (int v = 0; v < NVAR; ++v) {
    const double* Uv = U + (size_t)v * Nl + (size_t)e * npp;   // element e, variable v
    // x-faces: along each x-line (fixed b)
    for (int b = 0; b < np; ++b) {
      double sL = 0.0, sR = 0.0;
      for (int a = 0; a < np; ++a) {
        const double val = Uv[a + np * b];
        sL += lL[a] * val; sR += lR[a] * val;
      }
      xFaceL[((size_t)e * np + b) * NVAR + v] = sL;
      xFaceR[((size_t)e * np + b) * NVAR + v] = sR;
    }
    // y-faces: along each y-line (fixed a)
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
  // element's own flux on each face, with that face's still depth
  for (int b = 0; b < np; ++b) {
    sgn2d::fluxX(&xFaceL[((size_t)e * np + b) * NVAR], HfXL[e * np + b], par,
                 &FdXL[((size_t)e * np + b) * NVAR]);
    sgn2d::fluxX(&xFaceR[((size_t)e * np + b) * NVAR], HfXR[e * np + b], par,
                 &FdXR[((size_t)e * np + b) * NVAR]);
  }
  for (int a = 0; a < np; ++a) {
    sgn2d::fluxY(&yFaceB[((size_t)e * np + a) * NVAR], HfYB[e * np + a], par,
                 &GdYB[((size_t)e * np + a) * NVAR]);
    sgn2d::fluxY(&yFaceT[((size_t)e * np + a) * NVAR], HfYT[e * np + a], par,
                 &GdYT[((size_t)e * np + a) * NVAR]);
  }
}

// --- pack the two x-boundary face-lines, one thread per (ley, t) ------------
//   Thread idx = ley*np + t (row ley, transverse node t). Copies the left face of
//   the leftmost owned column and the right face of the rightmost owned column
//   at that row/node into the contiguous send buffers.
//   The work is tiny (a handful of copies per thread), so its cost is
//   dominated by the launch itself and is almost independent of mesh size.
__global__ void kPack(const double* xFaceL, const double* xFaceR, int w,
                      int Ny, int np, double* sendL, double* sendR) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ny * np) return;
  const int ley = idx / np, t = idx - ley * np;   // row and transverse node
  const int eLm = 0 + w * ley;                    // leftmost owned element in row ley
  const int eRm = (w - 1) + w * ley;              // rightmost owned element in row ley
  for (int v = 0; v < NVAR; ++v) {
    sendL[((size_t)ley * np + t) * NVAR + v] =
        xFaceL[((size_t)eLm * np + t) * NVAR + v];
    sendR[((size_t)ley * np + t) * NVAR + v] =
        xFaceR[((size_t)eRm * np + t) * NVAR + v];
  }
}

// --- interface fluxes, per (element, transverse node); ghost-aware ----------
// Evaluates BOTH edges of every owned element (interior interfaces are done
// twice with identical arguments -> identical values; bitwise == CPU-MPI).
//   Thread idx = e*np + t. The same t serves as the y-node b for the x edges and
//   as the x-node a for the y edges (both run over 0..np-1).
//   x neighbours: owned column when available, otherwise the ghost line
//   received from the neighbouring rank. y neighbours: always local, periodic.
__global__ void kInterfaces(const double* xFaceL, const double* xFaceR,
                            const double* yFaceB, const double* yFaceT,
                            const double* ghostL, const double* ghostR,
                            const double* HfXL, const double* HfXR,
                            const double* HfYB, const double* HfYT,
                            sgn2d::Params par, int w, int Ny, int np, int neL,
                            double* FxL, double* FxR, double* GyB, double* GyT,
                            double* hSxL, double* hSxR, double* hSyB,
                            double* hSyT) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= neL * np) return;
  const int e = idx / np, t = idx - e * np;   // owned element and transverse node
  const int lx = e % w, ley = e / w;          // its local column and row

  // x-interfaces (t = transverse node b).
  {
    const double* uL = (lx > 0)
        ? &xFaceR[((size_t)(e - 1) * np + t) * NVAR]   // owned left neighbour
        : &ghostL[((size_t)ley * np + t) * NVAR];      // slab edge: ghost from left rank
    const double* uMe = &xFaceL[((size_t)e * np + t) * NVAR];
    const double* uMeR = &xFaceR[((size_t)e * np + t) * NVAR];
    const double* uR = (lx < w - 1)
        ? &xFaceL[((size_t)(e + 1) * np + t) * NVAR]   // owned right neighbour
        : &ghostR[((size_t)ley * np + t) * NVAR];      // slab edge: ghost from right rank
    sgn2d::rusanov(uL, uMe, 0, HfXL[e * np + t], par,
                   &FxL[((size_t)e * np + t) * NVAR]);
    sgn2d::rusanov(uMeR, uR, 0, HfXR[e * np + t], par,
                   &FxR[((size_t)e * np + t) * NVAR]);
    hSxL[e * np + t] = 0.5 * (uL[H] + uMe[H]);
    hSxR[e * np + t] = 0.5 * (uMeR[H] + uR[H]);
  }
  // y-interfaces (t = transverse node a); local periodic in y.
  {
    const int eB = lx + w * ((ley - 1 + Ny) % Ny);   // bottom neighbour
    const int eT = lx + w * ((ley + 1) % Ny);        // top neighbour
    const double* uB = &yFaceT[((size_t)eB * np + t) * NVAR];
    const double* uMe = &yFaceB[((size_t)e * np + t) * NVAR];
    const double* uMeT = &yFaceT[((size_t)e * np + t) * NVAR];
    const double* uT = &yFaceB[((size_t)eT * np + t) * NVAR];
    sgn2d::rusanov(uB, uMe, 1, HfYB[e * np + t], par,
                   &GyB[((size_t)e * np + t) * NVAR]);
    sgn2d::rusanov(uMeT, uT, 1, HfYT[e * np + t], par,
                   &GyT[((size_t)e * np + t) * NVAR]);
    hSyB[e * np + t] = 0.5 * (uB[H] + uMe[H]);
    hSyT[e * np + t] = 0.5 * (uMeT[H] + uT[H]);
  }
}

// --- volume update, one thread per node -------------------------------------
//   Thread g = owned node -> element e, local node (a, b). All interface data of
//   e is stored at e itself (both edges), so no neighbour index is needed here.
//   Mirrors step 3d of SGN2DMpi::rhs.
__global__ void kVolume(const double* U, const double* Fnod,
                        const double* Gnod, const double* xFaceL,
                        const double* xFaceR, const double* yFaceB,
                        const double* yFaceT, const double* FdXL,
                        const double* FdXR, const double* GdYB,
                        const double* GdYT, const double* FxL,
                        const double* FxR, const double* GyB,
                        const double* GyT, const double* hSxL,
                        const double* hSxR, const double* hSyB,
                        const double* hSyT, const double* D,
                        const double* gLp, const double* gRp,
                        const double* Hnod, const double* Hx,
                        const double* Hy, sgn2d::Params par, int np, int Nl,
                        double JxInv, double JyInv, double* dU) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= Nl) return;
  const int npp = np * np;
  const int e = g / npp;                    // element of node g
  const int loc = g - e * npp;              // local node index
  const int a = loc % np, b = loc / np;     // x- and y-node

  // FR-corrected divergence of F along x and G along y
  double dF[NVAR], dG[NVAR];
  for (int v = 0; v < NVAR; ++v) {
    double dfx = 0.0, dgy = 0.0;
    const double* Fv = Fnod + (size_t)v * Nl + (size_t)e * npp;
    const double* Gv = Gnod + (size_t)v * Nl + (size_t)e * npp;
    for (int k = 0; k < np; ++k) {
      dfx += D[a * np + k] * Fv[k + np * b];
      dgy += D[b * np + k] * Gv[a + np * k];
    }
    const double cx =
        (FxL[((size_t)e * np + b) * NVAR + v] -
         FdXL[((size_t)e * np + b) * NVAR + v]) * gLp[a] +
        (FxR[((size_t)e * np + b) * NVAR + v] -
         FdXR[((size_t)e * np + b) * NVAR + v]) * gRp[a];
    const double cy =
        (GyB[((size_t)e * np + a) * NVAR + v] -
         GdYB[((size_t)e * np + a) * NVAR + v]) * gLp[b] +
        (GyT[((size_t)e * np + a) * NVAR + v] -
         GdYT[((size_t)e * np + a) * NVAR + v]) * gRp[b];
    dF[v] = (dfx + cx) * JxInv;
    dG[v] = (dgy + cy) * JyInv;
  }

  // FR-corrected depth gradients with the central common depths
  double dhx = 0.0, dhy = 0.0;
  {
    const double* hv = U + (size_t)H * Nl + (size_t)e * npp;   // nodal h of element e
    for (int k = 0; k < np; ++k) {
      dhx += D[a * np + k] * hv[k + np * b];
      dhy += D[b * np + k] * hv[a + np * k];
    }
    dhx += (hSxL[e * np + b] -
            xFaceL[((size_t)e * np + b) * NVAR + H]) * gLp[a] +
           (hSxR[e * np + b] -
            xFaceR[((size_t)e * np + b) * NVAR + H]) * gRp[a];
    dhy += (hSyB[e * np + a] -
            yFaceB[((size_t)e * np + a) * NVAR + H]) * gLp[b] +
           (hSyT[e * np + a] -
            yFaceT[((size_t)e * np + a) * NVAR + H]) * gRp[b];
  }
  const double hx = dhx * JxInv, hy = dhy * JyInv;

  // pointwise NCPs and sources, then assemble dU
  double Un[NVAR], Bx[NVAR], By[NVAR], S[NVAR];
  for (int v = 0; v < NVAR; ++v) Un[v] = U[(size_t)v * Nl + g];
  sgn2d::ncpX(Un, hx, par, Bx);
  sgn2d::ncpY(Un, hy, par, By);
  sgn2d::source(Un, Hnod[g], Hx[g], Hy[g], par, S);

  for (int v = 0; v < NVAR; ++v)
    dU[(size_t)v * Nl + g] = -dF[v] - dG[v] - Bx[v] - By[v] + S[v];
}

// --- RK combos + error norm --------------------------------------------------
// One thread per state entry; identical to the single-GPU ports (see the
// detailed comments in sgn1d_cuda.cu).
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
  double* ssum = sh;                 // running sums
  double* smax = sh + blockDim.x;    // running maxima
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

// device allocation / upload helpers
double* devAlloc(size_t n) {
  double* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(double)));
  return p;
}
void devUpload(double* d, const double* h, size_t n) {
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(double), cudaMemcpyHostToDevice));
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor: bind this rank to a GPU, allocate, upload the host setup.
// ---------------------------------------------------------------------------
SGN2DCudaMpi::SGN2DCudaMpi(const SGN2DMpi& host, MPI_Comm comm)
    : P_(host.order()), np_(host.order() + 1),
      npp_((host.order() + 1) * (host.order() + 1)),
      w_(host.partition().w), Ny_(host.Ny()), neL_(host.nOwnedElem()),
      Nl_(host.nLocalDof()), Jx_(host.Jx()), Jy_(host.Jy()),
      par_(host.params()), comm_(comm), part_(host.partition()) {
  // One rank == one GPU: pick by node-local rank.
  //   MPI_Comm_split_type(SHARED) groups the ranks running on the same node;
  //   this rank's index inside that group (0..ranksPerNode-1) selects the
  //   device, so 4 ranks on a 4-GPU node use GPUs 0,1,2,3.
  {
    MPI_Comm shm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm);
    int lrank = 0;
    MPI_Comm_rank(shm, &lrank);
    MPI_Comm_free(&shm);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(lrank % ndev));   // all later CUDA calls use this GPU
  }

  // 1D FR operator tables
  const auto& op = host.ops();
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

  // owned bathymetry: nodal values, gradients, and all four face values
  d_Hnod = devAlloc(Nl_);
  d_Hx = devAlloc(Nl_);
  d_Hy = devAlloc(Nl_);
  devUpload(d_Hnod, host.stillDepthNodal().data(), Nl_);
  devUpload(d_Hx, host.stillDepthGradX().data(), Nl_);
  devUpload(d_Hy, host.stillDepthGradY().data(), Nl_);
  const size_t nf = (size_t)neL_ * np_;          // face nodes per face family
  d_HfXL = devAlloc(nf);
  d_HfXR = devAlloc(nf);
  d_HfYB = devAlloc(nf);
  d_HfYT = devAlloc(nf);
  devUpload(d_HfXL, host.faceXL().data(), nf);
  devUpload(d_HfXR, host.faceXR().data(), nf);
  devUpload(d_HfYB, host.faceYB().data(), nf);
  devUpload(d_HfYT, host.faceYT().data(), nf);

  // state and RK stages
  const size_t nTot = (size_t)NVAR * Nl_;
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
  const size_t nFace = nf * NVAR;
  d_xFaceL = devAlloc(nFace);
  d_xFaceR = devAlloc(nFace);
  d_yFaceB = devAlloc(nFace);
  d_yFaceT = devAlloc(nFace);
  d_FdXL = devAlloc(nFace);
  d_FdXR = devAlloc(nFace);
  d_GdYB = devAlloc(nFace);
  d_GdYT = devAlloc(nFace);
  d_FxL = devAlloc(nFace);
  d_FxR = devAlloc(nFace);
  d_GyB = devAlloc(nFace);
  d_GyT = devAlloc(nFace);
  d_hSxL = devAlloc(nf);
  d_hSxR = devAlloc(nf);
  d_hSyB = devAlloc(nf);
  d_hSyT = devAlloc(nf);

  // halo lines: one x face per row and transverse node, all variables
  const size_t lineLen = (size_t)Ny_ * np_ * NVAR;
  d_sendL = devAlloc(lineLen);
  d_sendR = devAlloc(lineLen);
  d_ghostL = devAlloc(lineLen);
  d_ghostR = devAlloc(lineLen);

  // error partials
  nBlocksErr_ = nBlocks((int)nTot);
  d_psum = devAlloc(nBlocksErr_);
  d_pmax = devAlloc(nBlocksErr_);
}

// Destructor: release every device allocation.
SGN2DCudaMpi::~SGN2DCudaMpi() {
  for (double* p : {d_D, d_lL, d_lR, d_gLp, d_gRp, d_Hnod, d_Hx, d_Hy,
                    d_HfXL, d_HfXR, d_HfYB, d_HfYT, d_U, d_k1, d_k2, d_k3,
                    d_k4, d_tmp, d_uNew, d_Fnod, d_Gnod, d_xFaceL, d_xFaceR,
                    d_yFaceB, d_yFaceT, d_FdXL, d_FdXR, d_GdYB, d_GdYT,
                    d_FxL, d_FxR, d_GyB, d_GyT, d_hSxL, d_hSxR, d_hSyB,
                    d_hSyT, d_sendL, d_sendR, d_ghostL, d_ghostR, d_psum,
                    d_pmax})
    if (p) cudaFree(p);
}

// owned state host -> device
void SGN2DCudaMpi::upload(const State& U) {
  devUpload(d_U, U.data(), (size_t)NVAR * Nl_);
}

// owned state device -> host (waits for queued kernels)
void SGN2DCudaMpi::download(State& U) const {
  U.resize((size_t)NVAR * Nl_);
  CUDA_CHECK(cudaMemcpy(U.data(), d_U, (size_t)NVAR * Nl_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// rhsDevice: distributed dU = R(U) (see file header for the phase list).
// ---------------------------------------------------------------------------
void SGN2DCudaMpi::rhsDevice(const double* U, double* dU) {
  const size_t lineLen = (size_t)Ny_ * np_ * NVAR;   // doubles per halo line

  // 1. nodal fluxes
  kNodalFlux<<<nBlocks(Nl_), kThreads>>>(U, d_Hnod, par_, Nl_, d_Fnod, d_Gnod);
  // 2. face states + element face fluxes
  kFaces<<<nBlocks(neL_), kThreads>>>(U, d_lL, d_lR, d_HfXL, d_HfXR, d_HfYB,
                                      d_HfYT, par_, neL_, np_, Nl_, d_xFaceL,
                                      d_xFaceR, d_yFaceB, d_yFaceT, d_FdXL,
                                      d_FdXR, d_GdYB, d_GdYT);
  // 3. pack the boundary lines
  kPack<<<nBlocks(Ny_ * np_), kThreads>>>(d_xFaceL, d_xFaceR, w_, Ny_, np_,
                                          d_sendL, d_sendR);
  // Launches are asynchronous: wait until the pack kernel has written the send
  // buffers before MPI reads them from device memory.
  CUDA_CHECK(cudaDeviceSynchronize()); // packs complete before MPI touches them

  if (!part_.hasGhosts) {
    // Single rank: the ring wraps to itself, so the ghosts are this rank's own
    // opposite boundary lines (device-to-device copy, no MPI).
    CUDA_CHECK(cudaMemcpy(d_ghostL, d_sendR, lineLen * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_ghostR, d_sendL, lineLen * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  } else {
    // CUDA-aware MPI straight from device pointers.
    //   Exchange 1 (tag 0): send my rightmost column's right faces to the RIGHT
    //   rank, receive the left rank's rightmost faces into ghostL.
    MPI_Sendrecv(d_sendR, (int)lineLen, MPI_DOUBLE, part_.rightRank, 0,
                 d_ghostL, (int)lineLen, MPI_DOUBLE, part_.leftRank, 0, comm_,
                 MPI_STATUS_IGNORE);
    //   Exchange 2 (tag 1): send my leftmost column's left faces to the LEFT
    //   rank, receive the right rank's leftmost faces into ghostR.
    MPI_Sendrecv(d_sendL, (int)lineLen, MPI_DOUBLE, part_.leftRank, 1,
                 d_ghostR, (int)lineLen, MPI_DOUBLE, part_.rightRank, 1, comm_,
                 MPI_STATUS_IGNORE);
  }

  // 4. interface fluxes and central depths (reads the ghost lines in place)
  kInterfaces<<<nBlocks(neL_ * np_), kThreads>>>(
      d_xFaceL, d_xFaceR, d_yFaceB, d_yFaceT, d_ghostL, d_ghostR, d_HfXL,
      d_HfXR, d_HfYB, d_HfYT, par_, w_, Ny_, np_, neL_, d_FxL, d_FxR, d_GyB,
      d_GyT, d_hSxL, d_hSxR, d_hSyB, d_hSyT);
  // 5. assemble dU
  kVolume<<<nBlocks(Nl_), kThreads>>>(
      U, d_Fnod, d_Gnod, d_xFaceL, d_xFaceR, d_yFaceB, d_yFaceT, d_FdXL,
      d_FdXR, d_GdYB, d_GdYT, d_FxL, d_FxR, d_GyB, d_GyT, d_hSxL, d_hSxR,
      d_hSyB, d_hSyT, d_D, d_gLp, d_gRp, d_Hnod, d_Hx, d_Hy, par_, np_, Nl_,
      1.0 / Jx_, 1.0 / Jy_, dU);
  CUDA_CHECK(cudaGetLastError());   // report a failed launch
}

// One distributed RHS of the uploaded state, downloaded (parity tests).
void SGN2DCudaMpi::rhsOnce(State& dU) {
  rhsDevice(d_U, d_k1);
  dU.resize((size_t)NVAR * Nl_);
  CUDA_CHECK(cudaMemcpy(dU.data(), d_k1, (size_t)NVAR * Nl_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// integrateAdaptive: device-resident BS3(2) with a global error norm.
// Per attempted step on every rank:
//   - stages and candidate on the GPU (three distributed RHS evaluations)
//   - kErrNorm: per-block partials on the GPU; copy the nb partials to the host
//   - host: sum the blocks -> local (sum, max); two MPI_Allreduce -> global norm
//   - accept/reject and PI update exactly as in rk.cpp
// ---------------------------------------------------------------------------
AdaptiveRKStats SGN2DCudaMpi::integrateAdaptive(double t0, double tEnd,
                                                double dt0,
                                                const AdaptiveRKOptions& opt) {
  const int orderLow = 2;
  const double expo = 1.0 / (orderLow + 1.0);
  const double alpha = 0.7 * expo;   // PI exponents
  const double beta = 0.4 * expo;

  AdaptiveRKStats st;
  st.tReached = t0;
  const int n = NVAR * Nl_;                               // owned state size
  const int nb = nBlocks(n);                              // blocks per launch
  const size_t shBytes = 2 * kThreads * sizeof(double);   // kErrNorm shared scratch
  std::vector<double> psum(nBlocksErr_), pmax(nBlocksErr_);

  // global number of degrees of freedom (RMS normalisation)
  long nGlobal = 0;
  long nl = (long)n;
  MPI_Allreduce(&nl, &nGlobal, 1, MPI_LONG, MPI_SUM, comm_);

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

    // per-block partials on device, copied to the host
    kErrNorm<<<nb, kThreads, shBytes>>>(d_uNew, d_U, d_k1, d_k2, d_k3, d_k4,
                                        dt, opt.atol, opt.rtol, n, d_psum,
                                        d_pmax);
    CUDA_CHECK(cudaMemcpy(psum.data(), d_psum, nb * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pmax.data(), d_pmax, nb * sizeof(double),
                          cudaMemcpyDeviceToHost));
    // this rank's contribution
    double localSum = 0.0, localMax = 0.0;
    for (int bb = 0; bb < nb; ++bb) {
      localSum += psum[bb];
      localMax = std::max(localMax, pmax[bb]);
    }
    // global norm: SUM (order-dependent in floating point) and MAX (exact)
    double part[2] = {localSum, localMax}, glob[2] = {0.0, 0.0};
    MPI_Allreduce(&part[0], &glob[0], 1, MPI_DOUBLE, MPI_SUM, comm_);
    MPI_Allreduce(&part[1], &glob[1], 1, MPI_DOUBLE, MPI_MAX, comm_);
    const double errL2 = std::sqrt(glob[0] / (double)nGlobal);
    const double errNorm = std::max(errL2, 1e-300);

    if (errNorm <= 1.0) {             // accept: pointer swaps, no copies
      t += dt;
      std::swap(d_U, d_uNew);
      std::swap(d_k1, d_k4);          // FSAL
      ++st.accepted;
      st.tReached = t;
      st.dtLast = dt;
      st.maxNormLast = glob[1];
      errPrev = errNorm;
    } else {
      ++st.rejected;
    }

    // PI controller (identical on every rank, since errNorm is global)
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
