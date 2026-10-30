#ifndef SGN_SLAB1D_H
#define SGN_SLAB1D_H

// =============================================================================
// slab1d.h -- 1D slab decomposition of the periodic element mesh
// =============================================================================
//
// WHAT THIS FILE IS
//   The index bookkeeping of the domain decomposition used by every distributed
//   solver (1D MPI CPU, 2D MPI CPU, 2D multi-GPU). The global mesh is cut along
//   x into contiguous "slabs": each MPI rank (one process, and in the GPU solvers
//   one GPU) owns the element range [a, b). Ranks form a periodic ring, so every
//   rank talks only to its left and right neighbour.
//
//   Pure host C++ (no MPI, no CUDA), so this -- the most off-by-one-prone part of
//   the parallel port -- is unit-testable in ONE process before any MPI is
//   involved (test/test_slab1d.cpp). Same philosophy as the NSK-FR-CUDA slab
//   partition this reuses. In 2D the same partition is applied to element
//   COLUMNS (nElem = Nx), each column spanning all Ny rows.
//
// CONVENTIONS
//   Local element order matches the global order on the owned range:
//     owned local e in [0, w)  <->  global element (a + e)
//   Halo: each rank needs the face state of ONE neighbour element on each side
//   (the element just left of a and the element at b). When size == 1 the ring
//   wraps to itself: no ghosts and no messages, so a single-rank run follows
//   exactly the serial code path (bitwise identical to the serial solver).
// =============================================================================

#include <vector>

namespace sgn {

struct SlabPartition1D {
  int nElem = 0;            // global element count (element columns in 2D)
  int rank = 0, size = 1;   // this rank's position in the ring / number of ranks
  int a = 0, b = 0, w = 0;  // owned element range [a, b), width w = b - a
  int leftRank = 0, rightRank = 0; // ring neighbours (periodic)
  bool hasGhosts = false;   // false iff size==1 (no exchange needed)

  // Global element id of the left/right neighbour of the owned block (the
  // elements whose face traces are exchanged). Valid whether or not hasGhosts
  // (they wrap to own elements when size==1).
  int leftNbrGlobal = 0;    // = (a - 1 + nElem) % nElem
  int rightNbrGlobal = 0;   // = b % nElem

  // owned local element index -> global element index
  int ownedGlobal(int e) const { return a + e; }
};

// Split nElem elements over `size` ranks as evenly as possible (the first
// (nElem % size) ranks get one extra element), and fill in the neighbours of
// rank `rank`. Every rank calls this with its own rank and obtains a consistent,
// non-overlapping partition without any communication.
inline SlabPartition1D makeSlabPartition1D(int nElem, int rank, int size) {
  SlabPartition1D p;
  p.nElem = nElem; p.rank = rank; p.size = size;

  // base width and number of ranks that receive one extra element
  const int base = nElem / size, rem = nElem % size;
  // start of this rank's range: all previous ranks' widths
  p.a = rank * base + (rank < rem ? rank : rem);
  p.w = base + (rank < rem ? 1 : 0);
  p.b = p.a + p.w;
  p.hasGhosts = (size > 1);
  // periodic ring
  p.leftRank = (rank - 1 + size) % size;
  p.rightRank = (rank + 1) % size;
  p.leftNbrGlobal = (p.a - 1 + nElem) % nElem;
  p.rightNbrGlobal = p.b % nElem;
  return p;
}

} // namespace sgn

#endif // SGN_SLAB1D_H
