// =============================================================================
// test_slab1d.cpp -- slab-partition index bookkeeping (src/parallel/slab1d.h)
// =============================================================================
//
// HOW TO RUN   make run-test   (host CPU, no MPI; builds and runs all 12 host tests)
// PASS         every check prints "ok"; the program prints "ALL TESTS PASSED" and
//              exits with code 0. Any failed check prints "FAIL" and the exit code is 1.
// =============================================================================
//
// Unit tests for the 1D slab partition bookkeeping (src/parallel/slab1d.h).
// Pure host, single process -- validates the index arithmetic before any MPI.
//   1) Owned ranges over all ranks tile [0, nElem) exactly once (a partition).
//   2) Neighbour globals are the correct periodic wrap of the owned block.
//   3) Ring neighbour ranks are consistent (my right's left is me, etc).
//   4) size==1 -> no ghosts, neighbours wrap to self (bitwise-serial case).
//   5) Even split invariant: widths differ by at most 1; sum == nElem.
//
// Build: part of `make run-test` (no MPI needed).

#include "../src/parallel/slab1d.h"

#include <cstdio>
#include <vector>

// Test harness: check() prints "ok" or "FAIL", the description and the
// measured value, and counts failures; main() returns non-zero if any failed.
static int g_failures = 0;
static void check(bool ok, const char* what, long v) {
  std::printf("  %s: %-56s  %ld\n", ok ? "ok  " : "FAIL", what, v);
  if (!ok) ++g_failures;
}

int main() {
  std::printf("=== 1D slab partition ===\n\n");

  for (int nElem : {1, 2, 7, 16, 33, 100}) {
    for (int size : {1, 2, 3, 4, 5, 8}) {
      if (size > nElem) continue;
      std::vector<int> cover(nElem, 0);
      int wmin = nElem + 1, wmax = -1, wsum = 0;
      bool nbrOk = true, ringOk = true, ghostOk = true;

      for (int rank = 0; rank < size; ++rank) {
        const auto p = sgn::makeSlabPartition1D(nElem, rank, size);
        for (int e = 0; e < p.w; ++e) {
          const int gid = p.ownedGlobal(e);
          if (gid >= 0 && gid < nElem) cover[gid]++;
        }
        wmin = std::min(wmin, p.w);
        wmax = std::max(wmax, p.w);
        wsum += p.w;

        // Neighbour globals: periodic wrap of the owned block edges.
        if (p.leftNbrGlobal != (p.a - 1 + nElem) % nElem) nbrOk = false;
        if (p.rightNbrGlobal != p.b % nElem) nbrOk = false;

        // Ring consistency: my rightRank's leftRank == me.
        const auto pr = sgn::makeSlabPartition1D(nElem, p.rightRank, size);
        const auto pl = sgn::makeSlabPartition1D(nElem, p.leftRank, size);
        if (pr.leftRank != rank) ringOk = false;
        if (pl.rightRank != rank) ringOk = false;
        // Neighbour's owned block abuts mine (periodically).
        if (size > 1) {
          if (pr.a != p.b % nElem && !(p.b == nElem && pr.a == 0)) ringOk = false;
        }

        if (size == 1 && p.hasGhosts) ghostOk = false;
      }

      char buf[80];
      std::snprintf(buf, sizeof buf, "n=%d size=%d: tiles [0,n) once", nElem, size);
      bool tiled = true;
      for (int c : cover) if (c != 1) tiled = false;
      check(tiled, buf, 0);

      std::snprintf(buf, sizeof buf, "n=%d size=%d: neighbour globals correct", nElem, size);
      check(nbrOk, buf, 0);

      std::snprintf(buf, sizeof buf, "n=%d size=%d: ring consistent", nElem, size);
      check(ringOk, buf, 0);

      std::snprintf(buf, sizeof buf, "n=%d size=%d: widths balanced & sum==n", nElem, size);
      check(wsum == nElem && (wmax - wmin) <= 1, buf, wsum);

      std::snprintf(buf, sizeof buf, "n=%d size=%d: size1 => no ghosts", nElem, size);
      check(ghostOk, buf, 0);
    }
  }

  // Explicit size==1 self-wrap case.
  {
    const auto p = sgn::makeSlabPartition1D(16, 0, 1);
    bool ok = !p.hasGhosts && p.w == 16 && p.a == 0 && p.b == 16 &&
              p.leftRank == 0 && p.rightRank == 0 &&
              p.leftNbrGlobal == 15 && p.rightNbrGlobal == 0;
    check(ok, "size==1: full mesh, neighbours wrap to self", ok ? 1 : 0);
  }

  std::printf("\n");
  if (g_failures == 0) { std::printf("ALL TESTS PASSED\n"); return 0; }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
