#ifndef SGN_VTK2D_MPI_H
#define SGN_VTK2D_MPI_H

// =============================================================================
// vtk2d_mpi.h -- VTK output of the distributed 2D solver (visualization only)
// =============================================================================
//
// Legacy-VTK output for the DISTRIBUTED 2D solver (visualization only).
//
// COLLECTIVE: writeVtk2DMpi uses MPI_Gather / MPI_Gatherv, so EVERY rank must
// call it at the same point of the run (each passes its own owned state); only
// rank 0 writes the file. With the GPU solver, download() the state first.
//
// Each rank reduces its owned slab to one value per element (the quadrature
// mean of the GLL nodes), the element field is gathered on rank 0, coarsened by
// block averaging to a bounded raster, and written as a single STRUCTURED_POINTS
// file. ParaView opens the file series directly; "Warp By Scalar" on eta gives
// the animated free surface.
//
// The coarsening matters. At production resolution the element grid alone is
// 4000x2000 cells, which is far more than any figure or screen resolves and
// would make every frame hundreds of megabytes. Block averaging to at most
// `maxCells` per direction keeps a frame at a few megabytes while preserving
// everything a picture can show. The simulation itself is unaffected: this is
// output only, and the coarsening happens after the physics.
//
// Fields: eta = h - H (free-surface elevation), speed = |(u,v)|, and H.
// =============================================================================

#include "../parallel/sgn2d_mpi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <mpi.h>

namespace sgn {

//   sys       host-side distributed solver (partition, operators, bathymetry)
//   U         this rank's OWNED state (host memory)
//   path      output file (written by rank 0 only)
//   maxCells  upper bound of the written raster per direction
inline void writeVtk2DMpi(const SGN2DMpi& sys, const State& U,
                          const std::string& path, MPI_Comm comm,
                          int maxCells = 800) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const auto& part = sys.partition();
  const int P = sys.order(), np = P + 1, npp = np * np;    // nodes per direction / element
  const int Ny = sys.Ny(), NxG = part.nElem, w = part.w;   // rows, global columns, owned columns
  const int Nd = sys.nLocalDof();                          // owned nodes per variable
  const auto& qw = sys.ops().weights();                    // 1D quadrature weights
  const auto& Hn = sys.stillDepthNodal();                  // owned nodal bathymetry

  auto elemMean = [&](const std::vector<double>& f, int e) {
    double s = 0.0;
    for (int b = 0; b < np; ++b)
      for (int a = 0; a < np; ++a)
        s += qw[a] * qw[b] * f[(size_t)e * npp + a + np * b];
    return 0.25 * s;   // weights sum to 2 per direction
  };

  // --- local element means -------------------------------------------------
  const int nLocElem = w * Ny;
  std::vector<double> work(Nd), locEta(nLocElem), locSpd(nLocElem),
      locH(nLocElem);

  for (int i = 0; i < Nd; ++i)
    work[i] = U[(size_t)sgn2d::H * Nd + i] - Hn[i];
  for (int e = 0; e < nLocElem; ++e) locEta[e] = elemMean(work, e);

  for (int i = 0; i < Nd; ++i) {
    const double hh = U[(size_t)sgn2d::H * Nd + i];
    const double uu = U[(size_t)sgn2d::HU * Nd + i] / hh;
    const double vv = U[(size_t)sgn2d::HV * Nd + i] / hh;
    work[i] = std::sqrt(uu * uu + vv * vv);
  }
  for (int e = 0; e < nLocElem; ++e) locSpd[e] = elemMean(work, e);
  for (int e = 0; e < nLocElem; ++e) locH[e] = elemMean(Hn, e);

  // --- gather the element field on rank 0 ----------------------------------
  // First every rank's slab start a and width w (MPI_Gather), so rank 0 knows
  // how many elements each rank sends (cnt) and where they go (disp); then the
  // variable-size element arrays themselves (MPI_Gatherv).
  std::vector<int> aAll(size), wAll(size), cnt(size), disp(size);
  MPI_Gather(&part.a, 1, MPI_INT, aAll.data(), 1, MPI_INT, 0, comm);
  MPI_Gather(&part.w, 1, MPI_INT, wAll.data(), 1, MPI_INT, 0, comm);
  if (rank == 0) {
    int off = 0;
    for (int r = 0; r < size; ++r) {
      cnt[r] = wAll[r] * Ny;
      disp[r] = off;
      off += cnt[r];
    }
  }

  std::vector<double> gEta, gSpd, gH;   // rank-ordered blocks
  if (rank == 0) {
    gEta.resize((size_t)NxG * Ny);
    gSpd.resize((size_t)NxG * Ny);
    gH.resize((size_t)NxG * Ny);
  }
  MPI_Gatherv(locEta.data(), nLocElem, MPI_DOUBLE, gEta.data(), cnt.data(),
              disp.data(), MPI_DOUBLE, 0, comm);
  MPI_Gatherv(locSpd.data(), nLocElem, MPI_DOUBLE, gSpd.data(), cnt.data(),
              disp.data(), MPI_DOUBLE, 0, comm);
  MPI_Gatherv(locH.data(), nLocElem, MPI_DOUBLE, gH.data(), cnt.data(),
              disp.data(), MPI_DOUBLE, 0, comm);
  if (rank != 0) return;

  // --- reassemble rank blocks into a global (ex, ey) raster ----------------
  // Within a rank the owned elements are stored with the y index outermost:
  // e = ley * w + lx, and the global column is part.a + lx.
  std::vector<double> fEta((size_t)NxG * Ny), fSpd((size_t)NxG * Ny),
      fH((size_t)NxG * Ny);
  for (int r = 0; r < size; ++r)
    for (int ley = 0; ley < Ny; ++ley)
      for (int lx = 0; lx < wAll[r]; ++lx) {
        const size_t src = (size_t)disp[r] + (size_t)ley * wAll[r] + lx;
        const size_t dst = (size_t)ley * NxG + (aAll[r] + lx);
        fEta[dst] = gEta[src];
        fSpd[dst] = gSpd[src];
        fH[dst] = gH[src];
      }

  // --- block-average down to a bounded raster ------------------------------
  const int sx = std::max(1, (NxG + maxCells - 1) / maxCells);
  const int sy = std::max(1, (Ny + maxCells - 1) / maxCells);
  const int Cx = (NxG + sx - 1) / sx, Cy = (Ny + sy - 1) / sy;

  auto coarsen = [&](const std::vector<double>& f) {
    std::vector<double> c((size_t)Cx * Cy, 0.0);
    std::vector<int> n((size_t)Cx * Cy, 0);
    for (int ey = 0; ey < Ny; ++ey)
      for (int ex = 0; ex < NxG; ++ex) {
        const size_t k = (size_t)(ey / sy) * Cx + (ex / sx);
        c[k] += f[(size_t)ey * NxG + ex];
        ++n[k];
      }
    for (size_t k = 0; k < c.size(); ++k) if (n[k]) c[k] /= n[k];
    return c;
  };
  const auto cEta = coarsen(fEta);
  const auto cSpd = coarsen(fSpd);
  const auto cH = coarsen(fH);

  // --- write ---------------------------------------------------------------
  const double hx = 2.0 * sys.Jx(), hy = 2.0 * sys.Jy();  // element sizes
  std::FILE* fp = std::fopen(path.c_str(), "w");
  if (!fp) return;
  std::fprintf(fp, "# vtk DataFile Version 3.0\nSGN2D distributed element means"
                   "\nASCII\n");
  std::fprintf(fp, "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n", Cx, Cy);
  std::fprintf(fp, "ORIGIN %.6f %.6f 0\nSPACING %.6f %.6f 1\n",
               0.5 * hx * sx, 0.5 * hy * sy, hx * sx, hy * sy);
  std::fprintf(fp, "POINT_DATA %d\n", Cx * Cy);

  auto field = [&](const char* name, const std::vector<double>& c) {
    std::fprintf(fp, "SCALARS %s float 1\nLOOKUP_TABLE default\n", name);
    for (int j = 0; j < Cy; ++j) {
      for (int i = 0; i < Cx; ++i)
        std::fprintf(fp, "%.4e ", c[(size_t)j * Cx + i]);
      std::fprintf(fp, "\n");
    }
  };
  field("eta", cEta);
  field("speed", cSpd);
  field("H", cH);
  std::fclose(fp);
}

} // namespace sgn

#endif // SGN_VTK2D_MPI_H
