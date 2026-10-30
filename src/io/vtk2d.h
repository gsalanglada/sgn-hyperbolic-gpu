#ifndef SGN_VTK2D_H
#define SGN_VTK2D_H

// =============================================================================
// vtk2d.h -- VTK output of the serial 2D solver (visualization only)
// =============================================================================
//
// Minimal legacy-VTK output for 2D fields (host-only, single process).
//
// Writes ELEMENT-MEAN values (quadrature average of the GLL nodes) on the
// uniform Nx x Ny element grid as STRUCTURED_POINTS — small files that
// ParaView animates directly (open the file series, then Warp By Scalar on
// "eta" for the 3D surface). Fields: eta = h - H, u, v, p, and H (bathymetry,
// static but repeated per frame for convenience).
//
// Output only: nothing here feeds back into the simulation. The distributed
// solver uses vtk2d_mpi.h instead.
// =============================================================================

#include "../solver/sgn2d.h"

#include <cstdio>
#include <string>
#include <vector>

namespace sgn {

// Write one VTK frame of state U of solver `sys` to `path`.
inline void writeVtk2D(const SGN2D& sys, const State& U,
                       const std::string& path) {
  const auto& mesh = sys.mesh();
  const int Nx = mesh.Nx, Ny = mesh.Ny;                 // element grid
  const int P = sys.order(), np = P + 1, npp = np * np; // nodes per direction / element
  const int Nd = sys.nDof();                            // nodes per variable
  const auto& qw = sys.ops().weights();                 // 1D quadrature weights
  const auto& Hn = sys.stillDepthNodal();               // nodal bathymetry

  // Element means (quadrature weights sum to 2 per direction -> divide by 4).
  auto elemMean = [&](const double* f, int e) {
    double s = 0.0;
    for (int b = 0; b < np; ++b)
      for (int a = 0; a < np; ++a) s += qw[a] * qw[b] * f[e * npp + a + np * b];
    return 0.25 * s;
  };

  // VTK header: one point per element, placed at the element centre
  std::FILE* fp = std::fopen(path.c_str(), "w");
  if (!fp) return;                       // silently skip the frame if unwritable
  std::fprintf(fp, "# vtk DataFile Version 3.0\nSGN2D element means\nASCII\n");
  std::fprintf(fp, "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n", Nx, Ny);
  std::fprintf(fp, "ORIGIN %.6f %.6f 0\nSPACING %.6f %.6f 1\n",
               0.5 * mesh.hx, 0.5 * mesh.hy, mesh.hx, mesh.hy);
  std::fprintf(fp, "POINT_DATA %d\n", Nx * Ny);

  // work: scratch nodal field; writeField prints the element means of a nodal
  // field row by row (ey outer, ex inner), the order VTK expects
  std::vector<double> work(Nd);
  auto writeField = [&](const char* name, const std::vector<double>& f) {
    std::fprintf(fp, "SCALARS %s float 1\nLOOKUP_TABLE default\n", name);
    for (int ey = 0; ey < Ny; ++ey) {
      for (int ex = 0; ex < Nx; ++ex)
        std::fprintf(fp, "%.5e ", elemMean(f.data(), mesh.elem(ex, ey)));
      std::fprintf(fp, "\n");
    }
  };

  // eta = h - H
  for (int i = 0; i < Nd; ++i) work[i] = U[(size_t)sgn2d::H * Nd + i] - Hn[i];
  writeField("eta", work);
  // u, v, p (primitives)
  for (int i = 0; i < Nd; ++i)
    work[i] = U[(size_t)sgn2d::HU * Nd + i] / U[(size_t)sgn2d::H * Nd + i];
  writeField("u", work);
  for (int i = 0; i < Nd; ++i)
    work[i] = U[(size_t)sgn2d::HV * Nd + i] / U[(size_t)sgn2d::H * Nd + i];
  writeField("v", work);
  for (int i = 0; i < Nd; ++i)
    work[i] = U[(size_t)sgn2d::HP * Nd + i] / U[(size_t)sgn2d::H * Nd + i];
  writeField("p", work);
  // bathymetry (as depth H; render as -H for the bottom surface)
  writeField("H", Hn);

  std::fclose(fp);
}

} // namespace sgn

#endif // SGN_VTK2D_H
