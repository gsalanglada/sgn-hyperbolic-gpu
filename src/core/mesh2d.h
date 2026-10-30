#ifndef SGN_MESH2D_H
#define SGN_MESH2D_H

// =============================================================================
// mesh2d.h -- structured periodic 2D mesh
// =============================================================================
//
// Cartesian, periodic, axis-aligned quad mesh on [0,Lx] x [0,Ly] with Nx x Ny
// identical rectangular elements. Elements are affine (constant Jacobian), which
// isolates the 2D multi-element FR coupling from curved-metric complexity.
// (Same convention as the NSK engine.)
//
// Indexing:
//   Element id:        e = ex + Nx*ey      (ex in [0,Nx) column, ey in [0,Ny) row)
//   Face numbering:    0 = eta=-1 (bottom), 1 = xi=+1 (right),
//                      2 = eta=+1 (top),    3 = xi=-1 (left)
//   Periodicity: the neighbour across a domain edge wraps around to the element
//   on the opposite edge.
// =============================================================================

namespace sgn {

struct CartesianMesh2D {
  int Nx, Ny;             // number of elements in x and y
  double Lx, Ly, hx, hy;  // domain size Lx x Ly [m]; element size hx = Lx/Nx, hy = Ly/Ny [m]

  CartesianMesh2D(int Nx_, int Ny_, double Lx_, double Ly_)
      : Nx(Nx_), Ny(Ny_), Lx(Lx_), Ly(Ly_), hx(Lx_ / Nx_), hy(Ly_ / Ny_) {}

  int nElem() const { return Nx * Ny; }            // total number of elements
  int elem(int ex, int ey) const { return ex + Nx * ey; } // (column,row) -> id
  int ex(int e) const { return e % Nx; }           // id -> column
  int ey(int e) const { return e / Nx; }           // id -> row

  // Id of the element sharing face `face` of element e (periodic wrap-around).
  int neighbor(int e, int face) const {
    const int i = ex(e), j = ey(e);
    switch (face) {
      case 0: return elem(i, (j - 1 + Ny) % Ny); // bottom
      case 1: return elem((i + 1) % Nx, j);      // right
      case 2: return elem(i, (j + 1) % Ny);      // top
      case 3: return elem((i - 1 + Nx) % Nx, j); // left
      default: return -1;
    }
  }

  // Affine Jacobians dx/dxi and dy/deta: half the element size. A reference
  // derivative is divided by these to obtain the physical derivative.
  double Jx() const { return 0.5 * hx; }
  double Jy() const { return 0.5 * hy; }
};

} // namespace sgn

#endif // SGN_MESH2D_H
