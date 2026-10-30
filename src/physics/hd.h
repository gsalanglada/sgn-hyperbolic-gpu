#ifndef SGN_HD_H
#define SGN_HD_H

// =============================================================================
// hd.h -- host/device portability marker
// =============================================================================
//
// Physics functions tagged SGN_HD compile both as ordinary host C++ (the CPU
// reference solver, used as validation oracle) and as CUDA device code (the GPU
// kernels), from the SAME source text. The model physics (fluxes, eigenvalues,
// sources, Riemann solver) therefore lives in exactly one place, and the CPU
// and GPU solvers cannot drift apart: this is the first of the three
// reproducibility requirements described in the paper.
//
// How it works:
//   - When the file is compiled by nvcc, __CUDACC__ is defined and SGN_HD
//     expands to  __host__ __device__ : the function is emitted twice, once
//     callable from CPU code and once callable from inside a GPU kernel.
//   - When compiled by an ordinary C++ compiler (g++), SGN_HD expands to
//     nothing and the function is a plain inline C++ function.
//
// (Same discipline as the NSK-FR-CUDA engine this project reuses.)
//
// Rules for SGN_HD functions (anything else will not compile as device code):
//   - operate on plain scalars / fixed-size arrays (no std::vector, no STL
//     containers, no exceptions, no I/O),
//   - use std::sqrt / std::fabs / std::fmax etc. (supported in CUDA device code).
// =============================================================================

#if defined(__CUDACC__)
  #define SGN_HD __host__ __device__
#else
  #define SGN_HD
#endif

#endif // SGN_HD_H
