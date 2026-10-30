# sgn-hyperbolic-gpu

High-order, reproducible, multi-GPU solver for the hyperbolic relaxation of the
Serre–Green–Naghdi (SGN) water-wave equations.

This repository contains the source code, verification tests, benchmark and job
scripts, post-processing, digitized laboratory data and reference results needed
to reproduce every table and figure of the manuscript

> G. Sal-Anglada and O. Lehmkuhl, *A reproducible distributed multi-GPU
> flux-reconstruction solver for hyperbolic Serre–Green–Naghdi water waves*,
> Barcelona Supercomputing Center (BSC-CNS).

**In one paragraph.** The dispersive SGN equations normally require a global
elliptic solve at every time step. The hyperbolic relaxation of Escalante,
Dumbser and Castro (J. Comput. Phys. 394, 2019) replaces it by an extra evolution
equation, so the whole model becomes a set of local balance laws. This code
discretizes that system with high-order flux reconstruction (FR/CPR, degree
`P = 4` in production) and integrates it with an adaptive explicit Runge–Kutta
method. The same physics header is compiled for the CPU reference solver and for
the CUDA kernels, and the multi-GPU version exchanges element faces with
CUDA-aware MPI. The CPU, MPI, single-GPU and multi-GPU versions reproduce each
other bit for bit on the right-hand side.

Everything is self-contained: no external mesh generator, no proprietary
library, no hidden configuration. Every source file carries detailed English
comments explaining each step, variable and, in the CUDA files, each thread and
kernel.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [What you need, and for what](#2-what-you-need-and-for-what)
3. [Getting started on a normal computer (CPU only)](#3-getting-started-on-a-normal-computer-cpu-only)
4. [Adding MPI](#4-adding-mpi)
5. [Adding an NVIDIA GPU](#5-adding-an-nvidia-gpu)
6. [Multi-GPU runs (workstation or cluster)](#6-multi-gpu-runs-workstation-or-cluster)
7. [Verification tests](#7-verification-tests)
8. [Reproducing each paper result](#8-reproducing-each-paper-result)
9. [Figures](#9-figures)
10. [Benchmarks and profiling](#10-benchmarks-and-profiling)
11. [Expected runtimes](#11-expected-runtimes)
12. [Troubleshooting](#12-troubleshooting)
13. [Notes on the bitwise-reproducibility claims](#13-notes-on-the-bitwise-reproducibility-claims)
14. [Citation and licence](#14-citation-and-licence)

---

## 1. Repository layout

```
├── src/            solver (headers + implementation)
│   ├── physics/    fluxes, eigenvalues, sources, Riemann solver — ONE header
│   │               compiled for both CPU and GPU (the SGN_HD macro, hd.h)
│   ├── core/       FR operators (GLL nodes, differentiation, corrections), 2D mesh
│   ├── solver/     1D/2D CPU reference solvers, Runge–Kutta integrators
│   ├── parallel/   CPU MPI slab decomposition
│   ├── cuda/       single-GPU and multi-GPU (CUDA-aware MPI) solvers
│   └── io/         VTK output for visualization
├── apps/           one driver per paper result (see §8)
├── test/           17 self-checking verification tests (see §7)
├── scripts/        job scripts, post-processing, figure generation, packaging
├── data/           digitized laboratory data (Beji–Battjes / Dingemans 1994)
├── results/        reference output behind every table and figure
├── Makefile        every build, test and driver target (map in its header)
├── setup_env.src   module environment used on MareNostrum 5
├── MANIFEST.md     file inventory, toolchain, parameters, provenance
├── CITATION.cff    how to cite this software
└── LICENSE
```

Good entry points for reading the code:

1. `src/physics/sgn_flux.h` — the whole model, pointwise, in one file.
2. `src/solver/sgn1d.cpp`, function `rhs()` — the FR scheme, phase by phase.
3. `src/solver/rk.cpp` — the adaptive time integrator.
4. `src/cuda/sgn1d_cuda.cu` — its header is a from-scratch introduction to CUDA
   (kernels, threads, blocks, memory) using this code as the example.
5. `src/cuda/sgn2d_cuda_mpi.cu` — the production multi-GPU solver: the five
   right-hand-side kernels and the halo exchange.
6. `src/parallel/sgn1d_mpi.h` — an introduction to the MPI concepts used.

Each file in `apps/` and `test/` starts with a header giving its usage, its call
chain through the solver and its outputs.

---

## 2. What you need, and for what

The code is layered, so you only need the tools for the part you want to run.

| To run… | You need | Make targets |
|---|---|---|
| Host verification tests, all 1D validation cases, convergence studies | a C++17 compiler and GNU `make` | `run-test`, `beji`, `bolles`, `alpha`, `valfigs`, `rkcompare`, `convergence`, `ncptradeoff` |
| CPU MPI reproducibility tests | + an MPI implementation (Open MPI or MPICH) | `run-mpi` |
| Single-GPU tests and the shoal animation | + an NVIDIA GPU and the CUDA toolkit | `run-cuda`, `showcase` |
| Multi-GPU tests and production drivers | + a **CUDA-aware** MPI | `run-cuda-mpi`, `bench`, `shoalfocus`, `tsunami` |
| All figures from the shipped results | Python ≥ 3.8 with NumPy and Matplotlib | — |

No GPU is needed to verify the numerical method or to regenerate the figures:
`results/` already contains the output of every run.

Tested toolchains: GCC 11.5, CUDA 12.3 and Open MPI 4.1.5 on MareNostrum 5
(Linux, NVIDIA H100) for everything; GCC 13.3 on Ubuntu 24.04 under WSL for the
CPU targets. Other C++17 compilers (for example Clang on macOS) and CUDA 11 or
later should work but have not been tested. The `Makefile` assumes a Unix-like
shell, so on Windows use WSL.

---

## 3. Getting started on a normal computer (CPU only)

### 3.1 Install the tools

**Ubuntu / Debian (also Windows through WSL):**
```bash
sudo apt update
sudo apt install build-essential make git python3 python3-numpy python3-matplotlib
```

**macOS** (with [Homebrew](https://brew.sh)):
```bash
xcode-select --install
brew install make python
python3 -m pip install numpy matplotlib
```

**Windows:** install WSL (`wsl --install` in PowerShell, then reboot), open the
Ubuntu terminal and follow the Ubuntu instructions. Work inside the Linux file
system (for example `~/`), not under `/mnt/c`, which is much slower.

### 3.2 Get the code

```bash
git clone https://github.com/gsalanglada/sgn-hyperbolic-gpu.git
cd sgn-hyperbolic-gpu
```

### 3.3 Build and run the verification tests

```bash
make run-test
```

This compiles the 12 host tests into `build/` and runs them. It takes well under
a minute. Every check prints a line starting with `ok`, and each test ends with

```
ALL TESTS PASSED
```

A failing check prints `FAIL` and `make` stops with an error. The tests cover the
pointwise physics, the FR operators, the time integrators, the 1D and 2D solvers,
exact lake-at-rest preservation over bathymetry, linear dispersion, solitary-wave
propagation, soliton collision, friction and wave-generation zones.

### 3.4 Run your first results

Each of these targets builds a driver **and runs it**, writing CSV files into
`results/`:

```bash
make valfigs        # solitary wave, collisions, well-balancing   (~minutes)
make alpha          # accuracy/cost vs the relaxation parameter    (~minutes)
make convergence    # spatial (1D, 2D) and temporal convergence    (~minutes)
```

> **Note.** These targets overwrite the shipped reference files in `results/`
> with your own run. That is intended: the numbers should agree (see §13).
> To keep the originals, copy `results/` first, for example
> `cp -r results results_reference`.

To build a single program without running it, name its binary:
`make build/convergence1d`, `make build/beji_battjes`, and so on.

### 3.5 Make the figures

```bash
python3 scripts/make_figures.py
```

This reads `results/` and writes every CSV-driven figure of the paper, as PDF
and PNG, into `paper/figs/` (created if missing). It needs no simulation and no
GPU. Pass other directories if you prefer:
`python3 scripts/make_figures.py <results_dir> <output_dir>`.

---

## 4. Adding MPI

**Ubuntu / Debian:** `sudo apt install openmpi-bin libopenmpi-dev`
**macOS:** `brew install open-mpi`

```bash
make run-mpi MPIRANKS=2
```

This runs the CPU MPI solvers on 1, 2 (or `MPIRANKS`) and 4 processes and checks
that they are **bitwise identical** to the serial solver. `MPIRANKS` defaults to
3; lower it on a machine with few cores. If `mpirun` refuses to start more
processes than you have cores, add `--oversubscribe` to the `mpirun` lines of the
`Makefile` or use a smaller `MPIRANKS`.

---

## 5. Adding an NVIDIA GPU

Install the NVIDIA driver and the CUDA toolkit
([instructions](https://developer.nvidia.com/cuda-downloads)), and check that
`nvcc --version` and `nvidia-smi` work.

**Set the GPU architecture.** The `Makefile` compiles for the H100 of
MareNostrum 5 (`-arch=sm_90`). On any other GPU, pass your own architecture or
the kernels will not load:

| GPU family | examples | flag |
|---|---|---|
| Volta | V100 | `-arch=sm_70` |
| Turing | T4, RTX 20xx | `-arch=sm_75` |
| Ampere (data centre) | A100, A30 | `-arch=sm_80` |
| Ampere (consumer) | RTX 30xx, A40 | `-arch=sm_86` |
| Ada | RTX 40xx, L4, L40 | `-arch=sm_89` |
| Hopper | H100, H200 | `-arch=sm_90` |

```bash
make run-cuda NVCCFLAGS="-std=c++17 -O2 -arch=sm_86"
```

This checks the 1D and 2D GPU solvers against the CPU reference: the right-hand
sides agree to about `2e-14` relative (they differ only through fused
multiply–add rounding), and the adaptive integrator takes the same accept/reject
sequence. `make showcase NVCCFLAGS=...` builds and runs the single-GPU shoal
animation, which writes VTK frames (about 180 MB) into `results/`; open them
with ParaView.

---

## 6. Multi-GPU runs (workstation or cluster)

The production drivers (`sgn_bench`, `shoal_focus_mpi`, `tsunami_margin_mpi`)
and `test_cuda_mpi_sgn2d` run one MPI process per GPU and pass GPU memory
pointers directly to MPI. That requires an MPI library **built with CUDA
support**. For Open MPI you can check with

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

which must end in `true`. Distribution packages are often built without it; in
that case use the MPI module of your cluster, or build Open MPI with
`--with-cuda`. With a single process no GPU pointer is ever sent, so
`mpirun -n 1` works with any MPI.

Each process selects a GPU by its rank within the node, so on a workstation with
`n` GPUs start `n` processes:

```bash
make bench NVCCFLAGS="-std=c++17 -O2 -arch=sm_86"
mpirun -n 2 ./build/sgn_bench 512 128 4 0.02 0.5 3     # Nx Ny P tEnd cfl reps
make run-cuda-mpi NVCCFLAGS="-std=c++17 -O2 -arch=sm_86"  # needs 2 and 4 processes
```

**On a Slurm cluster** the job scripts in `scripts/` are ready for MareNostrum 5
(4× H100 per node). Elsewhere, edit their `#SBATCH` lines (account `-A`, queue
`-q`, GPUs per node) and the module names in `setup_env.src`. On MareNostrum 5:

```bash
source setup_env.src          # CUDA 12.3 + Open MPI 4.1.5 (CUDA-aware)
make all bench shoalfocus tsunami
export UCX_TLS=^cuda_ipc      # required there for CUDA-aware MPI; the job scripts set it
sbatch -n 16 scripts/tsunami.sbatch 1000 500 2.5 150
```

Compile on a login node that has the CUDA module, and never rebuild while jobs
that use `build/` are still queued: a queued job runs whatever binary is in
`build/` when it starts.

---

## 7. Verification tests

These are the tests behind the reproducibility claims of the paper. All are
self-checking: each check prints `ok`/`FAIL`, a passing test ends with
`ALL TESTS PASSED`, and a failure returns a non-zero exit code.

| Command | Runs on | Verifies |
|---|---|---|
| `make run-test` | any CPU, serial | physics, FR operators, Runge–Kutta, 1D/2D solvers, exact well-balancing, dispersion, solitary waves, collisions, friction and relaxation zones |
| `make run-mpi` | CPU, 1/`MPIRANKS`/4 processes | CPU-MPI **bitwise** identical to serial |
| `make run-cuda` | 1 GPU | GPU vs CPU right-hand side (`≤ 2e-14` relative) and identical step sequence |
| `make run-cuda-mpi` | 2 and 4 GPUs | multi-GPU **bitwise** identical to single GPU |

The header of each file in `test/` states exactly what it checks and why.

---

## 8. Reproducing each paper result

Run from the repository root; each driver writes into `results/`. The drivers
in the first seven rows run on any CPU; the rest need GPUs.

| Paper result | Command | Output |
|---|---|---|
| Spatial and temporal convergence, and cost versus `P = 1…6` | `make convergence` | `convergence1d.csv`, `convergence2d.csv`, `convergence_time.csv` |
| Central vs upwind interface state, and the origin of the even–odd order loss | `make ncptradeoff` | `ncp_tradeoff.csv`, `ncp_wellbalanced.csv`, `mms_bisect.csv` |
| Solitary wave, collisions, well-balancing | `make valfigs` | `val_solitary.csv`, `val_collision.csv`, `val_collision_uneq.csv`, `val_wb.csv`, `val_wb_bathy.csv` |
| Relaxation parameter α | `make alpha` | `alpha_sweep.csv` |
| Linear dispersion | analytic, drawn directly by `scripts/make_figures.py` | — |
| Beji–Battjes submerged bar | `make beji`, or `./build/beji_battjes [γ=2] [α=3] [P=4] [n_elem] [n_Manning] [A=0.010]` | `beji_gauges.csv`, `beji_profile.csv` (default run); `beji_g*_a*_n*_A*_*.csv` otherwise |
| Bolles rogue-wave statistics | `make bolles`, or `./build/bolles_step [seed] [σ_η] [T_rec=300] [n_Manning]` | `bolles_s*_stats.csv`, `*_hist.csv`, `*_series.csv` |
| Strong and weak scaling | `bash scripts/submit_bench.sh` | `strong_n*.out`, `strongBig_n*.out`, `weak_n*.out` |
| Kernel profiling (Nsight Systems / Compute) | `sbatch scripts/profile_nsys.sbatch`, `sbatch scripts/profile_ncu.sbatch`, then `python3 scripts/collect_ncu.py results/ncu_<jobid>.csv` | `nsys_*_kernels.txt`, `ncu_*.csv`, `ncu_*.out` |
| 2D focusing over a shoal: convergence and device-count independence | `sbatch -n <R> scripts/shoal_focus.sbatch <Nx> <Ny>` | `shoal_focus_*.csv` |
| Geometry sweep of the shoal | `sbatch -n 16 scripts/shoal_sweep.sbatch`, then `python3 scripts/collect_sweep.py` | `shoal_sweep.csv` |
| Dispersive tsunami over a continental margin | `sbatch -n 64 scripts/tsunami.sbatch 2000 1000 2.5 150` (and `1000 500` on 16 GPUs) | `tsunami_margin_*.csv`, `tsunami_*.out` |
| Non-dispersive (shallow-water) baseline of the tsunami | `sbatch -n 4 scripts/tsunami.sbatch 1000 500 2.5 150 0 800 0` (7th argument α = 0) | `tsunami_margin_1000x500_n4_swe.csv` |
| Time-integrator comparison (not used in the paper) | `make rkcompare` | `rk_compare.csv` |
| Shoal animation | `make showcase` | `shoal_*.vtk` (not shipped; ~180 MB) |

**Beji–Battjes mesh convergence** (the five meshes of the paper):

```bash
for ne in 350 700 1050 1400 2100; do ./build/beji_battjes 2.0 3.0 4 $ne; done
```

**Bolles ensemble** (six seeds, σ_η = 4 mm, 300 s records, Manning n = 0.02):

```bash
for s in 1 2 3 4 5 6; do ./build/bolles_step $s 0.0040 300 0.020; done
```

The two-dimensional cases were run on MareNostrum 5; `shoal_focus.sbatch` and
`tsunami.sbatch` need their account, queue and module lines adapted elsewhere
(§6). Passing a non-zero `vtkEvery` (5th argument) to the tsunami script writes
coarsened VTK frames for visualization; the time spent writing them is measured
and excluded from the reported wall time.

---

## 9. Figures

```bash
python3 scripts/make_figures.py            # every CSV-driven figure
python3 scripts/make_tsunami_geometry.py   # tsunami bathymetry and initial condition
python3 scripts/make_tsunami_field.py      # tsunami 2D field panels (needs VTK frames)
python3 scripts/make_showcase_figs.py      # shoal panels (needs VTK frames)
python3 scripts/beji_compare.py            # RMS error of the Beji–Battjes harmonics
```

`make_figures.py` runs on the shipped `results/` without any simulation and
without a GPU. `make_tsunami_geometry.py` evaluates the same depth function as
the solver, so the figure and the simulation cannot drift apart. The two
VTK-based scripts need frames that are not shipped because of their size (see
`MANIFEST.md` for how to regenerate them).

---

## 10. Benchmarks and profiling

`scripts/submit_bench.sh` submits the full strong- and weak-scaling campaign.
Each configuration is preceded by a warm-up, timed five times with MPI barriers
before and after, and the median is reported; no I/O is done inside the timed
region. For a fixed mesh the adaptive controller took the same number of steps
at every device count in these campaigns, so the wall times compare equal
amounts of work.

Throughput is defined once, everywhere, as

```
N_DOF × N_RHS / t_wall
```

degrees of freedom times right-hand-side evaluations (including those of
rejected steps) per wall-clock second. All computations use strict IEEE double
precision (FP64).

`scripts/profile_nsys.sbatch` (Nsight Systems: time per kernel) and
`scripts/profile_ncu.sbatch` (Nsight Compute: L2 hit rate, DRAM traffic,
throughput, occupancy) run on a single GPU while sweeping the mesh size;
`scripts/collect_ncu.py` reduces the counters to the tables of the paper.

---

## 11. Expected runtimes

Approximate; CPU times are for one core of a recent workstation or laptop.

| Task | Hardware | Wall time |
|---|---|---|
| Host verification tests | 1 CPU core | < 1 min |
| GPU and multi-GPU tests | 4× H100 | < 2 min |
| Convergence: spatial (1D + 2D) and temporal | 1 CPU core | ~10–15 min |
| Interface-state study and bisection | 1 CPU core | ~5 min |
| Validation figure data | 1 CPU core | ~5 min |
| Beji–Battjes, all five meshes | 1 CPU core | ~2 h |
| Bolles, six seeds | 1 CPU core | ~40 min |
| Full scaling campaign | 1 → 64 H100 | ~30 min of compute |
| Shoal convergence ladder (5 runs) | 4 → 64 H100 | ~45 min of compute |
| Shoal geometry sweep (16 runs) | 16 H100 | ~35 min |
| Tsunami, 2000 × 1000 elements, 2.5 h simulated | 64 H100 | ~5 min |

---

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvcc: command not found` | CUDA toolkit missing or not in `PATH`. CPU targets do not need it. |
| `no kernel image is available for execution on the device` | Wrong GPU architecture: pass `NVCCFLAGS="-std=c++17 -O2 -arch=sm_XX"` (§5) and rebuild after `make clean`. |
| `mpicxx: command not found` | MPI missing (§4). Only the MPI and multi-GPU targets need it. |
| Multi-GPU run crashes or hangs in `MPI_Sendrecv` | The MPI is not CUDA-aware (§6). On MareNostrum 5, also `export UCX_TLS=^cuda_ipc`. |
| `mpirun` refuses to start the processes | Fewer cores than processes: lower `MPIRANKS` or add `--oversubscribe`. |
| `No rule to make target 'convergence1d'` | Single binaries are built by path: `make build/convergence1d`. |
| `make` says "Nothing to be done" after editing | Header dependencies are tracked; if in doubt, `make clean`. |
| The reference numbers in `results/` changed | The run targets overwrite them; see the note in §3.4. |

---

## 13. Notes on the bitwise-reproducibility claims

Bitwise reproducibility holds **for a fixed binary on fixed hardware** across
decompositions: serial vs MPI vs single GPU vs multi-GPU. It is not a claim of
bit-identity across compilers, CUDA versions or GPU architectures, which would
require disabling fused multiply–add contraction and is not attempted here.

The mechanism: a single host-and-device physics header (no code drift), and an
arithmetic path per element that does not depend on how the domain is
partitioned. The global error norm of the adaptive controller is a sum, and a
floating-point sum depends on its order; identical step sequences across
configurations are therefore observed, not guaranteed, and very long runs can
eventually differ by a few steps (the paper quantifies this).

On a different machine the CPU results agree with the shipped ones to all
printed digits except where the error approaches round-off, where the last
digits can differ with the compiler and its maths library.

---

## 14. Citation and licence

If you use this code, please cite the paper above and this software
(`CITATION.cff`; GitHub shows it under "Cite this repository").

The code is released under the MIT licence (`LICENSE`). The digitized
Beji–Battjes / Dingemans (1994) laboratory data under `data/` are derived from
published figures, are included for verification only and remain due to their
original authors.
