# Manifest and provenance

## Software version

| Field | Value |
|---|---|
| Manuscript | *A reproducible distributed multi-GPU flux-reconstruction solver for hyperbolic Serre–Green–Naghdi water waves* |
| Authors | Gastón Sal-Anglada, Oriol Lehmkuhl (Barcelona Supercomputing Center) |
| Package version | v1.0.0 |
| Source repository / tag | https://github.com/gsalanglada/sgn-hyperbolic-gpu, tag `v1.0.0` |
| Archive DOI | assigned by Zenodo when the `v1.0.0` GitHub release is archived |
| Date | 2026-10-30 |

## Toolchain used for all reported results

| Component | Version |
|---|---|
| Machine | MareNostrum 5, ACC partition (Barcelona Supercomputing Center) |
| GPU | NVIDIA H100, 4 per node |
| CUDA | 12.3 |
| MPI | OpenMPI 4.1.5 (CUDA-aware, InfiniBand) |
| Host compiler | GCC 11.5 |
| Precision | IEEE double (FP64) throughout; no mixed or reduced precision |

Compile flags are recorded in `Makefile` and echoed into the benchmark job
output. `setup_env.src` pins the module environment. `UCX_TLS=^cuda_ipc` is
required for CUDA-aware MPI on this machine.

## Physical and numerical parameters

| Parameter | Value | Where |
|---|---|---|
| Polynomial degree | `P = 4` (GLL nodes, Huynh correction) | production everywhere |
| Riemann solver | Rusanov / local Lax–Friedrichs | all |
| Non-conservative product | central interface state `h* = (h⁻+h⁺)/2` | all |
| Relaxation celerity | `c = α √(g H₀)`, `α = 3` | production; swept in `alpha_sweep` |
| Dispersion parameter | `γ = 3/2` (canonical SGN tests) | dispersion, solitary, collisions, well-balancing |
| | `γ = 2` (Sainte-Marie variant) | Beji–Battjes, Bolles |
| Time integrator | Bogacki–Shampine 3(2), FSAL, PI controller | production |
| Tolerances | `atol = 1e-8`, `rtol = 1e-6` | production |
| PI constants | `α_c = 0.7/3`, `β_c = 0.4/3`, `S = 0.9`, `f_min = 0.2`, `f_max = 5` | production |
| Gravity | `g = 9.81 m/s²` | all |

## Laboratory data

`data/beji_lab_harmonics.csv` and the files under `data/beji_extracted/` are
harmonic amplitudes digitized from the published figures of the Beji–Battjes
submerged-bar experiment as reported by Dingemans (1994), condition A. The
digitization procedure and quality-control overlays are included so the
extraction can be audited. These data are reproduced here for verification
purposes; the original measurements are due to the cited authors.

The Bolles–Speer–Moore rogue-wave experiment is used as a *reference for
comparison only*; no experimental data from it are redistributed here. The
values quoted in the manuscript are read from the published paper.

## What is not shipped

| Item | Why | How to regenerate |
|---|---|---|
| `shoal_*.vtk` (~180 MB) | visualisation only, large, not needed for any quantitative result | `make showcase && ./build/showcase_shoal` |
| `tsunami_*.vtk` (~800 MB) | visualisation only; the quantitative output is in the CSV | pass a non-zero `vtkEvery` to `tsunami_margin_mpi` |
| Raw Nsight reports (`*.nsys-rep`, `*.sqlite`) | binary, tool-version specific; the per-kernel summaries are shipped as text | `sbatch scripts/profile_nsys.sbatch` |
| `build/` | binaries are machine-specific | `make` |
| Raw Bolles time series beyond the six reported seeds | superseded by the shipped ensemble | `./build/bolles_step <seed> 0.0040 0.020` |
| The manuscript sources | not part of the software artefact | — |

## Reference results included

Everything under `results/` is the exact output used to produce the manuscript
tables and figures:

- `convergence1d.csv` — MMS spatial convergence and cost to a fixed error,
  `P = 1…6` (with wall time, RHS evaluations and a round-off-floor flag);
  `convergence2d.csv` — the same in 2D, `P = 1…4`
- `val_solitary.csv`, `val_collision.csv`, `val_collision_uneq.csv`,
  `val_wb.csv`, `val_wb_bathy.csv` — validation ladder
- `alpha_sweep.csv` — relaxation-parameter study
- `beji_gauges.csv`, `beji_profile.csv` — Beji–Battjes production run;
  `beji_g*_a*_n*_gauges.csv` — γ, α and Manning sensitivity sweeps
- `bolles_s{1..6}_a0.0040_n0.020_*.csv` — six-seed rogue-wave ensemble;
  `bolles_s1_a0.0040_n0.0{00,10,20,30}_*.csv` — Manning sweep;
  `bolles_sweep.csv` — aggregated ensemble statistics
- `convergence_time.csv` — temporal convergence of the production integrator,
  with fixed-step RK4 as a control on the measurement
- `ncp_tradeoff.csv`, `ncp_wellbalanced.csv` — central versus upwinded interface
  state for the non-conservative product, and the lake-at-rest residual for both
- `mms_bisect.csv` — the bisection that localizes the even--odd order loss:
  relaxation and bathymetry switched on and off independently, `P = 1…4`
- `shoal_focus_*.csv` — two-dimensional focusing over a submerged shoal, five
  meshes plus the same mesh on two different device counts
- `shoal_sweep.csv` — amplification as a function of the obstacle geometry
- `tsunami_margin_*.csv`, `tsunami_*.out` — the capability case: shelf elevation,
  gauge records, cost; `tsunami_margin_1000x500_n4_swe.csv` is the non-dispersive
  (α = 0) baseline used to isolate the dispersive contribution
- `nsys_*_kernels.txt` — Nsight Systems per-kernel time summaries;
  `ncu_*.csv`, `ncu_*.out` — Nsight Compute hardware counters (L2 hit rate, DRAM
  traffic, throughput, occupancy) behind the kernel-profiling table
- `rk_compare.csv` — fixed-step vs adaptive time integration (not used in the
  paper; retained because it documents why the comparison was not included)
- `strong_n*.out`, `strongBig_n*.out`, `weak_n*.out` — raw benchmark job output,
  including the toolchain header recorded by the job script
