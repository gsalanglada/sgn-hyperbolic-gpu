#!/bin/bash
# Capability case: dispersive tsunami propagation across an ocean basin onto a
# continental margin, on the distributed multi-GPU solver.
#
# Resolution ladder (4000 x 2000 km basin, 2 h of propagation). Wall times are
# estimates from the measured per-GPU throughput; the single-GPU column is what
# makes the case:
#
#   dx      elements      DOF     64 GPUs    1 GPU     memory
#   800 m   1000x500      62M     ~1 min     ~1.7 h    3.5 GB
#   400 m   2000x1000    250M     ~8 min    ~13.7 h     14 GB
#   200 m   4000x2000   1000M    ~65 min   ~4.6 days    56 GB
#
#   sbatch -n 16 scripts/tsunami.sbatch 1000  500 2.0     # smoke test
#   sbatch -n 64 scripts/tsunami.sbatch 2000 1000 2.0     # production
#   sbatch -n 64 scripts/tsunami.sbatch 4000 2000 2.0     # headline, 1e9 DOF
#
# Args: $1=Nx $2=Ny $3=tEnd[h] $4=samples $5=vtkEvery $6=vtkCells $7=alpha $8=slope[km]

#SBATCH -A bsc21
#SBATCH -q acc_bsccase
#SBATCH -t 02:30:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 20
#SBATCH -o results/tsunami_%j.out

source setup_env.src
export UCX_TLS=^cuda_ipc

mkdir -p results

echo "# host=$(hostname)  date=$(date -u +%FT%TZ)"
echo "# nvcc: $(nvcc --version | tail -1)"
echo "# mpi:  $(mpirun --version 2>&1 | head -1)"
echo "# gcc:  $(gcc --version | head -1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 | sed 's/^/# gpu: /'

# Args: Nx Ny tEnd[hours] samples vtkEvery vtkCells alpha wSlopeKm (P is fixed at 4).
# vtkEvery defaults to 0 (no output) so that timing runs are not polluted by it;
# pass a positive value to write one frame every that many samples.
# alpha defaults to 3 (hyperbolic SGN). alpha = 0 runs the SAME configuration as
# non-dispersive shallow water -- the baseline that isolates dispersion from the
# non-adiabatic slope and the two-dimensional refraction:
#
#   sbatch -n 64 scripts/tsunami.sbatch 2000 1000 2.5 150 0 800 0
#
# wSlopeKm defaults to 50 (production). Widening it makes the margin more
# adiabatic and should reduce the reflection part of the shoaling deficit;
# whether it also changes the dispersive part is what the sweep measures.
srun ./build/tsunami_margin_mpi ${1:-1000} ${2:-500} 4 ${3:-2.0} ${4:-120} \
     ${5:-0} ${6:-800} ${7:-3.0} ${8:-50}
