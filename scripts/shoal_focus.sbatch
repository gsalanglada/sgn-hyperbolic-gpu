#!/bin/bash
# 2D production run on the distributed multi-GPU solver: dispersive focusing of
# a solitary wave over a submerged shoal, measuring the focusing amplification
# A = max(eta)/a downstream of the shoal, under mesh refinement.
#
# Mesh-convergence ladder (each level doubles the resolution in both directions,
# so the step count doubles too; ranks are scaled to keep the run time bounded):
#
# The directives below already request one GPU and 20 CPUs per task on full
# nodes, which is what MN5 requires (CPUs = GPUs x 20), so no overrides are
# needed for any rank count that is a multiple of four:
#
#   sbatch -n  4  scripts/shoal_focus.sbatch  384  192
#   sbatch -n  8  scripts/shoal_focus.sbatch  768  384
#   sbatch -n 16  scripts/shoal_focus.sbatch 1536  768
#   sbatch -n 64  scripts/shoal_focus.sbatch 3072 1536
#
# Plus a rank-independence check: the same mesh on a different number of GPUs.
# The reduction of A is a maximum (associative and exact in floating point), so
# it adds no rank-count dependence of its own; the adaptive trajectories feeding
# it can still differ slightly over ~1e5 steps, so equal digits are an observed
# result (they were, in the paper), not a guarantee:
#
#   sbatch -n 32  scripts/shoal_focus.sbatch 1536  768
#
# The finest level is 590M DOF: about half an hour on 64 GPUs, and well over a
# day on a single device — the case exists to be run distributed.
#
# Args: $1=Nx $2=Ny $3=P $4=tEnd $5=samples   (defaults 384 192 4 8.0 100)
# Output: results/shoal_focus_<jobid>.out (log + summary CSV line) and
#         results/shoal_focus_<Nx>x<Ny>_n<ranks>_d<d>_r<r>.csv (time series).

#SBATCH -A bsc21
#SBATCH -q acc_bsccase
#SBATCH -t 01:30:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 20
#SBATCH -o results/shoal_focus_%j.out

source setup_env.src
export UCX_TLS=^cuda_ipc

mkdir -p results

echo "# host=$(hostname)  date=$(date -u +%FT%TZ)"
echo "# nvcc: $(nvcc --version | tail -1)"
echo "# mpi:  $(mpirun --version 2>&1 | head -1)"
echo "# gcc:  $(gcc --version | head -1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 | sed 's/^/# gpu: /'

srun ./build/shoal_focus_mpi ${1:-384} ${2:-192} ${3:-4} ${4:-8.0} ${5:-100}
