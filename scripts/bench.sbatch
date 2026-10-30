#!/bin/bash
# Multi-GPU scaling benchmark on MN5 ACC (4x H100 per node).
#
#   strong scaling (fixed global mesh, growing GPUs):
#     for n in 1 2 4 8 16; do sbatch -n $n scripts/bench.sbatch 1024 256 4 0.02; done
#   weak scaling (fixed load per GPU: scale Nx with ranks):
#     for n in 1 2 4 8; do sbatch -n $n scripts/bench.sbatch $((256*n)) 256 4 0.02; done
#
# Args: $1=Nx  $2=Ny  $3=P  $4=tEnd  $5=reps   (defaults 512 256 4 0.02 5)
# The full campaigns of the paper are submitted by scripts/submit_bench.sh.
# Output: bench_<jobid>.out, or the -o file given by submit_bench.sh, with the
# toolchain lines and the CSV record printed by apps/sgn_bench.cu.
#
# MN5 rule: requested cpus must equal GPUs*20. The defaults below are for
# FULL-NODE jobs (4 GPUs/node, n multiple of 4). For n<4 override on the CLI:
#   sbatch -n 2 --gres=gpu:2 --ntasks-per-node=2 -c 20 ... scripts/bench.sbatch ...
#
#SBATCH -A bsc21
#SBATCH -q acc_debug
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 20
#SBATCH -o bench_%j.out

source setup_env.src
export UCX_TLS=^cuda_ipc

# Record the exact toolchain for reproducibility of the timing runs.
echo "# host=$(hostname)  date=$(date -u +%FT%TZ)"
echo "# nvcc: $(nvcc --version | tail -1)"
echo "# mpi:  $(mpirun --version 2>&1 | head -1)"
echo "# gcc:  $(gcc --version | head -1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 | sed 's/^/# gpu: /'

# Args: Nx Ny P tEnd [reps]. reps defaults to 5 (median + spread reported).
srun ./build/sgn_bench ${1:-512} ${2:-256} ${3:-4} ${4:-0.02} 0.5 ${5:-5}
