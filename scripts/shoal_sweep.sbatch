#!/bin/bash
# Parameter sweep: how the focusing amplification A = max(eta)/a downstream of a
# submerged shoal depends on the obstacle geometry.
#
#   sbatch -n 8 scripts/shoal_sweep.sbatch
#
# This is the question a bathymetry-based mitigation study has to answer: given
# a budget of material to place on the sea floor, what shape actually reduces
# the wave that reaches the coast, and is there a configuration that makes
# things worse? Each point of the sweep is a full 2D dispersive simulation.
#
# Mesh. The convergence ladder (scripts/shoal_focus.sbatch) established that
# 768x384 elements reproduce the 590M-DOF result to 2e-6 in A, so the sweep runs
# there: converged, and about three minutes per point on 8 GPUs.
#
# Grid: 5 crest elevations x 4 radii = 20 runs, ~1 h total on 8 GPUs.
# The d = 0.90 row (10 cm of water over the crest, three times less than the
# wave height) aborts at the dtMin floor: the wave would break over the crest,
# which the model does not represent. The paper's table omits that row.
# Every run appends its summary line to this job's .out; collect them with
#   python3 scripts/collect_sweep.py results/shoal_sweep_<jobid>.out

#SBATCH -A bsc21
#SBATCH -q acc_bsccase
#SBATCH -t 02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 20
#SBATCH -o results/shoal_sweep_%j.out

source setup_env.src
export UCX_TLS=^cuda_ipc

mkdir -p results

echo "# host=$(hostname)  date=$(date -u +%FT%TZ)"
echo "# nvcc: $(nvcc --version | tail -1)"
echo "# mpi:  $(mpirun --version 2>&1 | head -1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 | sed 's/^/# gpu: /'

NX=${NX:-768}
NY=${NY:-384}
TEND=${TEND:-8.0}
SAMPLES=${SAMPLES:-100}

# Crest elevation d (still depth over the crest is 1 - d) and Gaussian radius r.
# d = 0.90 leaves only 10 cm of water over the crest: the strongly blocking end.
# Override to shorten the sweep, e.g.  DLIST="0.30 0.60 0.90" RLIST="1.5 3.0 6.0"
DLIST=${DLIST:-"0.30 0.45 0.60 0.75 0.90"}
RLIST=${RLIST:-"1.5 3.0 4.5 6.0"}

for d in $DLIST; do
  for r in $RLIST; do
    echo "=== d=$d r=$r ==="
    srun ./build/shoal_focus_mpi "$NX" "$NY" 4 "$TEND" "$SAMPLES" "$d" "$r"
  done
done

echo "# sweep done. Collect with: scripts/collect_sweep.py"
