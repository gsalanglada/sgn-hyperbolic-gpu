#!/bin/bash
# WHICH kernels dominate the step. Cheap: nsys traces the real run at full
# speed and reports a per-kernel time breakdown, so it answers "which operators
# limit us" directly. It does NOT explain why throughput depends on the working
# set -- that needs hardware counters, see profile_ncu.sbatch.
#
# One GPU, two meshes spanning the working-set range measured in the paper:
#   512x128  ->  8.2M DOF  ~172 MB  (the weak-scaling footprint)
#   1024x256 -> 32.8M DOF  ~688 MB  (the large-footprint end)
#
#   sbatch scripts/profile_nsys.sbatch
#
# Output: results/nsys_<mesh>_<jobid>.txt  (kernel summary, sorted by time)

#SBATCH -A bsc21
#SBATCH -q acc_debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH -o results/nsys_%j.out

source setup_env.src
export UCX_TLS=^cuda_ipc
mkdir -p results

echo "# host=$(hostname)  date=$(date -u +%FT%TZ)"
nsys --version | head -1 | sed 's/^/# /'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/# gpu: /'
echo

for MESH in "512 128" "1024 256"; do
  set -- $MESH
  NX=$1; NY=$2
  TAG="${NX}x${NY}"
  REP="results/nsys_${TAG}_${SLURM_JOB_ID}"

  echo "=================================================================="
  echo "== mesh ${TAG}  ($(( NX * NY * 125 )) DOF, ~$(( NX * NY * 125 * 21 / 1000000 )) MB working set)"
  echo "=================================================================="

  srun -n 1 nsys profile \
       --trace=cuda,nvtx \
       --cuda-memory-usage=false \
       --force-overwrite=true \
       -o "$REP" \
       ./build/sgn_bench $NX $NY 4 0.05

  # Per-kernel summary. nsys 2023+ uses "cuda_gpu_kern_sum"; older builds
  # called it "gpukernsum". Try the new name, fall back to the old one.
  nsys stats --report cuda_gpu_kern_sum --format table "$REP.nsys-rep" \
       > "${REP}_kernels.txt" 2>/dev/null \
    || nsys stats --report gpukernsum --format table "$REP.nsys-rep" \
       > "${REP}_kernels.txt" 2>/dev/null

  echo "--- kernel time breakdown (${TAG}) ---"
  cat "${REP}_kernels.txt" 2>/dev/null || echo "  (nsys stats failed; open $REP.nsys-rep in the GUI)"
  echo
done

echo "# done. reports -> results/nsys_*_${SLURM_JOB_ID}*"
