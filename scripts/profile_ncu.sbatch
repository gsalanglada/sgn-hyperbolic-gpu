#!/bin/bash
# WHY per-GPU throughput depends on the working set. This is the measurement the
# paper says three times it is missing (Sections "Numerical method", "Performance
# and scaling", twice more in the discussion of the collapse).
#
# The experiment is deliberately SINGLE GPU. The paper infers the working-set
# dependence by comparing configurations that reach the same footprint with
# different device counts; here the footprint is swept directly on one device,
# with no MPI and no decomposition, so nothing else varies:
#
#   mesh        DOF      working set   vs H100 L2 (50 MB)
#   256x64     2.05M       43 MB       FITS in L2
#   512x64     4.10M       86 MB       1.7x
#   512x128    8.19M      172 MB       3.4x   <- weak-scaling footprint
#   1024x128  16.4M       344 MB       6.9x
#   1024x256  32.8M       688 MB      13.8x
#
# If the L2 hit rate rises as the footprint shrinks, and DRAM traffic falls with
# it, the superlinear strong scaling is a cache-residency effect and can be
# stated as such. If the hit rate is flat, that hypothesis is dead and the text
# must keep attributing the effect to something else.
#
#   sbatch scripts/profile_ncu.sbatch
#
# NOTE: ncu needs GPU performance counters enabled for non-root users. If the
# run fails with ERR_NVGPUCTRPERM the counters are locked and this needs a
# request to BSC support; nsys (profile_nsys.sbatch) still works without them.
#
# Output: results/ncu_<jobid>.csv  (one row per kernel launch, all metrics)

#SBATCH -A bsc21
#SBATCH -q acc_debug
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH -o results/ncu_%j.out

source setup_env.src
export UCX_TLS=^cuda_ipc
mkdir -p results

OUT="results/ncu_${SLURM_JOB_ID}.csv"

echo "# host=$(hostname)  date=$(date -u +%FT%TZ)"
ncu --version | head -2 | sed 's/^/# /'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/# gpu: /'
echo

# L2 hit rate is the discriminating metric; the rest bound the traffic and say
# whether the kernel is anywhere near a compute or bandwidth roof.
METRICS="lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum"

# The five RHS kernels. kComb*/kErrNorm belong to the integrator, not the
# operator, and are excluded so the numbers describe the right-hand side.
KERNELS="regex:kNodalFlux|kFaces|kPack|kInterfaces|kVolume"

FIRST=1
for MESH in "256 64" "512 64" "512 128" "1024 128" "1024 256"; do
  set -- $MESH
  NX=$1; NY=$2
  DOF=$(( NX * NY * 125 ))
  WS=$(( DOF * 21 / 1000000 ))

  echo "== mesh ${NX}x${NY}: ${DOF} DOF, ~${WS} MB working set =="

  # --launch-skip 40 steps past warm-up and the first steps, whose step size the
  # controller is still adjusting; 15 launches per step (5 kernels x 3 stages)
  # so 40 skipped is a bit under three full steps.
  srun -n 1 ncu --target-processes all \
       --kernel-name "$KERNELS" \
       --launch-skip 40 --launch-count 15 \
       --metrics "$METRICS" \
       --csv \
       ./build/sgn_bench $NX $NY 4 0.05 \
    | awk -v nx="$NX" -v ny="$NY" -v ws="$WS" -v first="$FIRST" '
        /^"?ID"?,/ { if (first) print "mesh,working_set_MB," $0; next }
        /^"/       { print nx "x" ny "," ws "," $0 }
      ' >> "$OUT"

  FIRST=0
  echo
done

echo "# done. metrics -> $OUT"
echo "# collect with: python3 scripts/collect_ncu.py $OUT"
