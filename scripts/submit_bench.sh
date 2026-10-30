#!/bin/bash
# Submit the full strong- and weak-scaling campaign.
#
#   bash scripts/submit_bench.sh            # submit everything
#   bash scripts/submit_bench.sh strong     # only the small-mesh strong scan
#   bash scripts/submit_bench.sh strongbig  # only the large-mesh strong scan
#   bash scripts/submit_bench.sh weak       # only weak scaling
#
# MN5 rules encoded here:
#   * requested cpus must equal GPUs*20;
#   * everything goes to acc_bsccase by default: acc_debug allows only one job
#     per user at a time, so a whole campaign cannot be queued there, and
#     acc_debug does not admit the 8- and 16-node jobs either. Every job here is
#     short (<=5 min of compute), so the regular queue is the right home.
#     Override with QUEUE=acc_debug to send a single job to the debug queue.
#   * n<4 needs explicit --gres/--ntasks-per-node because bench.sbatch defaults
#     to full nodes.
#
# tEnd is 0.2 (ten times the first campaign) so that every timed repetition
# lasts seconds rather than fractions of a second; reps=5, median reported.
#
# Environment overrides: TEND, REPS, QUEUE. Each job writes
# results/<tag>_n<ranks>.out containing the CSV record of apps/sgn_bench.cu;
# the paper's scaling tables are built from these files.

set -euo pipefail

mkdir -p results

TEND=${TEND:-0.2}
REPS=${REPS:-5}
QUEUE=${QUEUE:-acc_bsccase}
WHICH=${1:-all}

submit() {           # submit <ranks> <Nx> <Ny> <tag>
  local n=$1 nx=$2 ny=$3 tag=$4
  local nodes=$(( (n + 3) / 4 ))
  local queue="$QUEUE"

  local extra=()
  if [[ $n -lt 4 ]]; then
    extra=(--gres=gpu:$n --ntasks-per-node=$n -c $((n * 20)))
  fi

  echo "  ${tag}: ${n} GPU(s) / ${nodes} node(s) -> ${queue}"
  sbatch -n "$n" -q "$queue" -o "results/${tag}_n${n}.out" "${extra[@]}" \
         scripts/bench.sbatch "$nx" "$ny" 4 "$TEND" "$REPS"
}

if [[ $WHICH == all || $WHICH == strong ]]; then
  echo "strong scaling, 1024x256 elements (32.8M DOF):"
  for n in 1 2 4 8 16; do submit "$n" 1024 256 strong; done
fi

if [[ $WHICH == all || $WHICH == strongbig ]]; then
  echo "strong scaling, 2048x512 elements (131M DOF):"
  for n in 8 16 32 64; do submit "$n" 2048 512 strongBig; done
fi

if [[ $WHICH == all || $WHICH == weak ]]; then
  echo "weak scaling, 8.2M DOF per GPU:"
  for n in 1 2 4 8 16 32 64; do submit "$n" $((256 * n)) 256 weak; done
fi

echo
echo "queued. watch with:  squeue -u \$USER"
echo "output lands as <tag>_n<ranks>.out"
