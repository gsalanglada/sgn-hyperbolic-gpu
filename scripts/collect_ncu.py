#!/usr/bin/env python3
"""Reduce the raw ncu CSV of profile_ncu.sbatch to the table the paper needs.

Nsight Compute emits one row per (kernel launch, metric). This averages each
metric over the launches of a kernel, then reports, per working set:

  - the L2 hit rate, which is the quantity that decides whether the superlinear
    strong scaling of Table 5 is a cache-residency effect;
  - DRAM traffic per right-hand-side evaluation, to compare against the ~21
    bytes/DOF lower bound quoted in the paper (the ratio says how much extra
    traffic the unoptimized kernels actually generate);
  - the share of step time per kernel, which answers which operator limits.

  usage: python3 scripts/collect_ncu.py results/ncu_<jobid>.csv
"""
import csv
import sys
from collections import defaultdict


def main(path):
    """Read the Nsight Compute CSV at path and print the three summary tables."""
    # (mesh, ws, kernel, metric) -> [values]
    vals = defaultdict(list)
    order = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            mesh = row.get("mesh")
            if not mesh:
                continue
            try:
                ws = int(row["working_set_MB"])
            except (KeyError, ValueError):
                continue
            kern = (row.get("Kernel Name") or "").split("(")[0].strip()
            metric = (row.get("Metric Name") or "").strip()
            raw = (row.get("Metric Value") or "").replace(",", "").strip()
            try:
                v = float(raw)
            except ValueError:
                continue
            if (mesh, ws) not in order:
                order.append((mesh, ws))
            vals[(mesh, ws, kern, metric)].append(v)

    if not order:
        sys.exit("no usable rows in %s -- did ncu fail? check the .out for "
                 "ERR_NVGPUCTRPERM (locked performance counters)" % path)
    order.sort(key=lambda t: t[1])

    def avg(mesh, ws, kern, metric):
        """Mean of one metric of one kernel at one mesh, or None if absent."""
        v = vals.get((mesh, ws, kern, metric))
        return sum(v) / len(v) if v else None

    kernels = sorted({k for (_, _, k, _) in vals})

    print("=" * 78)
    print("L2 HIT RATE vs WORKING SET  (H100 L2 = 50 MB)")
    print("=" * 78)
    print("  %-10s %8s   %s" % ("mesh", "WS[MB]", "  ".join("%-14s" % k[:14] for k in kernels)))
    for mesh, ws in order:
        cells = []
        for k in kernels:
            h = avg(mesh, ws, k, "lts__t_sector_hit_rate.pct")
            cells.append("%-14s" % ("%.1f%%" % h if h is not None else "--"))
        print("  %-10s %8d   %s" % (mesh, ws, "  ".join(cells)))

    print()
    print("=" * 78)
    print("DRAM TRAFFIC AND TIME PER RHS EVALUATION")
    print("=" * 78)
    print("  %-10s %8s %12s %12s %10s %10s" %
          ("mesh", "WS[MB]", "DRAM[MB]", "B/DOF", "us/RHS", "%peak BW"))
    for mesh, ws in order:
        nx, ny = (int(x) for x in mesh.split("x"))
        dof = nx * ny * 125
        tot_b = tot_t = 0.0
        bwpk = []
        for k in kernels:
            for m in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
                b = avg(mesh, ws, k, m)
                if b:
                    tot_b += b
            t = avg(mesh, ws, k, "gpu__time_duration.sum")
            if t:
                tot_t += t
            p = avg(mesh, ws, k, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
            if p:
                bwpk.append(p)
        print("  %-10s %8d %12.1f %12.1f %10.1f %10s" %
              (mesh, ws, tot_b / 1e6, tot_b / dof, tot_t / 1e3,
               "%.1f%%" % (sum(bwpk) / len(bwpk)) if bwpk else "--"))
    print()
    print("  B/DOF vs the ~21 bytes/DOF lower bound in the paper: the ratio is")
    print("  the extra traffic the unoptimized kernels generate.")

    print()
    print("=" * 78)
    print("WHICH OPERATOR LIMITS  (share of RHS time)")
    print("=" * 78)
    for mesh, ws in order:
        times = [(k, avg(mesh, ws, k, "gpu__time_duration.sum") or 0.0) for k in kernels]
        tot = sum(t for _, t in times) or 1.0
        print("  %s (%d MB):" % (mesh, ws))
        for k, t in sorted(times, key=lambda x: -x[1]):
            sm = avg(mesh, ws, k, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
            oc = avg(mesh, ws, k, "sm__warps_active.avg.pct_of_peak_sustained_active")
            print("      %-16s %5.1f%% of RHS   SM %5s   occup %5s" %
                  (k[:16], 100.0 * t / tot,
                   "%.1f%%" % sm if sm is not None else "--",
                   "%.1f%%" % oc if oc is not None else "--"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
