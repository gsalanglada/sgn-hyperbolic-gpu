#!/usr/bin/env python3
"""Collect the shoal parameter sweep into one table.

Reads the summary lines that shoal_focus_mpi prints (rank 0) from the SLURM
output of scripts/shoal_sweep.sbatch and writes results/shoal_sweep.csv.

The header is read from the file rather than assumed, so runs produced by
different builds of the driver (for instance before and after a column was
added) are collected together; missing columns come back empty.

    python3 scripts/collect_sweep.py [results/shoal_sweep_*.out]
"""

import csv
import glob
import os
import sys

# Every known column, in display order. Rows that lack one leave it blank.
ALL = ["ranks", "Nx", "Ny", "P", "dof", "dShoal", "rShoal", "hCrest", "A",
       "xA", "yA", "tA", "etaRms", "acc", "rej", "wall_s", "massDrift"]

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = os.path.join(root, "results")

paths = sys.argv[1:] or sorted(glob.glob(os.path.join(res, "shoal_sweep_*.out")))
if not paths:
    sys.exit("no sweep output found (results/shoal_sweep_*.out)")

rows, order = [], {}
for p in paths:
    with open(p, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("ranks,Nx,Ny,P,dof,dShoal"):
            continue
        cols = s.split(",")
        if i + 1 >= len(lines):
            continue
        vals = lines[i + 1].strip().split(",")
        if len(vals) != len(cols):
            continue
        rec = {c: "" for c in ALL}
        rec.update(dict(zip(cols, vals)))
        key = (rec["Nx"], rec["Ny"], rec["dShoal"], rec["rShoal"])
        if key in order:                   # a rerun supersedes the earlier one
            rows[order[key]] = rec
        else:
            order[key] = len(rows)
            rows.append(rec)

if not rows:
    sys.exit("no summary lines found in: " + ", ".join(paths))

rows.sort(key=lambda r: (float(r["dShoal"]), float(r["rShoal"])))
out = os.path.join(res, "shoal_sweep.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=ALL)
    w.writeheader()
    w.writerows(rows)

print("collected %d runs -> %s\n" % (len(rows), out))

# A as a function of blockage (rows) and radius (columns).
ds = sorted({float(r["dShoal"]) for r in rows})
rs = sorted({float(r["rShoal"]) for r in rows})
grid = {(float(r["dShoal"]), float(r["rShoal"])): r for r in rows}

print("focusing amplification A = max(eta)/a")
print("%10s" % "d [m] \\ r", end="")
for rr in rs:
    print("%10.1f" % rr, end="")
print("   h_crest")
for dd in ds:
    print("%10.2f" % dd, end="")
    hc = ""
    for rr in rs:
        rec = grid.get((dd, rr))
        if rec:
            print("%10.3f" % float(rec["A"]), end="")
            hc = "%.2f m" % float(rec["hCrest"])
        else:
            print("%10s" % "--", end="")
    print("   %s" % hc)

print("\nfocal distance downstream of the shoal centre (x_A - 18 m)")
print("%10s" % "d [m] \\ r", end="")
for rr in rs:
    print("%10.1f" % rr, end="")
print()
for dd in ds:
    print("%10.2f" % dd, end="")
    for rr in rs:
        rec = grid.get((dd, rr))
        print("%10.2f" % (float(rec["xA"]) - 18.0) if rec else "%10s" % "--",
              end="")
    print()

amax = max(rows, key=lambda r: float(r["A"]))
amin = min(rows, key=lambda r: float(r["A"]))
print("\nstrongest focusing: A=%.4f at d=%.2f r=%.2f (crest depth %.2f m)" %
      (float(amax["A"]), float(amax["dShoal"]), float(amax["rShoal"]),
       float(amax["hCrest"])))
print("weakest focusing:   A=%.4f at d=%.2f r=%.2f (crest depth %.2f m)" %
      (float(amin["A"]), float(amin["dShoal"]), float(amin["rShoal"]),
       float(amin["hCrest"])))
print("max mass drift over the sweep: %.2e" %
      max(abs(float(r["massDrift"])) for r in rows))
missing = [(d, r) for d in ds for r in rs if (d, r) not in grid]
if missing:
    print("\nnot completed: " +
          ", ".join("d=%.2f r=%.1f" % m for m in missing))
