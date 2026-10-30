#!/usr/bin/env python3
"""Compare Beji harmonic amplitudes (any run) against the digitized lab data.

Usage:
  python scripts/beji_compare.py [results_dir] [gauges_csv ...]
If no gauge files are given, every results/beji_*gauges.csv (and beji_gauges.csv)
is compared. Prints per-run RMS relative error for A1, A2, A3.
"""
import glob
import math
import os
import sys
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
files = sys.argv[2:] if len(sys.argv) > 2 else None
if not files:
    files = sorted(glob.glob(os.path.join(RES, "beji_gauges.csv")) +
                   glob.glob(os.path.join(RES, "beji_g*_gauges.csv")))

T = 2.02
w = 2 * np.pi / T
GX = [5.7, 10.5, 12.5, 13.5, 14.5, 15.7, 17.3, 19.0]

# lab harmonics (first four numeric cols).
lab = {}
for cand in (os.path.join(RES, "beji_lab_harmonics.csv"),
             os.path.join(RES, "beji_extracted_results", "data",
                          "beji_lab_harmonics.csv")):
    if os.path.exists(cand):
        with open(cand) as f:
            next(f)
            for line in f:
                p = line.split(",")
                if len(p) >= 4 and p[0].strip():
                    lab[float(p[0])] = [float(p[1]), float(p[2]), float(p[3])]
        break
if not lab:
    print("no lab file found under", RES)
    sys.exit(1)


def harmonics(path):
    """Harmonic amplitudes at every gauge of a gauges CSV (DFT over the last 10 periods)."""
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    t = d[:, 0]
    m = t > t.max() - 10 * T
    tt = t[m]
    out = {}
    for gi, x in enumerate(GX):
        eta = d[m, gi + 1]
        out[x] = [2 * abs(np.mean(eta * np.exp(-1j * n * w * tt))) * 100
                  for n in range(1, 4)]  # cm
    return out


def rms(model, n):
    """RMS relative error [%] of harmonic n over the gauges that have lab data."""
    num = den = 0.0
    for x in GX:
        if x in lab:
            num += (model[x][n] - lab[x][n]) ** 2
            den += lab[x][n] ** 2
    return 100 * math.sqrt(num / den) if den else float("nan")


print(f"{'run':<42} {'A1%':>6} {'A2%':>6} {'A3%':>6}")
for f in files:
    if not os.path.exists(f):
        continue
    try:
        m = harmonics(f)
    except Exception as e:
        print(f"{os.path.basename(f):<42} (error: {e})")
        continue
    name = os.path.basename(f).replace("beji_", "").replace("_gauges.csv", "")
    print(f"{name:<42} {rms(m,0):6.1f} {rms(m,1):6.1f} {rms(m,2):6.1f}")
