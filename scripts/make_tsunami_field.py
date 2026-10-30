#!/usr/bin/env python3
"""Snapshot panels of the tsunami field from the VTK frames.

Reads the legacy-VTK frames written by apps/tsunami_margin_mpi.cu and draws the
free-surface elevation at selected times, with the continental margin outlined
so the refraction at the promontory is legible.

    python3 scripts/make_tsunami_field.py [vtkdir] [outdir]
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VTK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

TEND_MIN = 150.0          # simulated window
VTK_EVERY = 3             # frames were written every 3 samples of 150


def read_vtk(path):
    """Return (fields dict, Cx, Cy, origin, spacing) from a legacy ASCII VTK."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    tok = txt.split()
    i = tok.index("DIMENSIONS")
    Cx, Cy = int(tok[i + 1]), int(tok[i + 2])
    i = tok.index("ORIGIN")
    ox, oy = float(tok[i + 1]), float(tok[i + 2])
    i = tok.index("SPACING")
    sx, sy = float(tok[i + 1]), float(tok[i + 2])

    fields, n = {}, Cx * Cy
    start = 0
    while True:
        try:
            k = tok.index("SCALARS", start)
        except ValueError:
            break
        name = tok[k + 1]
        d = tok.index("default", k) + 1          # after LOOKUP_TABLE default
        fields[name] = np.array(tok[d:d + n], dtype=float).reshape(Cy, Cx)
        start = d + n
    return fields, Cx, Cy, (ox, oy), (sx, sy)


frames = sorted(glob.glob(os.path.join(VTK, "tsunami_*x*_[0-9][0-9][0-9][0-9].vtk")))
if not frames:
    sys.exit("no VTK frames found in %s" % VTK)
print("found %d frames" % len(frames))

# Times to show: the source, the crossing, arrival at the margin, the focus, and
# the state after the peak. Frame k is at t = k * VTK_EVERY * TEND_MIN / nSample.
nframe = len(frames)
dt_frame = TEND_MIN / (nframe - 1)
want = [0.0, 30.0, 60.0, 90.0, 105.0, 118.0]
idx = [min(nframe - 1, int(round(t / dt_frame))) for t in want]

fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.6), sharex=True, sharey=True)
axes = axes.ravel()

# A common symmetric scale, set from the strongest frame so the panels are
# comparable; the source is far above it, so it is clipped deliberately.
ref, Cx, Cy, org, sp = read_vtk(frames[idx[3]])
lim = float(np.percentile(np.abs(ref["eta"]), 99.9))
lim = max(lim, 0.25)

x = (org[0] + sp[0] * np.arange(Cx)) / 1e3
y = (org[1] + sp[1] * np.arange(Cy)) / 1e3

for ax, k in zip(axes, idx):
    f, *_ = read_vtk(frames[k])
    im = ax.pcolormesh(x, y, f["eta"], cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="auto", rasterized=True)
    # Margin outline: the 1000 m and 3000 m isobaths bracket the slope.
    ax.contour(x, y, f["H"], levels=[1000.0, 3000.0], colors="0.35",
               linewidths=0.5)
    ax.text(0.025, 0.93, "$t = %.0f$ min" % (k * dt_frame),
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.7", lw=0.5))
    ax.set_aspect("equal")

for ax in axes[-2:]:
    ax.set_xlabel("x [km]")
for ax in (axes[0], axes[2], axes[4]):
    ax.set_ylabel("y [km]")

cb = fig.colorbar(im, ax=axes.tolist(), pad=0.02, shrink=0.85)
cb.set_label(r"free-surface elevation $\eta$ [m]", fontsize=8)
cb.ax.tick_params(labelsize=7)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_tsunami_field." + ext),
                bbox_inches="tight", dpi=170)
print("wrote fig_tsunami_field.pdf/.png -> %s" % OUT)
print("colour scale clipped at +/- %.3f m" % lim)
