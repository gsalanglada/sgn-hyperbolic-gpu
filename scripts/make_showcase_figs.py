#!/usr/bin/env python3
"""Publication figures from the shoal-showcase VTK series.

Reads results/shoal_*.vtk (STRUCTURED_POINTS, ASCII) and produces:
  fig_shoal_panels.pdf : 2x2 eta maps at four instants (refraction/focusing)
  fig_shoal_focus.pdf  : centreline eta + peak-amplification-vs-time curve
Optional PNG previews. Bathymetry contour of the shoal is overlaid.

Usage: python scripts/make_showcase_figs.py [results_dir] [out_dir]
"""
import os
import re
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "paper", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# Case geometry (must match apps/showcase_shoal.cu).
Lx, Ly, tEnd = 40.0, 20.0, 8.0
SHOAL_C = (18.0, 10.0)
SHOAL_R = 3.0


def read_vtk(path):
    """Return (dims(nx,ny), origin(ox,oy), spacing(dx,dy), fields dict)."""
    with open(path) as f:
        txt = f.read()
    nx, ny = map(int, re.search(r"DIMENSIONS (\d+) (\d+)", txt).groups())
    ox, oy = map(float, re.search(r"ORIGIN ([\d.eE+-]+) ([\d.eE+-]+)", txt).groups())
    dx, dy = map(float, re.search(r"SPACING ([\d.eE+-]+) ([\d.eE+-]+)", txt).groups())
    fields = {}
    for m in re.finditer(r"SCALARS (\w+) float 1\nLOOKUP_TABLE default\n", txt):
        name = m.group(1)
        start = m.end()
        end = txt.find("SCALARS", start)
        chunk = txt[start:] if end < 0 else txt[start:end]
        vals = np.fromstring(chunk, sep=" ")
        fields[name] = vals[: nx * ny].reshape(ny, nx)
    return (nx, ny), (ox, oy), (dx, dy), fields


def load_frame(idx):
    """Read frame idx of the shoal VTK series: (x, y, fields), or None if absent."""
    p = os.path.join(RES, f"shoal_{idx:04d}.vtk")
    if not os.path.exists(p):
        return None
    (nx, ny), (ox, oy), (dx, dy), fld = read_vtk(p)
    x = ox + dx * np.arange(nx)
    y = oy + dy * np.arange(ny)
    return x, y, fld


def frame_count():
    """Number of consecutive shoal_NNNN.vtk frames present in RES."""
    n = 0
    while os.path.exists(os.path.join(RES, f"shoal_{n:04d}.vtk")):
        n += 1
    return n


def save(fig, name):
    """Write fig to OUT as <name>.pdf and <name>.png, then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def shoal_circle(ax):
    """Draw the shoal outline as a dashed circle on ax."""
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(SHOAL_C[0] + SHOAL_R * np.cos(th),
            SHOAL_C[1] + SHOAL_R * np.sin(th), "k--", lw=0.6, alpha=0.6)


def main():
    """Build fig_shoal_panels and fig_shoal_focus from all available frames."""
    nF = frame_count()
    if nF == 0:
        print("  no shoal_*.vtk found in", RES)
        return
    print(f"  {nF} frames in {RES}")

    # amplitude scale from a mid-late frame
    fr = load_frame(min(nF - 1, int(0.6 * nF)))
    vmax = 0.9 * np.abs(fr[2]["eta"]).max()

    # ---- panel figure: four instants ------------------------------------
    picks = [int(f * (nF - 1)) for f in (0.30, 0.50, 0.68, 0.90)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True, sharey=True)
    for ax, idx in zip(axes.flat, picks):
        x, y, fld = load_frame(idx)
        pc = ax.pcolormesh(x, y, fld["eta"], cmap="RdBu_r", vmin=-vmax,
                           vmax=vmax, shading="auto", rasterized=True)
        shoal_circle(ax)
        ax.set_title(f"t = {idx / (nF - 1) * tEnd:.1f} s", fontsize=8)
        ax.set_aspect("equal")
    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]")
    for ax in axes[1, :]:
        ax.set_xlabel("x [m]")
    fig.colorbar(pc, ax=axes, shrink=0.85, label=r"$\eta$ [m]", pad=0.02)
    save(fig, "fig_shoal_panels")

    # ---- centreline + peak amplification --------------------------------
    x0, y0, f0 = load_frame(0)
    jc = np.argmin(np.abs(y0 - SHOAL_C[1]))
    a0 = f0["eta"].max()
    times, peaks = [], []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    cmap = plt.cm.viridis
    for k, idx in enumerate([int(f * (nF - 1)) for f in
                             (0.25, 0.4, 0.55, 0.7, 0.85)]):
        x, y, fld = load_frame(idx)
        a1.plot(x, fld["eta"][jc], color=cmap(k / 4),
                label=f"{idx / (nF - 1) * tEnd:.1f} s", lw=1.0)
    a1.axvspan(SHOAL_C[0] - SHOAL_R, SHOAL_C[0] + SHOAL_R, color="0.9")
    a1.set_xlabel("x [m]")
    a1.set_ylabel(r"$\eta$ on $y=y_c$ [m]")
    a1.legend(frameon=False, fontsize=6.5, ncol=2)

    for idx in range(nF):
        x, y, fld = load_frame(idx)
        times.append(idx / (nF - 1) * tEnd)
        peaks.append(fld["eta"].max())
    a2.plot(times, np.array(peaks) / a0, "-", color="C3")
    a2.axhline(1.0, color="k", lw=0.6, ls="--")
    a2.set_xlabel("t [s]")
    a2.set_ylabel(r"peak $\eta\,/\,\eta_0$")
    fig.tight_layout()
    save(fig, "fig_shoal_focus")


print("showcase figures ->", OUT)
main()
print("done")
