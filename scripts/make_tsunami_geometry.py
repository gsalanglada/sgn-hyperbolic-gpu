#!/usr/bin/env python3
"""Geometry of the tsunami capability case: bathymetry and initial condition.

Draws the same depth function the solver uses (apps/tsunami_margin_mpi.cu), so
the figure and the simulation cannot drift apart.

    python3 scripts/make_tsunami_geometry.py [outdir]
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

# --- parameters, mirroring the driver ------------------------------------
Lx, Ly = 4.0e6, 2.0e6
hDeep, hShelf = 4000.0, 150.0
xW, xE, wSlope, bend = 1.3e6, 2.7e6, 5.0e4, 2.0e5
xSrc, ySrc, aSrc = 2.0e6, 1.0e6, 2.0
sxSrc, sySrc = 5.0e4, 2.0e5


def depth(x, y):
    """Still-water DEPTH H > 0: distance from the undisturbed surface down."""
    xEy = xE + bend * np.cos(2.0 * np.pi * (y - ySrc) / Ly)
    s = 0.5 * (np.tanh((x - xW) / wSlope) - np.tanh((x - xEy) / wSlope))
    return hShelf + (hDeep - hShelf) * s


def eta0(x, y):
    """Initial free-surface elevation about the still level z = 0."""
    return aSrc * np.exp(-(((x - xSrc) / sxSrc) ** 2 + ((y - ySrc) / sySrc) ** 2))


def panel_label(ax, text):
    """Corner label only; the description belongs in the LaTeX caption."""
    ax.text(0.012, 0.955, text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.22", fc="w", ec="0.7", lw=0.6))


fig = plt.figure(figsize=(7.0, 5.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.30)

# --- (a) cross-section through the source --------------------------------
ax = fig.add_subplot(gs[0])
x = np.linspace(0.0, Lx, 4000)
H = depth(x, ySrc)
e = eta0(x, ySrc)

ax.fill_between(x / 1e3, -H, -hDeep * 1.12, color="0.72", lw=0)     # solid earth
ax.fill_between(x / 1e3, -H, 0.0, color="#bcd8ee", lw=0)            # water
ax.plot(x / 1e3, -H, color="0.25", lw=1.2)                          # sea bed
ax.axhline(0.0, color="C0", lw=1.0, ls="--")                        # still level

# The surface displacement is metres against a 4 km depth: exaggerate to see it.
EXAG = 250.0
ax.plot(x / 1e3, e * EXAG, color="C3", lw=1.6)
ax.annotate(r"initial $\eta$ = 2 m (drawn $\times$%d)" % EXAG,
            xy=(xSrc / 1e3, aSrc * EXAG), xytext=(xSrc / 1e3 + 380, 730),
            color="C3", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="C3", lw=0.9))
# Labels sit clear of the bed line and the surface line, with leader arrows
# where the feature is too thin to write on: over the shelf the water column is
# 150 m against a 4 km vertical range, so no text fits inside it.
ax.annotate("still water level  $z=0$", xy=(430, 260), fontsize=8, color="C0")
ax.annotate("abyssal plain,  $H=4000$ m", xy=(2000, -2150), fontsize=8,
            ha="center", color="0.15")
ax.annotate("sea bed  $z=-H$", xy=(1520, -3520), fontsize=8, color="0.15")
_arrow = dict(arrowstyle="->", color="0.15", lw=0.8, shrinkA=1.0, shrinkB=1.0)
ax.annotate("shelf,  $H=150$ m", xy=(620, -150), xytext=(600, -1500),
            fontsize=8, ha="center", color="0.15", arrowprops=_arrow)
ax.annotate("continental slope", xy=(2900, -2075), xytext=(3400, -2900),
            fontsize=8, ha="center", color="0.15", arrowprops=_arrow)

ax.set_xlim(0, Lx / 1e3)
ax.set_ylim(-hDeep * 1.12, 1200)
ax.set_xlabel("x [km]")
ax.set_ylabel("z [m]")
panel_label(ax, "(a)")

# --- (b) map view of the depth -------------------------------------------
ax2 = fig.add_subplot(gs[1])
xg = np.linspace(0.0, Lx, 700)
yg = np.linspace(0.0, Ly, 350)
X, Y = np.meshgrid(xg, yg)
Hm = depth(X, Y)

im = ax2.pcolormesh(X / 1e3, Y / 1e3, Hm, cmap="Blues", shading="auto")
cb = fig.colorbar(im, ax=ax2, pad=0.015)
cb.set_label("still depth $H$ [m]", fontsize=8)
cb.ax.tick_params(labelsize=7)
ax2.contour(X / 1e3, Y / 1e3, Hm, levels=[1000.0, 3000.0], colors="0.35",
            linewidths=0.6)

# source footprint and gauges
th = np.linspace(0, 2 * np.pi, 200)
ax2.plot((xSrc + sxSrc * np.cos(th)) / 1e3, (ySrc + sySrc * np.sin(th)) / 1e3,
         color="C3", lw=1.4)
ax2.annotate("source", xy=(xSrc / 1e3, (ySrc + sySrc) / 1e3 + 60), color="C3",
             fontsize=8, ha="center")
gy = np.array([ySrc + (k - 2) * 0.18 * Ly for k in range(5)])
gx = xE + bend * np.cos(2.0 * np.pi * (gy - ySrc) / Ly) + 1.2e5
ax2.plot(gx / 1e3, gy / 1e3, "o", ms=4, color="k", mfc="w", mew=1.1)
ax2.annotate("gauges", xy=(gx.max() / 1e3 + 90, gy[0] / 1e3), fontsize=8)

ax2.set_xlabel("x [km]")
ax2.set_ylabel("y [km]")
panel_label(ax2, "(b)")
ax2.set_aspect("equal")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_tsunami_geometry." + ext),
                bbox_inches="tight", dpi=160)
print("wrote fig_tsunami_geometry.pdf/.png -> %s" % OUT)

# Also emit the panels separately, in case they read better as two figures.
for ax_, tag in ((ax, "section"), (ax2, "map")):
    ext_box = ax_.get_tightbbox(fig.canvas.get_renderer())
    box = ext_box.transformed(fig.dpi_scale_trans.inverted()).expanded(1.06,
                                                                      1.06)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "fig_tsunami_%s.%s" % (tag, ext)),
                    bbox_inches=box, dpi=160)
print("wrote fig_tsunami_section.* and fig_tsunami_map.* (separate panels)")

# --- numbers worth stating -----------------------------------------------
print("\nstill depth H > 0 is measured DOWN from the undisturbed surface:")
print("  z = 0        undisturbed free surface (reference level)")
print("  z = -H       sea bed          (H = 4000 m basin, 150 m shelf)")
print("  z = eta      actual surface   (eta = h - H, here at most a few m)")
print("  h = H + eta  total water column, the solver's unknown")
print("\nlong-wave speed sqrt(gH): %.0f m/s in the basin, %.0f m/s on the shelf"
      % (np.sqrt(9.81 * hDeep), np.sqrt(9.81 * hShelf)))
print("source to slope toe: %.0f km -> arrival after about %.0f min"
      % ((xE - xSrc) / 1e3, (xE - xSrc) / np.sqrt(9.81 * hDeep) / 60.0))
print("Green's law shoaling 4000 m -> 150 m: x%.2f  (2.0 m -> %.2f m)"
      % ((hDeep / hShelf) ** 0.25, aSrc * (hDeep / hShelf) ** 0.25))
