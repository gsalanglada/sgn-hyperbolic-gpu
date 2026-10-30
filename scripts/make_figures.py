#!/usr/bin/env python3
"""Paper figures from results/*.csv.

Usage:  python scripts/make_figures.py [results_dir] [out_dir]
Defaults: results -> <repo>/results, output -> <repo>/paper/figs.

Figure (pdf + png)    input in results_dir                   produced by
fig_scaling           strong_n*, strongBig_n*, weak_n* .out  scripts/submit_bench.sh
fig_alpha             alpha_sweep.csv                        apps/alpha_sweep.cpp
fig_convergence       convergence1d.csv, convergence2d.csv   apps/convergence{1,2}d.cpp
fig_dispersion        (analytic, no input)                   -
fig_solitary          val_solitary.csv                       apps/val_figures.cpp
fig_collision         val_collision.csv                      apps/val_figures.cpp
fig_collision_uneq    val_collision_uneq.csv                 apps/val_figures.cpp
fig_wb                val_wb.csv, val_wb_bathy.csv           apps/val_figures.cpp
fig_rkcompare         rk_compare.csv                         apps/rk_compare.cpp
fig_bolles_stats      bolles_s{1..6}_a0.0040_n0.020_stats.csv apps/bolles_step.cpp
fig_bolles_hist       bolles_s{1,2}_a0.0040_n0.020_hist.csv  apps/bolles_step.cpp
fig_beji_harmonics    beji_gauges.csv + digitized lab data   apps/beji_battjes.cpp

A figure whose input is missing is skipped with a message. The shoal and
tsunami figures come from make_showcase_figs.py, make_tsunami_geometry.py and
make_tsunami_field.py.
"""
import csv
import os
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


def save(fig, name):
    """Write fig to OUT as <name>.pdf and <name>.png, then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def read_csv(path):
    """Read a numeric CSV; return (header list, 2D numpy array of the rows)."""
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = [[float(x) for x in row] for row in r if row]
    return head, np.array(rows)


# ---------------------------------------------------------------- scaling ---
def fig_scaling():
    """Strong and weak scaling from the sgn_bench .out files (strong_n*, strongBig_n*, weak_n*)."""
    def read_bench(pattern):
        """Parse the CSV record of every <pattern>*.out file into {ranks: {wall, ms}}."""
        out = {}
        for f in os.listdir(RES):
            if f.startswith(pattern) and f.endswith(".out"):
                for line in open(os.path.join(RES, f)):
                    if line[0].isdigit():
                        v = line.strip().split(",")
                        out[int(v[0])] = dict(wall=float(v[8]), ms=float(v[9]))
        return dict(sorted(out.items()))

    strong = read_bench("strong_n")     # 32.8M DOF, 1-16 GPU
    strongB = read_bench("strongBig_n")  # 131M DOF, 16-64 GPU
    weak = read_bench("weak_n")
    if not strong or not weak:
        print("  (scaling outputs missing, skipped)")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

    # Strong: two mesh sizes on one log-log wall-time panel, on WHOLE NODES only
    # (4 GPUs per node), as Table tab:scaling. Efficiencies are referred to the
    # smallest whole-node configuration of each campaign; the sub-node points
    # (1 and 2 GPUs) belong to the intra-node table and are not plotted here, so
    # the figure never mixes filling one node with adding nodes.
    GPN = 4

    def plot_strong(bench, color, marker, label):
        """Plot one strong-scaling campaign vs nodes, with ideal line and efficiencies."""
        keep = [k for k in bench if k % GPN == 0]
        n = np.array(keep) / GPN
        w = np.array([bench[k]["wall"] for k in keep])
        a1.loglog(n, w, marker + "-", color=color, label=label)
        a1.loglog(n, w[0] * n[0] / n, color=color, ls="--", lw=0.7)
        for ni, wi in zip(n, w):
            eff = 100.0 * w[0] * n[0] / (ni * wi)
            a1.annotate(f"{eff:.0f}%", (ni, wi), textcoords="offset points",
                        xytext=(3, 5), fontsize=6.5, color=color)
        return n
    n1 = plot_strong(strong, "C0", "o", "32.8M DOF")
    alln = n1
    if strongB:
        alln = np.array(sorted(set(list(n1)) | set(k / GPN for k in strongB if k % GPN == 0)))
        plot_strong(strongB, "C1", "s", "131M DOF")
    alln = alln.astype(int)
    a1.set_xlim(alln[0] / 1.25, alln[-1] * 1.5)       # room for the efficiency labels
    lo, hi = a1.get_ylim()
    a1.set_ylim(lo, hi * 1.6)
    a1.set_xlabel("nodes (4 GPUs each)")
    a1.set_ylabel("wall time [s]")
    a1.set_title("strong scaling", fontsize=9)
    a1.set_xticks(alln, [str(k) for k in alln])
    a1.legend(frameon=False, fontsize=7.5)

    nw = np.array(list(weak))
    ms = np.array([weak[k]["ms"] for k in weak])
    a2.semilogx(nw, ms, "s-", color="C3")
    a2.axhline(ms[0], color="k", ls="--", lw=0.8, label="ideal")
    a2.set_ylim(0, 1.3 * ms.max())
    a2.set_xlabel("GPUs")
    a2.set_ylabel("time per step [ms]")
    a2.set_title("weak scaling (8.2M DOF/GPU)", fontsize=9)
    a2.set_xticks(nw, [str(k) for k in nw])
    a2.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_scaling")


# ----------------------------------------------------------- bolles stats ---
def fig_bolles_stats():
    """Skewness and excess kurtosis vs distance from the step, 6-seed ensemble."""
    ens = []
    for s in range(1, 7):
        p = os.path.join(RES, f"bolles_s{s}_a0.0040_n0.020_stats.csv")
        if os.path.exists(p):
            _, d = read_csv(p)
            ens.append(d)
    if not ens:
        print("  (bolles ensemble missing, skipped)")
        return
    x = ens[0][:, 0]
    sk = np.array([d[:, 2] for d in ens])
    ku = np.array([d[:, 3] for d in ens])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.4, 4.0), sharex=True)
    for a, q, lab in ((a1, sk, "skewness"), (a2, ku, "excess kurtosis")):
        a.fill_between(x, q.min(0), q.max(0), alpha=0.25, color="C0",
                       label="ensemble range (6 seeds)")
        a.plot(x, q.mean(0), "-", color="C0", label="ensemble mean")
        a.axvline(0, color="k", lw=0.8)
        a.axvline(15, color="C3", ls="--", lw=0.9, label="lab hot spot (+15 cm)")
        a.axhline(0, color="k", lw=0.4, alpha=0.5)
        a.set_ylabel(lab)
    a1.legend(frameon=False, fontsize=7)
    a2.set_xlabel(r"$x - x_{\rm step}$ [cm]")
    fig.tight_layout()
    save(fig, "fig_bolles_stats")


# ------------------------------------------------------------ bolles hist ---
def fig_bolles_hist():
    """Hot-spot histogram of eta with the gamma and Gaussian fits."""
    p = os.path.join(RES, "bolles_s2_a0.0040_n0.020_hist.csv")
    if not os.path.exists(p):
        p = os.path.join(RES, "bolles_s1_a0.0040_n0.020_hist.csv")
    if not os.path.exists(p):
        print("  (bolles hist missing, skipped)")
        return
    _, d = read_csv(p)
    eta, dens, pg, pn = d[:, 0] * 100, d[:, 1] / 100, d[:, 2] / 100, d[:, 3] / 100
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(eta, dens, width=0.9 * (eta[1] - eta[0]), color="0.75",
           label=r"simulated $\eta$ (hot spot)")
    ax.plot(eta, pg, "--", color="C0", lw=1.4, label="gamma fit")
    ax.plot(eta, pn, ":", color="C2", lw=1.4, label="Gaussian")
    ax.set_yscale("log")
    ax.set_ylim(dens[dens > 0].min() * 0.5, dens.max() * 3)
    ax.set_xlabel(r"$\eta$ [cm]")
    ax.set_ylabel(r"$p(\eta)$ [1/cm]")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_bolles_hist")


# --------------------------------------------------------- beji harmonics ---
def fig_beji():
    """Beji-Battjes harmonic amplitudes vs position: model vs digitized lab data."""
    p = os.path.join(RES, "beji_gauges.csv")
    if not os.path.exists(p):
        print("  (beji gauges missing, skipped)")
        return
    head, d = read_csv(p)
    t = d[:, 0]
    T, w = 2.02, 2 * np.pi / 2.02
    mask = t > t.max() - 10 * T
    tt = t[mask]
    gx = [5.7, 10.5, 12.5, 13.5, 14.5, 15.7, 17.3, 19.0]
    A = np.zeros((len(gx), 4))
    for gi in range(len(gx)):
        eta = d[mask, gi + 1]
        for n in range(1, 5):
            A[gi, n - 1] = 2 * abs(np.mean(eta * np.exp(-1j * n * w * tt))) * 100

    # Optional lab-data overlay: beji_lab_harmonics.csv with columns
    # x_m, A1_cm, A2_cm, A3_cm[, ...] (Dingemans 1994 digitization). Searched in
    # the results root and in beji_extracted_results/data/. Only the first four
    # numeric columns are used (extra text columns are ignored).
    lab = None
    for cand in (os.path.join(RES, "beji_lab_harmonics.csv"),
                 os.path.join(RES, "beji_extracted_results", "data",
                              "beji_lab_harmonics.csv")):
        if os.path.exists(cand):
            rows = []
            with open(cand) as f:
                next(f)
                for line in f:
                    p = line.split(",")
                    if len(p) >= 4 and p[0].strip():
                        rows.append([float(p[0]), float(p[1]), float(p[2]),
                                     float(p[3])])
            lab = np.array(rows)
            break

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    colors = ["C0", "C1", "C2"]
    for n in range(3):
        ax.plot(gx, A[:, n], "-o", color=colors[n], ms=4,
                label=f"model $A_{n+1}$")
        if lab is not None and lab.shape[1] > n + 1:
            ax.plot(lab[:, 0], lab[:, n + 1], "s", color=colors[n],
                    mfc="none", ms=6, label=f"lab $A_{n+1}$")
    # bar sketch (scaled into the lower part of the axes)
    def depth(xx):
        """Beji-Battjes bar profile H(x), identical to bejiDepth in apps/beji_battjes.cpp."""
        if xx < 6: return 0.40
        if xx < 12: return 0.40 - (xx - 6) / 20
        if xx < 14: return 0.10
        if xx < 17: return 0.10 + (xx - 14) / 10
        return 0.40
    xb = np.linspace(4, 21, 200)
    hb = np.array([depth(v) for v in xb])
    y0, ys = -0.35, 0.35
    ax.fill_between(xb, y0 + ys * (0.40 - hb), y0, color="0.85", zorder=0)
    ax.set_xlim(4, 21)
    ax.set_ylim(y0, None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("harmonic amplitude [cm]")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "fig_beji_harmonics")


# ----------------------------------------------------------- alpha sweep ----
def fig_alpha():
    """Celerity error and accepted steps vs the relaxation parameter alpha."""
    p = os.path.join(RES, "alpha_sweep.csv")
    if not os.path.exists(p):
        print("  (alpha_sweep missing, skipped)")
        return
    _, d = read_csv(p)
    al, err, drift, nst = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

    # accuracy: celerity error ~ 1/alpha^2
    a1.loglog(al, err, "o-", color="C0", label="celerity error")
    ref = err[3] * (al[3] / al) ** 2  # anchor at alpha=3 (index 3)
    a1.loglog(al, ref, "k--", lw=0.8, label=r"$\propto \alpha^{-2}$")
    a1.axvline(3.0, color="C3", lw=0.8, ls=":")
    a1.set_xlabel(r"relaxation parameter $\alpha$")
    a1.set_ylabel("relative celerity error")
    a1.set_title("accuracy", fontsize=9)
    a1.legend(frameon=False, fontsize=8)
    a1.set_xticks(al, [f"{v:g}" for v in al])

    # cost: number of accepted steps ~ linear in alpha
    a2.plot(al, nst, "s-", color="C1", label="accepted steps")
    lin = nst[3] * al / al[3]
    a2.plot(al, lin, "k--", lw=0.8, label=r"$\propto \alpha$")
    a2.axvline(3.0, color="C3", lw=0.8, ls=":", label=r"$\alpha=3$ (used)")
    a2.set_xlabel(r"relaxation parameter $\alpha$")
    a2.set_ylabel("accepted steps / transit")
    a2.set_title("cost", fontsize=9)
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_alpha")


# ---------------------------------------------------- validation figures ----
def fig_dispersion():
    """Analytic phase-speed error of the relaxed system vs Airy (no simulation input)."""
    # Pure-analytic: slow-branch phase-speed error vs Airy, per (gamma, alpha).
    # Mirrors sgn_dispersion.h (Escalante et al. eq. 15).
    def airy(kH):
        """Airy phase speed squared, Cp^2/gH = tanh(kH)/(kH)."""
        return np.tanh(kH) / kH
    def slow(kH, gamma, alpha):
        """Slow-branch phase speed squared of the relaxed system (as sgn::slowCp2)."""
        J = 1 + kH * kH / (2 * gamma)
        mu = (kH / alpha) ** 2 / (2 * gamma)
        half = 0.5 * (J + mu)
        disc = np.maximum(half * half - mu, 0.0)
        return 1.0 / (half + np.sqrt(disc))
    kH = np.linspace(0.05, 3.0, 200)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    styles = [(1.5, 3, "C0", "-"), (1.5, 5, "C0", "--"),
              (2.0, 3, "C1", "-"), (2.0, 5, "C1", "--")]
    for gamma, alpha, c, ls in styles:
        err = 100 * (np.sqrt(slow(kH, gamma, alpha) / airy(kH)) - 1)
        ax.plot(kH, err, ls, color=c, lw=1.2,
                label=rf"$\gamma={gamma}$, $\alpha={alpha}$")
    ax.axhline(0, color="k", lw=0.4)
    ax.fill_between([0, 3], -3, 3, color="0.9", zorder=0)  # +-3% band
    ax.set_xlim(0, 3)
    ax.set_ylim(-8, 5)
    ax.set_xlabel(r"$kH$")
    ax.set_ylabel(r"phase-speed error vs Airy [\%]".replace("\\%", "%"))
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("linear dispersion (slow branch)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_dispersion")


def fig_solitary():
    """Solitary wave before and after one periodic-domain transit."""
    p = os.path.join(RES, "val_solitary.csv")
    if not os.path.exists(p):
        print("  (val_solitary missing, skipped)"); return
    _, d = read_csv(p)
    x, e0, eN = d[:, 0], d[:, 1], d[:, 2]
    o = np.argsort(x)
    x, e0, eN = x[o], e0[o], eN[o]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.plot(x, e0, "-", color="0.6", lw=2.5, label="initial")
    ax.plot(x, eN, "--", color="C0", lw=1.2, label="after one transit")
    ax.set_xlabel("x [m]"); ax.set_ylabel(r"$\eta$ [m]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("solitary wave: permanent form", fontsize=9)
    fig.tight_layout(); save(fig, "fig_solitary")


def fig_collision():
    """Head-on collision of two equal solitons, five snapshots."""
    p = os.path.join(RES, "val_collision.csv")
    if not os.path.exists(p):
        print("  (val_collision missing, skipped)"); return
    _, d = read_csv(p)
    o = np.argsort(d[:, 0]); d = d[o]
    x = d[:, 0]
    labels = [r"$t=0$", r"$0.6\,t_c$", r"$t_c$ (impact)", r"$1.4\,t_c$",
              r"$2\,t_c$"]
    fig, axes = plt.subplots(5, 1, figsize=(5.2, 4.6), sharex=True)
    for s, ax in enumerate(axes):
        ax.plot(x, d[:, s + 1], color=plt.cm.viridis(s / 4), lw=1.0)
        ax.axhline(0, color="k", lw=0.3)
        ax.set_ylabel(labels[s], fontsize=7.5, rotation=0, ha="right", va="center")
        ax.set_ylim(-0.05, 0.45)
    axes[-1].set_xlabel("x [m]")
    axes[0].set_title(r"head-on collision ($\eta$, m)", fontsize=9)
    fig.tight_layout(); save(fig, "fig_collision")


def fig_wb():
    """Lake at rest over a bump: max |eta| and max |RHS| vs time."""
    p = os.path.join(RES, "val_wb.csv")
    pb = os.path.join(RES, "val_wb_bathy.csv")
    if not os.path.exists(p):
        print("  (val_wb missing, skipped)"); return
    _, d = read_csv(p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    if os.path.exists(pb):
        _, b = read_csv(pb)
        o = np.argsort(b[:, 0]); xb, Hb = b[o, 0], b[o, 1]
        a1.fill_between(xb, -Hb, -1.05, color="0.8", label="bed")
        a1.axhline(0, color="C0", lw=1.5, label="free surface (at rest)")
        a1.set_xlabel("x [m]"); a1.set_ylabel("z [m]")
        a1.set_ylim(-1.05, 0.15); a1.legend(frameon=False, fontsize=7)
        a1.set_title("lake at rest over bump", fontsize=9)
    a2.semilogy(d[:, 0], np.maximum(d[:, 1], 1e-18), "o-", color="C0", ms=3,
                label=r"$\max|\eta - \eta_{\rm rest}|$")
    a2.semilogy(d[:, 0], np.maximum(d[:, 2], 1e-18), "s-", color="C3", ms=3,
                label=r"$\max|\mathrm{RHS}|$")
    a2.axhline(2.2e-16, color="k", ls=":", lw=0.8, label="machine eps")
    a2.set_xlabel("t [s]"); a2.set_ylabel("residual")
    a2.set_ylim(1e-18, 1e-10)
    a2.legend(frameon=False, fontsize=7)
    a2.set_title("well-balancing preserved", fontsize=9)
    fig.tight_layout(); save(fig, "fig_wb")


def fig_collision_uneq():
    """Head-on collision of two unequal solitons, five snapshots."""
    p = os.path.join(RES, "val_collision_uneq.csv")
    if not os.path.exists(p):
        print("  (val_collision_uneq missing, skipped)"); return
    _, d = read_csv(p)
    o = np.argsort(d[:, 0]); d = d[o]
    x = d[:, 0]
    labels = [r"$t=0$", r"$0.6\,t_c$", r"$t_c$ (impact)", r"$1.4\,t_c$",
              r"$2\,t_c$"]
    fig, axes = plt.subplots(5, 1, figsize=(5.2, 4.6), sharex=True)
    for s, ax in enumerate(axes):
        ax.plot(x, d[:, s + 1], color=plt.cm.plasma(s / 4), lw=1.0)
        ax.axhline(0, color="k", lw=0.3)
        ax.set_ylabel(labels[s], fontsize=7.5, rotation=0, ha="right", va="center")
        ax.set_ylim(-0.06, 0.42)
    axes[-1].set_xlabel("x [m]")
    axes[0].set_title(r"unequal collision ($a_R{=}0.25$, $a_L{=}0.10$; $\eta$, m)",
                      fontsize=9)
    fig.tight_layout(); save(fig, "fig_collision_uneq")


def fig_rkcompare():
    """Celerity error vs RHS evaluations: fixed RK4 vs adaptive BS3(2)."""
    p = os.path.join(RES, "rk_compare.csv")
    if not os.path.exists(p):
        print("  (rk_compare missing, skipped)"); return
    rows = [l.strip().split(",") for l in open(p) if l.strip()][1:]
    rk = [(float(r[2]), float(r[3])) for r in rows
          if r[0] == "rk4" and r[2] != "nan"]
    ad = [(float(r[2]), float(r[3])) for r in rows if r[0] == "adaptive"]
    if not rk or not ad:
        print("  (rk_compare data incomplete, skipped)"); return
    rk.sort(); ad.sort()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.loglog([e for e, _ in rk], [n for _, n in rk], "o-", color="C1",
              label="fixed RK4 (sweep $\\Delta t$)")
    ax.loglog([e for e, _ in ad], [n for _, n in ad], "s-", color="C0",
              label="adaptive BS3(2) (sweep tol)")
    ax.set_xlabel("relative celerity error")
    ax.set_ylabel("RHS evaluations / transit")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("time-integration cost vs accuracy", fontsize=9)
    ax.invert_xaxis()  # better accuracy to the right
    fig.tight_layout(); save(fig, "fig_rkcompare")


def fig_convergence():
    """Manufactured-solution L2 error of h vs mesh size, 1D and 2D, P = 1..4."""
    import matplotlib.ticker as mticker
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, fn, dim in ((axes[0], "convergence1d.csv", "1D"),
                        (axes[1], "convergence2d.csv", "2D")):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            print(f"  ({fn} missing, skipped)"); continue
        hdr, d = read_csv(p)
        # columns by name: convergence1d.csv also carries dof, steps, wall
        # time, ... (P,ne,dof,steps,hL2err,...), convergence2d.csv is P,N,hL2err
        cP, cN = hdr.index("P"), hdr.index("ne" if "ne" in hdr else "N")
        cE = hdr.index("hL2err")
        for P in (1, 2, 3, 4):
            m = d[d[:, cP] == P]
            if len(m) == 0:
                continue
            h = 1.0 / m[:, cN]           # ~ mesh spacing
            err = m[:, cE]
            ax.loglog(h, err, "o-", ms=4, label=f"$P={P}$")
            order = P if P % 2 == 0 else P + 1
            ax.loglog(h, err[-1] * (h / h[-1]) ** order, "k--", lw=0.6)
        ax.set_xlabel(r"mesh spacing $\Delta x$")
        ax.set_title(f"{dim} MMS convergence", fontsize=9)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
        # clean decade ticks only, no minor-tick label clutter
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", labelbottom=False)
    axes[0].set_ylabel(r"$L^2$ error of $h$")
    fig.tight_layout()
    save(fig, "fig_convergence")


print("figures ->", OUT)
fig_scaling()
fig_alpha()
fig_convergence()
fig_dispersion()
fig_solitary()
fig_collision()
fig_collision_uneq()
fig_wb()
fig_rkcompare()
fig_bolles_stats()
fig_bolles_hist()
fig_beji()
print("done")
