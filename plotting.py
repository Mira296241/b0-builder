"""Figure styling, kept separate so the look can be changed in one place."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK   = "#1d2430"
MUTED = "#6b7785"
GRID  = "#dfe4ea"

def use_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": 11,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.9,
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "text.color": INK,
    })

# smooth diverging map, a little softer than RdBu_r
FIELD_CMAP = LinearSegmentedColormap.from_list("field", [
    "#20325c", "#2e6ca8", "#6fb0d8", "#c9e3f0",
    "#f7f7f5",
    "#fbdfc2", "#f0a05a", "#d0512c", "#7d1d12"])

MAT_COLOURS = {"glass": "#9aa7b8", "liquid": "#f0b05a",
               "outside": "#eef1f4", "coil": "#c0703a", "other": "#c8ced6"}


def _tip_line(ax, x, label=True):
    """z = 0 is the tube tip, the datum every height in the app is measured
    from. Drawing it costs one line and removes the main way a reader can
    misread these panels."""
    ax.axhline(0.0, color=INK, lw=1.0, alpha=0.55)
    if label:
        ax.text(x[0] + 0.04*(x[-1]-x[0]), 0.0, " tube tip", color=INK,
                fontsize=8, alpha=0.75, va="bottom", ha="left")


def geometry_panel(ax, x, z, chi, glass_m, liquid_m, jslice, coil_m=None,
                   act=None, r_act=None, z1=None, z2=None):
    img = np.zeros(chi[:, jslice, :].shape + (3,))
    def fill(mask, hexcol):
        c = np.array(mpl.colors.to_rgb(hexcol))
        img[mask] = c
    g = glass_m[:, jslice, :]
    l = liquid_m[:, jslice, :]
    fill(np.ones_like(g, bool), MAT_COLOURS["outside"])
    fill(g, MAT_COLOURS["glass"])
    fill(l, MAT_COLOURS["liquid"])
    if coil_m is not None and coil_m.any():
        fill(coil_m[:, jslice, :], MAT_COLOURS["coil"])
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", aspect="equal",
              extent=[x[0], x[-1], z[0], z[-1]], interpolation="nearest")
    if z1 is not None:
        ax.add_patch(plt.Rectangle((-r_act, z1), 2*r_act, z2-z1, fill=False,
                                   ec=INK, lw=1.4, ls=(0, (4, 3))))
    _tip_line(ax, x)
    ax.set_title("geometry")
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("height above tube tip  (mm)")


def field_panel(fig, ax, x, z, dB, liquid_m, jslice, ref,
                r_act, z1, z2, lim):
    d = np.where(liquid_m, dB - ref, np.nan)[:, jslice, :]
    m = ax.imshow(d.T, origin="lower", aspect="equal", cmap=FIELD_CMAP,
                  vmin=-lim, vmax=lim, interpolation="bilinear",
                  extent=[x[0], x[-1], z[0], z[-1]])
    ax.add_patch(plt.Rectangle((-r_act, z1), 2*r_act, z2-z1, fill=False,
                               ec=INK, lw=1.4, ls=(0, (4, 3))))
    cb = fig.colorbar(m, ax=ax, fraction=0.05, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, width=0.9, colors=MUTED, labelsize=9)
    cb.set_label("$\\Delta B_0/B_0$  (ppm)", color=MUTED, fontsize=9.5)
    _tip_line(ax, x, label=False)
    ax.set_title("field in the sample")
    ax.set_xlabel("x  (mm)")


def hist_panel(ax, vals, f_MHz):
    v = vals - vals.mean()
    lo, hi = np.percentile(v, [0.3, 99.7])
    pad = 0.08*(hi-lo+1e-9)
    ax.hist(np.clip(v, lo, hi), bins=64, range=(lo-pad, hi+pad),
            color="#3d7ab8", edgecolor="none")
    ax.set_xlim(lo-pad, hi+pad)
    ax.axvline(0, color=MUTED, lw=0.9, ls="--")
    ax.set_title("spread over the active volume")
    ax.set_xlabel("$\\Delta B_0/B_0$  (ppm, mean removed)")
    ax.set_ylabel("voxels")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    sec = ax.secondary_xaxis("top", functions=(lambda p: p*f_MHz,
                                               lambda h: h/f_MHz))
    sec.set_xlabel("Hz", color=MUTED, fontsize=9.5)
    sec.tick_params(colors=MUTED, labelsize=9, length=3, width=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
