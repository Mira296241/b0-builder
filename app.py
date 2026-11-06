"""
B0 homogeneity explorer for NMR / DNP sample tubes.
Run with:   python -m streamlit run app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from b0core import (MATERIALS_PPM, SOURCES, grid, build_tube, build_coil,
                    field_shift_ppm, report, scan_windows, shim, shim_coef,
                    build_rod, build_cwg, build_annulus, lineshape_report, region_basis, breakdown,
                    match_region, residual_with, nearest_materials)
import thz as TZ
import plotting as P
P.use_style()

st.set_page_config(page_title="B0 tube explorer", layout="wide")
st.title("B₀ homogeneity of a sample tube")
st.caption("Susceptibility-induced field distortion, k-space method. "
           "Validated against the sphere and the parallel and perpendicular "
           "infinite cylinder.")

c = st.sidebar
c.header("Tube")
c.caption("Defaults = FRASCAN II probe (Fusion CAD, 30 Oct 2026): 5 mm tube, "
          "slotted-tube coil, CWG, tube tip 0.5 mm above the CWG.")
tube_OD = c.slider("outer diameter (mm)", 2.0, 8.0, 5.0, 0.1)
wall    = c.slider("wall thickness (mm)", 0.1, 1.5, 0.5, 0.05)
d_cwg   = c.slider("tube tip above the launcher (mm)", 0.0, 15.0, 0.5, 0.5,
                   help="The only launcher-referenced number in the app. "
                        "Every other height is measured from the tube tip.")
fill_h  = c.slider("liquid column (mm)", 5.0, 50.0, 45.0, 1.0)

c.header("Bottom")
bottom = c.radio("shape", ["round", "flat", "punt"], horizontal=True)
bot_depth = c.slider("round-bottom depth (mm)", 0.5, 6.0, 2.0, 0.1,
                     disabled=(bottom != "round"))
punt_r = c.slider("punt radius (mm)", 0.5, 3.0, 2.0, 0.1,
                  disabled=(bottom != "punt"))

c.header("Liquid surface")
men_depth = c.slider("meniscus dip on axis (mm)", 0.0, 3.0, 0.0, 0.1,
                     help="0 = flat. Larger values = more concave.")

c.header("Insert")
ins_kind = c.radio("type", ["none", "ball", "plug"], horizontal=True)
ins_r    = c.slider("ball radius / plug radius (mm)", 0.3, 2.0, 1.25, 0.05,
                    disabled=(ins_kind == "none"))
ins_h    = c.slider("plug height (mm)", 0.5, 12.0, 4.0, 0.25,
                    disabled=(ins_kind != "plug"))
ins_d    = c.slider("plug dome height (mm)", 0.0, 4.0, 2.0, 0.1,
                    disabled=(ins_kind != "plug"))

c.header("Coaxial inner tube")
inner_on = c.checkbox("include an inner tube", value=False)
in_OD    = c.slider("inner tube OD (mm)", 0.5, 3.5, 2.8, 0.05, disabled=not inner_on)
in_wall  = c.slider("inner tube wall (mm)", 0.05, 0.8, 0.35, 0.05, disabled=not inner_on)
in_z0    = c.slider("inner tube starts at (mm above tip)", -5.0, 35.0, 7.0, 0.5,
                    disabled=not inner_on)

c.header("RF coil")
coil_on = c.checkbox("include the coil", value=True)
coil_r      = c.slider("inner radius (mm)", 2.5, 10.0, 5.0, 0.1, disabled=not coil_on)
coil_t      = c.slider("foil thickness (µm)", 5.0, 500.0, 100.0, 5.0, disabled=not coil_on)
coil_len    = c.slider("length (mm)", 2.0, 40.0, 39.0, 0.5, disabled=not coil_on)
lock_on     = c.checkbox("coil fixed to the waveguide", value=True, disabled=not coil_on,
                         help="Like the real probe: the coil and the waveguide are "
                              "one assembly. Moving the tube (the gap slider) then "
                              "moves the coil and the active volume with it.")
D_win       = c.slider("window bottom above CWG top (mm)", 0.0, 30.0, 7.5, 0.1,
                       disabled=not (coil_on and lock_on),
                       help="Measured in the FRASCAN II CAD: 330.60 - 323.10 = 7.50 mm.")
coil_zc     = c.slider("centre (mm above tip)", -5.0, 40.0, 14.5, 0.5,
                       disabled=not coil_on or lock_on)
n_slots     = c.slider("number of slots", 0, 8, 2, 1, disabled=not coil_on)
slot_deg    = c.slider("slot width (deg)", 0.0, 90.0, 88.0, 1.0, disabled=not coil_on)
slot_phase  = c.slider("slot angular offset (deg)", 0.0, 180.0, 20.0, 5.0, disabled=not coil_on)
win_on      = c.checkbox("windows + slit (FRASCAN II shape)", value=True, disabled=not coil_on,
                         help="Off: slots run the full coil length. On: each slot is a "
                              "window of the width above, fed by a narrow slit from the "
                              "coil bottom, like the real FRASCAN II coil.")
win_z0      = c.slider("window starts above coil bottom (mm)", 0.0, 30.0, 12.0, 0.5,
                       disabled=not (coil_on and win_on))
win_h       = c.slider("window height (mm)", 2.0, 30.0, 15.0, 0.5,
                       disabled=not (coil_on and win_on))
slit_mm     = c.slider("slit width (mm)", 0.2, 3.0, 1.2, 0.1,
                       disabled=not (coil_on and win_on))

c.header("Waveguide (CWG) below the tube")
cwg_on   = c.checkbox("include the waveguide end", value=True,
                      help="The metal waveguide that brings the THz beam up "
                           "to the tube. Its gap to the tube tip is the "
                           "'tube tip above the launcher' slider.")
cwg_id   = c.slider("CWG inner diameter (mm)", 2.0, 10.0, 5.0, 0.1, disabled=not cwg_on)
cwg_wall = c.slider("CWG wall (mm)", 0.2, 3.0, 1.25, 0.05, disabled=not cwg_on)
cwg_len  = c.slider("CWG length modelled (mm)", 5.0, 40.0, 20.0, 1.0, disabled=not cwg_on,
                    help="Only the top end matters; 20 mm is plenty.")
cwg_mat  = c.selectbox("CWG metal", list(MATERIALS_PPM),
                       index=list(MATERIALS_PPM).index("brass"), disabled=not cwg_on)

c.header("Probe-head parts (FRASCAN II CAD)")
c.caption("Heights above the waveguide top. All copper.")
bs_on = c.checkbox("bottom RF shield (disc, r 8.0-17.9, 0.19 mm)", value=True)
ss_on = c.checkbox("side RF shield (r 9.0-9.1)", value=True)
ss_z0 = c.slider("side shield bottom (mm)", -10.0, 40.0, 4.95, 0.05, disabled=not ss_on)
ss_z1 = c.slider("side shield top (mm)", -10.0, 60.0, 25.05, 0.05, disabled=not ss_on)
mc_on = c.checkbox("modulation coil (r 9.1-12.7, ~75 % copper)", value=True)
mc_z0 = c.slider("modulation coil bottom (mm)", -15.0, 40.0, 5.0, 0.05, disabled=not mc_on)
mc_z1 = c.slider("modulation coil top (mm)", -10.0, 60.0, 24.55, 0.05, disabled=not mc_on)
mc_fill = c.slider("modulation coil copper fill", 0.3, 1.0, 0.75, 0.05, disabled=not mc_on)

c.header("Materials")
mats = list(MATERIALS_PPM) + ["custom…"]
def pick(label, default):
    m = c.selectbox(label, mats, index=mats.index(default))
    if m == "custom…":
        v = c.number_input(f"{label}: χ (ppm)", -60.0, 60.0, 0.0, 0.1,
                           key=f"chi_{label}")
        MATERIALS_PPM["custom…"] = float(v)
    return m
glass   = pick("tube", "fused quartz")
liquid  = pick("sample", "toluene")
outside = pick("surroundings", "air")
ins_mat  = c.selectbox("insert material", list(MATERIALS_PPM),
                       index=list(MATERIALS_PPM).index("fused quartz"),
                       disabled=(ins_kind == "none"))
in_mat   = c.selectbox("inner tube material", list(MATERIALS_PPM),
                       index=list(MATERIALS_PPM).index("fused quartz"),
                       disabled=not inner_on)
for _lab, _m in (("tube", glass), ("sample", liquid)):
    _k, _src = SOURCES.get(_m, ("custom", "entered by you"))
    c.caption(f"{_lab}: χ = {MATERIALS_PPM[_m]:+.2f} ppm, **{_k}**. {_src}")
coil_mat = c.selectbox("coil metal", list(MATERIALS_PPM),
                       index=list(MATERIALS_PPM).index("copper"),
                       disabled=not coil_on)

c.header("Active volume")
c.caption("Heights above the **tube tip**, like everything else in the app.")
_locked = coil_on and lock_on and win_on
h1 = c.slider("bottom (mm above tip)", -5.0, 35.0, 7.0, 0.5, disabled=_locked)
h2 = c.slider("top (mm above tip)", -5.0, 45.0, 22.0, 0.5, disabled=_locked)
if _locked:
    # everything follows the gap: window bottom sits D_win above the CWG top,
    # i.e. D_win - d_cwg above the tube tip; the coil centre follows from it
    h1 = D_win - d_cwg
    h2 = h1 + win_h
    coil_zc = h1 - win_z0 + coil_len / 2
    c.info(f"Coil fixed to the waveguide. Active volume = the window: "
           f"**{h1:.1f} to {h2:.1f} mm** above the tube tip; coil centre "
           f"**{coil_zc:.1f} mm**. Move the tube with 'tube tip above the "
           f"launcher' only.")
r_act = c.slider("radius (mm)", 0.2, 4.0, 1.5, 0.1)

c.header("Numerics")
V = c.select_slider("voxel (mm)", [0.4, 0.3, 0.25, 0.2, 0.15], value=0.25)
shim_order = c.select_slider("shim order removed", [0, 1, 2], value=2)
f_MHz = c.number_input("¹H frequency (MHz)", 100.0, 1200.0, 500.0, 10.0)


# ----------------------------------------------------------------------
# ONE convention for everything the user sees: height above the TUBE TIP.
# b0core and thz.beam_through_bottom are built in launcher coordinates,
# where the tip sits at d_cwg, because the beam has to propagate that gap
# and build_tube has to place the tube somewhere. So the conversion lives
# here, at the boundary, and nothing below the UI knows about it.
#
# thz.axial_profile is the exception: it already returns height above the
# tube tip, so it is plotted raw. That mismatch, against a window drawn in
# launcher z, was the real bug.
# ----------------------------------------------------------------------
ZTIP = d_cwg


def from_tip(h):
    """user height above the tube tip -> internal z"""
    return np.asarray(h, float) + ZTIP if np.ndim(h) else float(h) + ZTIP


def to_tip(z):
    """internal z -> height above the tube tip"""
    return np.asarray(z, float) - ZTIP if np.ndim(z) else float(z) - ZTIP


z1, z2 = from_tip(h1), from_tip(h2)


def field_for(V, kw, coilkw):
    """Build the susceptibility map and solve the field. The expensive step."""
    half = max(6.0, (coilkw["coil_r"] + 2.0) if coilkw else 0.0) + kw["tube_OD"]/2
    if cwgkw:
        half = max(half, cwgkw["inner_d"]/2 + cwgkw["wall"] + 4.0)
    zlo, zhi = -3.0, kw["d_cwg"] + kw["fill_h"] + 22.0
    if cwgkw:
        zlo = -cwgkw["length"] - 3.0
    x, y, z = grid(half, zlo, zhi, V)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    chi, liq, gls = build_tube(x, y, z, **kw)
    cmask = np.zeros_like(liq)
    if coilkw:
        cchi, cmask = build_coil(x, y, z, **coilkw)
        chi = np.where(cmask, cchi, chi)
    if partskw:
        half = max(half, 15.0)
        x, y, z = grid(half, zlo, zhi, V)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        chi, liq, gls = build_tube(x, y, z, **kw)
        cmask = np.zeros_like(liq)
        if coilkw:
            cchi, cmask = build_coil(x, y, z, **coilkw)
            chi = np.where(cmask, cchi, chi)
        for p in partskw:
            pchi, pm = build_annulus(x, y, z, outside=kw["outside"], **p)
            chi = chi + pchi
            cmask = cmask | pm
    if cwgkw:
        wchi, wmask = build_cwg(x, y, z, outside=kw["outside"], **cwgkw)
        chi = chi + wchi
        cmask = cmask | wmask
    dB = field_shift_ppm(chi, (V, V, V), pad=2)
    return x, y, z, X, Y, Z, chi, dB, liq, gls, cmask


def solve(V, kw, coilkw, r_act, z1, z2, shim_order, f_MHz):
    x, y, z, X, Y, Z, chi, dB, liq, gls, cmask = field_for(V, kw, coilkw)
    R = np.sqrt(X**2 + Y**2)
    act = (R <= r_act) & (Z >= z1) & (Z <= z2) & liq
    if act.sum() < 50:
        return None
    rep = report(dB, act, X, Y, Z, f_MHz=f_MHz, order=shim_order)
    return x, y, z, chi, dB, act, rep, liq, gls, cmask


insert = None
if ins_kind == "ball":
    insert = dict(kind="ball", radius=ins_r, material=ins_mat)
elif ins_kind == "plug":
    insert = dict(kind="plug", radius=ins_r, height=ins_h, dome=ins_d,
                  material=ins_mat)
inner_tube = dict(OD=in_OD, wall=in_wall, z_bottom=from_tip(in_z0),
                  material=in_mat) if inner_on else None

kw = dict(tube_OD=tube_OD, wall=wall, d_cwg=d_cwg, fill_h=fill_h,
          bottom=bottom, punt_r=punt_r, bot_depth=bot_depth,
          glass=glass, liquid=liquid, outside=outside,
          men_depth=men_depth, insert=insert, inner_tube=inner_tube)
partskw = []
if bs_on: partskw.append(dict(r_in=8.0, r_out=17.925, z0=-0.74, z1=-0.55))
if ss_on: partskw.append(dict(r_in=9.0, r_out=9.1, z0=ss_z0, z1=ss_z1))
if mc_on: partskw.append(dict(r_in=9.1, r_out=12.7, z0=mc_z0, z1=mc_z1, fill=mc_fill))
cwgkw = dict(inner_d=cwg_id, wall=cwg_wall, length=cwg_len,
             material=cwg_mat) if cwg_on else None
coilkw = dict(coil_r=coil_r, thickness_um=coil_t, length=coil_len,
              z_centre=from_tip(coil_zc), n_slots=n_slots, slot_deg=slot_deg,
              slot_phase=slot_phase, material=coil_mat,
              window=(dict(win_z0=win_z0, win_h=win_h, slit_mm=slit_mm)
                      if win_on else None)) if coil_on else None

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Single design", "Sweep", "Recommend", "Microwave (1D)",
     "Lineshape & matching"])

# ------------------------------------------------------------------ single
with tab1:
    with st.spinner("solving…"):
        res = solve(V, kw, coilkw, r_act, z1, z2, shim_order, f_MHz)
    if res is None:
        st.error("Active volume contains no sample. Check the z range and radius.")
        st.stop()
    x, y, z, chi, dB, act, rep, liq, gls, cmask = res
    j = len(y)//2

    k = st.columns(4)
    k[0].metric("raw spread (p-p)", f"{rep['raw_ppm_pp']:.3f} ppm")
    k[1].metric(f"after shim order {shim_order}", f"{rep['shim_ppm_pp']:.4f} ppm")
    k[2].metric("residual rms", f"{rep['shim_ppm_rms']:.4f} ppm")
    k[3].metric("linewidth contribution", f"{rep['shim_Hz_rms']:.1f} Hz")

    target = 0.04
    if rep["shim_ppm_rms"] <= target:
        st.success(f"Residual {rep['shim_ppm_rms']:.3f} ppm. "
                   f"Within the ~{target} ppm needed for 2D spectroscopy.")
    else:
        st.warning(f"Residual {rep['shim_ppm_rms']:.3f} ppm, above the ~{target} ppm "
                   f"rule of thumb for useful 2D spectra.")

    lim = max(1e-4, np.percentile(np.abs(dB[act] - dB[act].mean()), 99) * 1.4)
    ztop = to_tip(min(z[-1], max(z2 + 8.0, d_cwg + fill_h + 4.0)))
    zbot = to_tip(max(z[0], -4.0 if cwg_on else d_cwg - 4.0))
    z_t = to_tip(z)
    xlim = max((coil_r + 1.0) if coil_on else (tube_OD/2 + 1.6),
               (cwg_id/2 + cwg_wall + 1.0) if cwg_on else 0.0,
               (10.0 if (ss_on or mc_on) else 0.0))

    fig, ax = plt.subplots(1, 3, figsize=(12.4, 5.3),
                           gridspec_kw={"width_ratios": [0.62, 0.84, 2.1]})
    P.geometry_panel(ax[0], x, z_t, chi, gls, liq, j, coil_m=cmask,
                     r_act=r_act, z1=h1, z2=h2)
    P.field_panel(fig, ax[1], x, z_t, dB, liq, j, dB[act].mean(),
                  r_act, h1, h2, lim)
    P.hist_panel(ax[2], dB[act], f_MHz)
    for a in ax[:2]:
        a.set_ylim(zbot, ztop); a.set_xlim(-xlim, xlim)
    fig.tight_layout(w_pad=2.2)
    st.pyplot(fig)

    if coil_on:
        st.caption("Coil included as a thin shell: real foil thickness is below "
                   "the voxel size, so χ is scaled to preserve χ × thickness.")
    else:
        st.caption("Coil not included. Copper this close to the sample usually "
                   "dominates: tick the box in the sidebar.")

# ------------------------------------------------------------------- sweep
with tab2:
    st.write("Vary one parameter and plot the residual. "
             "Everything else comes from the sidebar.")

    PARAMS = {
        "active volume position (mm above tip)": ("zpos", 0.0, 25.0),
        "wall thickness (mm)":         ("wall", 0.15, 1.2),
        "punt radius (mm)":            ("punt_r", 0.5, 3.0),
        "round-bottom depth (mm)":     ("bot_depth", 0.5, 6.0),
        "tube outer diameter (mm)":    ("tube_OD", 2.5, 7.0),
        "liquid column (mm)":          ("fill_h", 8.0, 45.0),
        "coil inner radius (mm)":      ("coil_r", 2.5, 9.0),
        "coil length (mm)":            ("coil_len", 3.0, 35.0),
        "meniscus dip (mm)":           ("men_depth", 0.0, 3.0),
        "insert radius (mm)":          ("ins_r", 0.3, 1.45),
        "inner tube start (mm above tip)": ("in_z0", 0.0, 25.0),
    }
    cc = st.columns([2, 1, 1, 1, 1])
    pname = cc[0].selectbox("parameter", list(PARAMS))
    key, lo0, hi0 = PARAMS[pname]
    lo = cc[1].number_input("from", value=float(lo0), step=0.25, key="swlo")
    hi = cc[2].number_input("to", value=float(hi0), step=0.25, key="swhi")
    n  = cc[3].number_input("points", 3, 25, 9, 1, key="swn")
    Vs = cc[4].select_slider("voxel", [0.4, 0.3, 0.25, 0.2], value=0.3, key="swv")

    if key in ("coil_r", "coil_len") and not coil_on:
        st.info("Tick 'include the coil' in the sidebar to sweep a coil parameter.")
    elif st.button("Run sweep", type="primary"):
        vals = np.linspace(lo, hi, int(n))
        rms, hz = [], []
        bar = st.progress(0.0, text="solving…")
        for i, v in enumerate(vals):
            k2, c2 = dict(kw), (dict(coilkw) if coilkw else None)
            a1, a2 = z1, z2
            if key == "zpos":
                span = z2 - z1
                a1, a2 = from_tip(v), from_tip(v) + span
            elif key in ("coil_r", "coil_len"):
                c2["coil_r" if key == "coil_r" else "length"] = float(v)
            elif key == "ins_r":
                if not k2.get("insert"):
                    st.warning("Choose an insert type in the sidebar first.")
                    st.stop()
                k2["insert"] = dict(k2["insert"], radius=float(v))
            elif key == "in_z0":
                if not k2.get("inner_tube"):
                    st.warning("Tick 'include an inner tube' first.")
                    st.stop()
                k2["inner_tube"] = dict(k2["inner_tube"],
                                        z_bottom=from_tip(float(v)))
            else:
                k2[key] = float(v)
            r = solve(Vs, k2, c2, r_act, a1, a2, shim_order, f_MHz)
            rms.append(np.nan if r is None else r[6]["shim_ppm_rms"])
            hz.append(np.nan if r is None else r[6]["shim_Hz_rms"])
            bar.progress((i+1)/len(vals), text=f"{pname} = {v:.2f}")
        bar.empty()

        fig2, ax2 = plt.subplots(figsize=(8.2, 4.4))
        ax2.plot(vals, rms, "o-", color="#2f6fb5", lw=2.2, ms=7)
        ax2.axhline(0.04, color="#1f8a4c", ls="--", lw=1.4)
        ax2.text(vals[-1], 0.041, " 2D target", color="#1f8a4c",
                 fontsize=9, ha="right", va="bottom")
        ax2.set_xlabel(pname); ax2.set_ylabel("residual rms  (ppm)")
        ax2.set_title(f"B₀ residual vs {pname}")
        ax2.grid(color=P.GRID, lw=0.8); ax2.set_axisbelow(True)
        sec = ax2.secondary_yaxis("right", functions=(lambda p: p*f_MHz,
                                                      lambda h: h/f_MHz))
        sec.set_ylabel("Hz", color=P.MUTED, fontsize=9.5)
        sec.tick_params(colors=P.MUTED, labelsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)

        best = int(np.nanargmin(rms))
        st.success(f"Best: {pname} = {vals[best]:.2f}  →  "
                   f"{rms[best]:.4f} ppm  ({hz[best]:.1f} Hz)")
        st.download_button("Download CSV",
            "value,rms_ppm,rms_Hz\n" + "\n".join(
                f"{v:.4f},{r:.6f},{h:.3f}" for v, r, h in zip(vals, rms, hz)),
            file_name="b0_sweep.csv", mime="text/csv")

# --------------------------------------------------------------- recommend
with tab3:
    st.write("Pick your sample and how much of it you have. The tool tries a "
             "set of bottom shapes, slides the detection window along each "
             "one, and reports the combination with the lowest residual.")
    st.caption("This optimises **B₀ only**. Microwave B₁ is not modelled here: "
               "it comes from the HFSS side and pulls the detection volume the "
               "other way, downwards towards the beam. Treat the answer as one "
               "half of the trade-off, not the final design.")

    g = st.columns([1, 1, 1, 1])
    vol_uL = g[0].number_input("sample volume (µL)", 5.0, 500.0, 100.0, 5.0)
    target = g[1].number_input("target residual (ppm)", 0.005, 1.0, 0.04, 0.005,
                               format="%.3f")
    Vr     = g[2].select_slider("voxel (mm)", [0.4, 0.3, 0.25], value=0.3,
                                key="recv")
    step   = g[3].select_slider("window step (mm)", [1.0, 0.5, 0.25],
                                value=0.5, key="recs")

    shapes = st.multiselect(
        "bottom shapes to try",
        ["round 1.5", "round 2.0", "round 3.0", "flat",
         "punt 1.5", "punt 1.75", "punt 2.0", "punt 2.5"],
        default=["round 2.0", "flat", "punt 1.75", "punt 2.0"])

    # the detection window length follows from the volume and the active radius
    area = np.pi * r_act**2                       # mm^2
    win_len = (vol_uL) / area                     # 1 uL = 1 mm^3
    st.caption(f"At active radius {r_act:.1f} mm that volume is a column "
               f"**{win_len:.1f} mm** long. Change the radius in the sidebar "
               f"to trade length for width.")

    if win_len > fill_h:
        st.error(f"A {win_len:.1f} mm column does not fit in a {fill_h:.0f} mm "
                 f"fill. Raise the liquid column or the active radius.")
    elif not shapes:
        st.info("Choose at least one bottom shape.")
    elif st.button("Find the best setup", type="primary", key="recbtn"):
        rows, curves = [], {}
        bar = st.progress(0.0, text="solving…")
        for i, s in enumerate(shapes):
            k2 = dict(kw)
            k2["insert"] = None            # inserts always hurt: leave them out
            k2["inner_tube"] = None
            if s == "flat":
                k2["bottom"] = "flat"
            elif s.startswith("round"):
                k2["bottom"] = "round"; k2["bot_depth"] = float(s.split()[1])
            else:
                k2["bottom"] = "punt";  k2["punt_r"]   = float(s.split()[1])
            bar.progress(i/len(shapes), text=f"{s}…")
            try:
                _, _, _, X, Y, Z, _, dB, liq, _, _ = field_for(Vr, k2, coilkw)
            except Exception as e:                      # geometry that cannot build
                st.warning(f"{s}: could not build ({e})")
                continue
            w = scan_windows(dB, X, Y, Z, liq, r_act, win_len,
                             step=step, order=shim_order, f_MHz=f_MHz)
            if not w:
                st.warning(f"{s}: the window never fits inside the liquid.")
                continue
            # scan_windows works in internal z; everything shown is tip-relative
            curves[s] = [(to_tip(a), to_tip(b2), r, h) for a, b2, r, h in w]
            z1b, z2b, rms, hz = min(curves[s], key=lambda t: t[2])
            rows.append(dict(shape=s, z1=z1b, z2=z2b, rms=rms, hz=hz))
        bar.empty()

        if not rows:
            st.error("Nothing solved. Loosen the geometry and try again.")
            st.stop()

        rows.sort(key=lambda r: r["rms"])
        b = rows[0]

        st.subheader("Recommended setup")
        m = st.columns(4)
        m[0].metric("bottom", b["shape"])
        m[1].metric("detection window",
                    f"{b['z1']:.1f} – {b['z2']:.1f} mm above tip")
        m[2].metric("residual rms", f"{b['rms']:.4f} ppm")
        m[3].metric("linewidth", f"{b['hz']:.1f} Hz")

        ok = b["rms"] <= target
        (st.success if ok else st.warning)(
            f"{'Meets' if ok else 'Misses'} the {target:g} ppm target"
            + ("." if ok else ". Try a longer fill, a smaller sample volume, "
                            "or a less susceptible tube material."))

        st.markdown(
            f"- tube **{glass}**, sample **{liquid}**, surroundings **{outside}**\n"
            f"- OD **{tube_OD:.1f} mm**, wall **{wall:.2f} mm**, "
            f"fill **{fill_h:.0f} mm**\n"
            f"- coil: span at least **{b['z1']-1:.1f} to {b['z2']+1:.1f} mm "
            f"above the tube tip**, "
            f"i.e. length ≥ **{win_len+2:.1f} mm**, centred at "
            f"**{(b['z1']+b['z2'])/2:.1f} mm**. A coil shorter than the "
            f"detection volume is the single worst coil mistake.\n"
            f"- nothing inside the tube: a ball, plug or inner tube ending "
            f"near the window costs a factor of 1.7 to 2.9.")

        # ---- comparison plot
        fig3, ax3 = plt.subplots(figsize=(8.6, 4.6))
        for s, w in curves.items():
            zc = [(a + bb)/2 for a, bb, _, _ in w]
            rr = [t[2] for t in w]
            ax3.plot(zc, rr, "-", lw=2.0, label=s,
                     alpha=1.0 if s == b["shape"] else 0.55)
        ax3.axhline(target, color="#1f8a4c", ls="--", lw=1.4)
        ax3.text(ax3.get_xlim()[1], target*1.03, " target", color="#1f8a4c",
                 fontsize=9, ha="right", va="bottom")
        ax3.plot([(b["z1"]+b["z2"])/2], [b["rms"]], "o", ms=11, mfc="none",
                 mec="#2b2f36", mew=1.8)
        ax3.set_yscale("log")
        ax3.set_xlabel("centre of the detection window  "
                       "(mm above tube tip)")
        ax3.set_ylabel("residual rms  (ppm)")
        ax3.set_title("where to detect, for each bottom shape")
        ax3.grid(color=P.GRID, lw=0.8, which="both"); ax3.set_axisbelow(True)
        ax3.legend(frameon=False, fontsize=9)
        fig3.tight_layout()
        st.pyplot(fig3)

        st.dataframe(
            [{"bottom": r["shape"],
              "window (mm above tip)": f"{r['z1']:.1f} – {r['z2']:.1f}",
              "residual (ppm)": round(r["rms"], 4),
              "linewidth (Hz)": round(r["hz"], 1)} for r in rows],
            hide_index=True, use_container_width=True)

# --------------------------------------------------------------- microwave
with tab4:
    st.write("Reduced microwave models: a 1D transfer matrix up the tube "
             "axis, and a Gaussian beam through the bottom. Milliseconds "
             "instead of an HFSS run.")
    st.caption("These are **not** a replacement for a full-wave solver. They "
               "are exact in their own limits and useless outside them. "
               "`python validate_thz.py` prints the checks and states where "
               "the beam model fails.")

    q = st.columns(4)
    f_GHz = q[0].number_input("frequency (GHz)", 1.0, 1000.0, 330.0, 5.0)
    w0    = q[1].number_input("beam waist at the launcher (mm)",
                              0.2, 10.0, 1.0, 0.1)
    zmaxb = q[2].number_input("plot up to (mm above tip)", 5.0, 75.0, 35.0, 5.0)
    st.caption(" ")

    # permittivities: fall back to editable numbers if the material is not
    # in the microwave table (or is a custom susceptibility)
    def eps_for(label, name, default):
        if name in TZ.MW_MATERIALS:
            e, td = TZ.MW_MATERIALS[name]
            cc = st.columns([2, 1, 1])
            cc[0].markdown(f"**{label}**: `{name}`")
            e = cc[1].number_input(f"ε′ ({label})", 1.0, 40.0, float(e), 0.05,
                                   key=f"e_{label}")
            td = cc[2].number_input(f"tan δ ({label})", 0.0, 2.0, float(td),
                                    0.001, format="%.4f", key=f"t_{label}")
        else:
            cc = st.columns([2, 1, 1])
            cc[0].markdown(f"**{label}**: `{name}` not in the microwave table")
            e = cc[1].number_input(f"ε′ ({label})", 1.0, 40.0, default, 0.05,
                                   key=f"e_{label}")
            td = cc[2].number_input(f"tan δ ({label})", 0.0, 2.0, 0.01, 0.001,
                                    format="%.4f", key=f"t_{label}")
        return e, td

    gname = f"_mw_tube"
    lname = f"_mw_sample"
    TZ.MW_MATERIALS[gname] = eps_for("tube", glass, 4.15)
    TZ.MW_MATERIALS[lname] = eps_for("sample", liquid, 2.45)
    st.caption("These permittivities are literature estimates. Replace them "
               "with the measured values before anything goes in a paper.")

    if st.button("Run the microwave models", type="primary", key="mwbtn"):
        zp, H2, sol = TZ.axial_profile(
            f_GHz, bottom=bottom, wall=wall, punt_r=punt_r,
            bot_depth=bot_depth, fill_h=fill_h, glass=gname, liquid=lname)
        nl = TZ.index(lname)
        alpha = 4*np.pi*abs(nl.imag)/(TZ.C_MM_GHZ/f_GHz)

        m = st.columns(4)
        m[0].metric("reflected", f"{sol['R']*100:.1f} %")
        m[1].metric("absorbed in the sample", f"{sol['A']*100:.1f} %")
        m[2].metric("out of the top", f"{sol['T']*100:.2f} %")
        m[3].metric("1/e power depth",
                    "∞" if alpha <= 0 else f"{1/alpha:.1f} mm")

        if sol["A"] > 0.5:
            st.info(f"The sample absorbs {sol['A']*100:.0f} % of what gets in. "
                    f"Most of the beam never reaches the top of the column, "
                    f"so a reflection returning from above is small by the "
                    f"time it comes back. That is the quantitative answer to "
                    f"the reflected-beam question.")

        # beam_through_bottom propagates from the launcher, so it is called
        # in internal z and converted on the way out.
        zb, wb, zf = TZ.beam_through_bottom(
            f_GHz, w0, d_cwg=d_cwg, bottom=bottom, tube_OD=tube_OD,
            wall=wall, punt_r=punt_r, bot_depth=bot_depth,
            glass=gname, liquid=lname, z_max=from_tip(zmaxb))
        zb_t = to_tip(zb)
        zf_t = to_tip(zf) if np.isfinite(zf) else np.nan
        ri = tube_OD/2 - wall

        fig4, ax4 = plt.subplots(1, 2, figsize=(11.6, 4.4))

        a = ax4[0]
        # axial_profile already returns height above the tube tip
        a.plot(zp, H2/H2.max(), color="#c0703a", lw=1.1)
        a.axvspan(h1, h2, color="#2b2f36", alpha=.07)
        a.set_xlabel("height above tube tip  (mm)")
        a.set_ylabel("|H|²  (normalised)")
        a.set_title("standing wave along the axis")
        a.set_xlim(zp.min(), min(zp.max(), zmaxb))

        b = ax4[1]
        b.plot(zb_t, wb, color="#2f6fb5", lw=2.2)
        b.axhline(ri, color="#9aa7b8", ls="--", lw=1.3)
        b.text(zb_t[-1], ri, " bore wall ", color="#6b7280", fontsize=8.5,
               ha="right", va="bottom")
        b.axvspan(h1, h2, color="#2b2f36", alpha=.07)
        b.axvline(0.0, color="#9aa7b8", lw=1.0)
        b.text(0.0, b.get_ylim()[1]*0.02, " tube tip", color="#6b7280",
               fontsize=8.5, va="bottom")
        if np.isfinite(zf_t):
            b.axvline(zf_t, color="#1f8a4c", ls=":", lw=1.6)
            b.text(zf_t, b.get_ylim()[1]*0.95, f" focus {zf_t:.1f} mm",
                   color="#1f8a4c", fontsize=9, va="top")
        b.set_xlabel("height above tube tip  (mm)")
        b.set_ylabel("beam radius  (mm)")
        b.set_title("beam through the bottom")
        b.set_xlim(zb_t.min(), zmaxb)
        b.set_ylim(0, max(ri*3, np.nanmin(wb)*4))

        for a_ in ax4:
            a_.grid(color=P.GRID, lw=0.8); a_.set_axisbelow(True)
        fig4.tight_layout(w_pad=2.0)
        st.pyplot(fig4)

        if bottom == "punt":
            st.warning("The beam model does not apply to the punt. It predicts "
                       "pure divergence, because a paraxial Gaussian cannot "
                       "represent the tube acting as a dielectric waveguide "
                       "once the beam reaches the wall. The transfer matrix "
                       "on the left is still valid. For the punt's B₁, use HFSS.")
        elif np.isfinite(zf_t):
            inside = h1 <= zf_t <= h2
            wf = float(np.nanmin(wb[zb > d_cwg + wall]))
            (st.success if inside else st.warning)(
                f"Focus **{zf_t:.1f} mm above the tube tip**, waist "
                f"**{wf:.2f} mm**. "
                + ("Inside the detection window."
                   if inside else
                   f"Outside the detection window ({h1:.1f} to {h2:.1f} mm "
                   f"above the tip). "
                   f"On the HFSS wall sweep the figure of merit went as "
                   f"1/waist², so moving the focus into the window is worth "
                   f"more than any interference effect."))

# ------------------------------------------------- lineshape and matching
with tab5:
    st.write("What the residual field actually looks like as a spectrum, "
             "and which part of the tube is responsible for it.")

    nat = st.number_input("natural linewidth of your sample (Hz)",
                          0.05, 100.0, 0.5, 0.05,
                          help="The width you would get in a perfect field. "
                               "Everything beyond this is the tube's fault.")

    res5 = solve(V, kw, coilkw, r_act, z1, z2, shim_order, f_MHz)
    if res5 is None:
        st.error("Active volume contains no sample.")
        st.stop()
    x5, y5, z5, _, dB5, act5, rep5, liq5, gls5, cm5 = res5
    X5, Y5, Z5 = np.meshgrid(x5, y5, z5, indexing="ij")
    resid5 = shim(dB5, X5, Y5, Z5, act5, order=shim_order)

    nu, sp, wd = lineshape_report(resid5, f_MHz, nat_Hz=nat)

    def fmt(v):
        return "—" if not np.isfinite(v) else f"{v:.1f} Hz"

    k5 = st.columns(4)
    k5[0].metric("natural", f"{nat:g} Hz")
    k5[1].metric("observed FWHM", fmt(wd["fwhm"]))
    k5[2].metric("width at 0.55 %", fmt(wd["w055"]))
    k5[3].metric("width at 0.11 %", fmt(wd["w011"]))
    st.caption("The three heights are the standard NMR lineshape test. A "
               "good FWHM with a wide foot is the classic signature of "
               "susceptibility mismatch: a sharp line sitting on a broad "
               "hump, which is exactly what ruins a 2D experiment.")

    rr = wd["resolution"]
    if rr > 0.5 * nat:
        st.warning(f"Sampling resolution is {rr:.2f} Hz, coarser than the "
                   f"{nat:g} Hz natural line. {int(act5.sum())} voxels cannot "
                   f"resolve a line this narrow, so the widths above are "
                   f"broadened by the sampling and will shrink if you reduce "
                   f"the voxel size. The rms residual is far less sensitive "
                   f"to this than the widths are.")
    else:
        st.caption(f"Sampling resolution {rr:.2f} Hz from "
                   f"{int(act5.sum())} voxels, fine enough for these widths.")

    figL, axL = plt.subplots(figsize=(9.0, 4.2))
    axL.plot(nu, sp, color="#2f6fb5", lw=2.0, label="your tube")
    pure = (nat/2)**2 / ((nu - nu[int(np.argmax(sp))])**2 + (nat/2)**2)
    axL.plot(nu, pure, color="#9aa7b8", lw=1.4, ls="--",
             label=f"perfect field ({nat:g} Hz)")
    for frac, lab in ((0.5, "50 %"), (0.0055, "0.55 %"), (0.0011, "0.11 %")):
        axL.axhline(frac, color="#c8ccd2", lw=0.8)
        axL.text(nu[-1], frac, f" {lab}", fontsize=8, color=P.MUTED,
                 ha="right", va="bottom")
    axL.set_yscale("log"); axL.set_ylim(3e-4, 1.6)
    axL.set_xlabel("offset  (Hz)"); axL.set_ylabel("intensity")
    axL.set_title("simulated lineshape")
    axL.legend(frameon=False, fontsize=9)
    axL.grid(color=P.GRID, lw=0.8, which="major"); axL.set_axisbelow(True)
    figL.tight_layout()
    st.pyplot(figL)

    st.divider()
    st.subheader("Which part of the tube is to blame")
    st.caption("One field solve per material region at unit susceptibility. "
               "The method is linear in χ, so every material combination is "
               "a weighted sum of these and costs no further computation.")

    # The basis survives reruns so the material selector below stays live.
    # It is invalidated whenever the geometry or the window changes.
    if cwg_on:
        st.caption("The waveguide is included in the line above but not in "
                   "this breakdown, which covers the tube regions only.")
    sig = repr((kw, V, r_act, z1, z2, shim_order))
    if st.button("Break it down", type="primary", key="brk"):
        with st.spinner("one field solve per region…"):
            st.session_state["basis"] = region_basis(
                x5, y5, z5, kw, (V, V, V), X5, Y5, Z5, act5, order=shim_order)
            st.session_state["basis_sig"] = sig

    if st.session_state.get("basis_sig") == sig:
        B = st.session_state["basis"]
        rows, tot = breakdown(B)

        st.dataframe([{"region": n, "χ (ppm)": round(c, 2),
                       "contributes (ppm)": round(v, 5),
                       "share": f"{v/sum(r[2] for r in rows)*100:.0f} %"}
                      for n, c, v in rows],
                     hide_index=True, use_container_width=True)
        st.caption(f"Total residual {tot:.5f} ppm. Contributions do not add "
                   f"linearly: they partly cancel.")

        dom = rows[0][0]
        st.info(f"**{dom}** dominates. Matching a material can only help "
                f"with the region that dominates, so that is the one to "
                f"change, if you can change it.")

        tgt = st.selectbox("find the best susceptibility for", list(B))
        cb, rb, rn = match_region(B, tgt)
        if not np.isfinite(cb):
            st.warning("That region has no measurable effect here.")
        else:
            mm = st.columns(3)
            mm[0].metric("best χ", f"{cb:+.2f} ppm")
            mm[1].metric("residual there", f"{rb:.4f} ppm")
            mm[2].metric("gain", f"{rn/rb:.2f}×" if rb > 0 else "—")

            if rn / max(rb, 1e-12) < 1.3:
                st.warning(f"Matching the {tgt} buys only {rn/rb:.2f}×. "
                           f"Not worth custom glass. The residual here is "
                           f"not dominated by this region.")
            else:
                st.success(f"Matching the {tgt} is worth {rn/rb:.1f}×.")
                st.write("Closest real materials:")
                st.dataframe(
                    [{"material": m, "χ (ppm)": v,
                      "residual (ppm)": round(residual_with(B, {tgt: v}), 5)}
                     for m, v in nearest_materials(cb, 5)],
                    hide_index=True, use_container_width=True)

            cs = np.linspace(cb - 14, cb + 14, 81)
            rs = [residual_with(B, {tgt: c}) for c in cs]
            figM, axM = plt.subplots(figsize=(8.4, 4.0))
            axM.plot(cs, rs, color="#2f6fb5", lw=2.2)
            axM.plot([cb], [rb], "o", ms=10, mfc="none", mec="#2b2f36", mew=1.8)
            axM.axvline(B[tgt][1], color="#c0703a", ls="--", lw=1.4)
            axM.text(B[tgt][1], max(rs), " now", color="#c0703a", fontsize=9,
                     va="top")
            for m, v in nearest_materials(cb, 5):
                if cs[0] <= v <= cs[-1]:
                    axM.plot([v], [residual_with(B, {tgt: v})], "s", ms=5,
                             color="#1f8a4c")
                    axM.annotate(m, (v, residual_with(B, {tgt: v})),
                                 textcoords="offset points", xytext=(4, 6),
                                 fontsize=8, color="#1f8a4c")
            axM.set_xlabel(f"χ of the {tgt}  (ppm)")
            axM.set_ylabel("residual rms  (ppm)")
            axM.set_title(f"susceptibility matching: {tgt}")
            axM.grid(color=P.GRID, lw=0.8); axM.set_axisbelow(True)
            figM.tight_layout()
            st.pyplot(figM)
