"""
B0 builder: the simple version of the B0 tool.

Start from a preset, add only the parts your probe has, press Run.
Run with:   python -m streamlit run builder.py

The full version with sweeps, recommendations, microwave models and
susceptibility matching is app.py. Both use the same physics (b0core.py).
"""
import io
import json
import uuid

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

import builder_core as C
from b0core import MATERIALS_PPM, SOURCES
import plotting as P

P.use_style()
st.set_page_config(page_title="B0 builder", layout="wide")

# ------------------------------------------------------------------ state
# Two setups, A and B. Widget values live in ss.cfg[A or B], not in the
# widgets, so a part keeps its settings when you switch it off and on again
# (Streamlit forgets the value of a widget that is not on screen) and when
# you switch between editing A and B. Widget keys carry the setup letter.
ss = st.session_state
if "cfg" not in ss:
    ss.cfg = {"A": dict(C.BLANK), "B": dict(C.BLANK)}
    ss.cust = {"A": [], "B": []}
    ss.preset = {"A": "Blank (tube only)", "B": "Blank (tube only)"}
    ss.solved = {}
    ss.edit = "A"
    ss.auto_run = True
AB = bool(ss.get("ab_on", False))
if not AB:
    ss.edit = "A"
E = ss.get("edit", "A")
S = ss.cfg[E]
WK = f"w{E}_"                       # widget-key prefix for this setup
ABCOL = {"A": "#2f6fb5", "B": "#d9822b"}

BASE_MATS = [m for m in MATERIALS_PPM if not m.startswith("custom")]
SOLIDS = [m for m in BASE_MATS if m not in ("air", "vacuum",
    "water", "D2O", "toluene", "toluene-d8", "DMSO", "chloroform-d",
    "acetone-d6", "acetonitrile-d3", "benzene-d6", "methanol-d4", "THF-d8",
    "methanol", "ethanol", "glycerol/water 60:40")]
LIQUIDS = [m for m in BASE_MATS if m not in SOLIDS + ["air", "vacuum"]]


def mat_label(m):
    if m.startswith("custom"):
        return "custom χ…"
    return f"{m}  ({MATERIALS_PPM[m]:+.2f} ppm)"


def _forget_widgets(e):
    """Drop the widget keys of setup e so they reload from ss.cfg[e]."""
    pids = {p["id"] for p in ss.cust[e]}
    for k in list(ss.keys()):
        k = str(k)
        if k.startswith(f"w{e}_") or (k.startswith("cw_")
                                       and k.split("_")[1] in pids):
            del ss[k]


def apply_preset():
    e = ss.edit
    name = ss[f"preset_pick_{e}"]
    _forget_widgets(e)
    ss.preset[e] = name
    ss.cfg[e] = dict(C.PRESETS[name][0])
    ss.cust[e] = []
    ss.auto_run = True


def copy_setup(src, dst):
    _forget_widgets(dst)
    ss.cfg[dst] = dict(ss.cfg[src])
    ss.cust[dst] = [dict(p, id=uuid.uuid4().hex[:8]) for p in ss.cust[src]]
    ss.preset[dst] = ss.preset[src]
    ss.auto_run = True


def W(widget, label, key, **kw):
    """A widget whose value is remembered in S[key]."""
    wk = WK + key
    if wk not in ss:
        ss[wk] = S[key]
    val = widget(label, key=wk, **kw)
    S[key] = val
    return val


def num(label, key, lo, hi, step, help=None, fmt=None, disabled=False):
    return W(st.number_input, label, key, min_value=float(lo),
             max_value=float(hi), step=float(step), help=help,
             format=fmt, disabled=disabled)


def mat(label, key, options, help=None):
    if S[key] not in options:
        options = options + [S[key]]
    return W(st.selectbox, label, key, options=options,
             format_func=mat_label, help=help)


def source_note(m):
    if m.startswith("custom"):
        return
    kind, src = SOURCES.get(m, ("?", ""))
    st.caption(f"χ = {MATERIALS_PPM[m]:+.2f} ppm · {kind}. {src}")


def cfg_of(e):
    c = dict(ss.cfg[e])
    c["custom"] = [{k: v for k, v in p.items() if k != "id"}
                   for p in ss.cust[e]]
    return c


def cfg_now():
    return cfg_of(E)


# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("B₀ builder")
sb.caption("Build your probe one part at a time and see how it disturbs "
           "the magnetic field in your sample.")

def _ab_changed():
    if ss.ab_on and not ss.get("b_init"):
        copy_setup("A", "B")          # B starts as a copy of A
        ss.b_init = True
    ss.auto_run = True


sb.toggle("Compare two setups (A / B)", key="ab_on", on_change=_ab_changed,
          help="Build two versions side by side (with and without the "
               "waveguide, two tube sizes, two positions...) and see both "
               "results together.")
if AB:
    sb.radio("Editing", ["A", "B"], key="edit", horizontal=True,
             format_func=lambda e: f"setup {e}")
    o = "B" if E == "A" else "A"
    sb.button(f"Copy setup {o} into {E}", on_click=copy_setup, args=(o, E),
              width="stretch",
              help=f"Start {E} as an exact copy of {o}, then change one thing.")
    sb.divider()

names = list(C.PRESETS)
sb.radio("Start from" + (f" (setup {E})" if AB else ""), names,
         index=names.index(ss.preset[E]),
         key=f"preset_pick_{E}", on_change=apply_preset,
         help="Loads a complete setup. You can change anything afterwards.")
sb.caption(C.PRESETS[ss.preset[E]][1])

sb.divider()
sb.subheader("Save / load")
sb.download_button("Save this setup (.json)" if not AB
                   else f"Save setup {E} (.json)",
                   json.dumps(cfg_now(), indent=1),
                   file_name="b0_setup.json", mime="application/json",
                   width="stretch")
up = sb.file_uploader("Load a setup" + (f" into {E}" if AB else ""),
                      type="json",
                      help="A file saved with the button above.")
if up is not None and ss.get("loaded_id") != up.file_id:
    try:
        d = json.loads(up.getvalue())
        new = dict(C.BLANK)
        for k, v in d.items():
            if k in C.BLANK:
                d0 = C.BLANK[k]
                new[k] = (float(v) if isinstance(d0, float) else
                          int(v) if isinstance(d0, int) and
                          not isinstance(d0, bool) else v)
        _forget_widgets(E)
        ss.cfg[E] = new
        ss.cust[E] = [dict(C.CUSTOM_DEFAULT, **p, id=uuid.uuid4().hex[:8])
                      for p in d.get("custom", [])]
        ss.loaded_id = up.file_id
        ss.auto_run = True
        st.rerun()
    except Exception as e:
        sb.error(f"Could not read that file: {e}")

sb.divider()
with sb.expander("Accuracy (advanced)" + (f", setup {E}" if AB else "")):
    W(st.select_slider, "voxel size (mm)", "V", options=C.VOXELS,
      help="The cube size the space is cut into. Smaller = more accurate "
           "but slower and more memory. 0.25 mm is a good default.")
    st.caption("Big probes are coarsened automatically if they would not "
               "fit in memory.")
    if st.button("Check the solver", width="stretch",
                 help="Solves a case with an exact textbook answer (a "
                      "uniform sphere) at this voxel size and reports the "
                      "error."):
        chk = C.self_check(float(S["V"]))
        st.caption(f"Uniform sphere, voxel {chk['voxel']} mm: field inside "
                   f"flat to **{chk['inside']*100:.1f} %**, dipole field "
                   f"outside right to **{chk['outside']*100:.1f} %** of the "
                   "largest exact value.")
sb.caption("Need sweeps, recommendations or the microwave models? "
           "Run the full app: `python -m streamlit run app.py`")


# ----------------------------------------------------------------- layout
left, right = st.columns([1.0, 1.45], gap="large")

with left:
    if AB:
        st.markdown(f"<h3 style='margin-bottom:0'>Setup <span style='color:"
                    f"{ABCOL[E]}'>{E}</span></h3>", unsafe_allow_html=True)
        st.caption("Switch between A and B in the sidebar.")
    else:
        st.subheader("Your probe")
    st.caption("All heights in mm, measured from **z = 0**, the probe "
               "reference (the waveguide top, if you add one).")

    # ---------------------------------------------------------- tube
    with st.expander("Sample tube", expanded=True):
        a, b = st.columns(2)
        with a:
            num("outer diameter", "t_OD", 1.0, 12.0, 0.1,
                help="Outside diameter of the tube (mm).")
            num("liquid column", "t_fill", 2.0, 80.0, 1.0,
                help="Height of the liquid, from the inner bottom (mm).")
        with b:
            num("wall thickness", "t_wall", 0.05, 2.0, 0.05,
                help="Glass wall thickness (mm).")
            num("tube tip height", "t_tip", -20.0, 40.0, 0.5,
                help="Where the outside tip of the tube sits, above z = 0. "
                     "Change this to move the sample up or down in the probe.")
        g = mat("tube material", "t_glass", SOLIDS + ["custom (tube)"])
        if g.startswith("custom"):
            num("tube χ (ppm)", "t_glass_chi", -60.0, 60.0, 0.1)
        else:
            source_note(g)
        l = mat("sample", "t_liquid", LIQUIDS + ["custom (sample)"])
        if l.startswith("custom"):
            num("sample χ (ppm)", "t_liquid_chi", -60.0, 60.0, 0.1)
        else:
            source_note(l)
        if st.toggle("more tube options", key="adv_tube"):
            W(st.radio, "bottom shape", "t_bottom",
              options=["round", "flat", "punt"], horizontal=True,
              help="punt = a dome pushed up into the bottom, like a wine "
                   "bottle.")
            a, b = st.columns(2)
            with a:
                num("round bottom depth", "t_botdepth", 0.5, 6.0, 0.1,
                    disabled=S["t_bottom"] != "round")
                num("meniscus dip", "t_men", 0.0, 3.0, 0.1,
                    help="0 = flat liquid surface. Bigger = more curved.")
            with b:
                num("punt radius", "t_punt", 0.5, 3.0, 0.1,
                    disabled=S["t_bottom"] != "punt")
            mat("surroundings", "t_out", ["air", "vacuum"] + SOLIDS,
                help="What fills the probe around the tube.")

    st.markdown("**Probe parts**")
    st.caption("Switch on only what your probe has.")

    # ------------------------------------------------------------ coil
    if W(st.toggle, "RF coil", "c_on",
         help="The NMR coil around the tube. Modelled as a thin slotted "
              "metal tube (a slotted-tube resonator)."):
        with st.expander("RF coil settings", expanded=False):
            a, b = st.columns(2)
            with a:
                num("inner radius", "c_r", 1.0, 15.0, 0.1)
                num("length", "c_len", 1.0, 60.0, 0.5)
                W(st.number_input, "number of slots", "c_nslots",
                  min_value=0, max_value=8, step=1)
            with b:
                num("foil thickness (µm)", "c_t", 5.0, 500.0, 5.0)
                num("centre height", "c_zc", -20.0, 80.0, 0.5)
                num("slot width (deg)", "c_slotdeg", 0.0, 180.0, 1.0)
            mat("coil metal", "c_mat", SOLIDS)
            if st.toggle("more coil options", key="adv_coil"):
                num("slot angle offset (deg)", "c_phase", 0.0, 180.0, 5.0,
                    help="Rotates the slots around the tube.")
                W(st.checkbox, "slots are windows with a feed slit "
                  "(FRASCAN II shape)", "c_win")
                if S["c_win"]:
                    a, b = st.columns(2)
                    with a:
                        num("window starts above coil bottom", "c_winz0",
                            0.0, 50.0, 0.5)
                        num("slit width", "c_slit", 0.1, 5.0, 0.1)
                    with b:
                        num("window height", "c_winh", 1.0, 50.0, 0.5)

    # ------------------------------------------------------- waveguide
    if W(st.toggle, "Waveguide", "w_on",
         help="A metal waveguide below the tube, its top at z = 0."):
        with st.expander("Waveguide settings", expanded=False):
            a, b = st.columns(2)
            with a:
                num("inner diameter", "w_id", 1.0, 20.0, 0.1)
                num("length modelled", "w_len", 2.0, 40.0, 1.0,
                    help="Only its top end matters. 20 mm is plenty.")
            with b:
                num("wall", "w_wall", 0.1, 5.0, 0.05)
            mat("metal", "w_mat", SOLIDS,
                help="Brass is an estimate. Measure it if it matters.")

    # --------------------------------------------------------- shields
    if W(st.toggle, "RF shields", "rs_on",
         help="Thin metal shields: a flat ring under the coil and a tube "
              "around it."):
        with st.expander("RF shield settings", expanded=False):
            W(st.checkbox, "bottom disc", "rs_bot")
            if S["rs_bot"]:
                a, b = st.columns(2)
                with a:
                    num("inner radius", "rs_b_rin", 0.0, 30.0, 0.1)
                    num("bottom height", "rs_b_z0", -30.0, 60.0, 0.01,
                        fmt="%.2f")
                with b:
                    num("outer radius", "rs_b_rout", 0.1, 30.0, 0.1)
                    num("top height", "rs_b_z1", -30.0, 60.0, 0.01,
                        fmt="%.2f")
            W(st.checkbox, "side tube", "rs_side")
            if S["rs_side"]:
                a, b = st.columns(2)
                with a:
                    num("inner radius ", "rs_s_rin", 0.0, 30.0, 0.1)
                    num("bottom height ", "rs_s_z0", -30.0, 80.0, 0.05)
                with b:
                    num("outer radius ", "rs_s_rout", 0.1, 30.0, 0.05)
                    num("top height ", "rs_s_z1", -30.0, 80.0, 0.05)
            mat("metal ", "rs_mat", SOLIDS)

    # ------------------------------------------------- modulation coil
    if W(st.toggle, "Modulation coil", "m_on",
         help="A wound coil around the probe (EPR field modulation). "
              "Modelled as a copper ring, part copper, part gaps."):
        with st.expander("Modulation coil settings", expanded=False):
            a, b = st.columns(2)
            with a:
                num("inner radius  ", "m_rin", 0.0, 30.0, 0.1)
                num("bottom height  ", "m_z0", -30.0, 80.0, 0.05)
            with b:
                num("outer radius  ", "m_rout", 0.1, 30.0, 0.1)
                num("top height  ", "m_z1", -30.0, 80.0, 0.05)
            num("copper fill", "m_fill", 0.1, 1.0, 0.05,
                help="Fraction of the winding that is metal. Round wire "
                     "packs to about 0.75.")
            mat("wire metal", "m_mat", SOLIDS)

    # ---------------------------------------------------- custom parts
    for p in list(ss.cust[E]):
        pid = p["id"]
        with st.expander(f"Custom part · {p['kind']} · {p['mat']}",
                         expanded=False):
            def cw(widget, label, k, **kw):
                key = f"cw_{pid}_{k}"
                if key not in ss:
                    ss[key] = p[k]
                p[k] = widget(label, key=key, **kw)
                return p[k]

            def cnum(label, k, lo, hi, step):
                return cw(st.number_input, label, k, min_value=float(lo),
                          max_value=float(hi), step=float(step))

            cw(st.radio, "shape", "kind", options=["cylinder / ring", "box"],
               horizontal=True)
            opts = SOLIDS if p["mat"] in SOLIDS else SOLIDS + [p["mat"]]
            cw(st.selectbox, "material", "mat", options=opts,
               format_func=mat_label)
            a, b = st.columns(2)
            if p["kind"] == "box":
                with a:
                    cnum("size x", "sx", 0.05, 40.0, 0.1)
                    cnum("centre x", "cx", -20.0, 20.0, 0.1)
                with b:
                    cnum("size y", "sy", 0.05, 40.0, 0.1)
                    cnum("centre y", "cy", -20.0, 20.0, 0.1)
            else:
                with a:
                    cnum("inner radius (0 = solid)", "r_in", 0.0, 20.0, 0.1)
                    cnum("centre x", "cx", -20.0, 20.0, 0.1)
                with b:
                    cnum("outer radius", "r_out", 0.05, 20.0, 0.1)
                    cnum("centre y", "cy", -20.0, 20.0, 0.1)
            with a:
                cnum("bottom height", "z0", -40.0, 80.0, 0.1)
            with b:
                cnum("top height", "z1", -40.0, 80.0, 0.1)
            cnum("fill fraction", "fill", 0.05, 1.0, 0.05)
            if st.button("Remove this part", key=f"rm_{pid}"):
                ss.cust[E] = [q for q in ss.cust[E] if q["id"] != pid]
                st.rerun()
    if st.button("＋ Add a custom part", width="stretch",
                 help="Any cylinder, ring or box of any material: a holder, "
                      "a spacer, a screw, a lens..."):
        ss.cust[E].append(dict(C.CUSTOM_DEFAULT, id=uuid.uuid4().hex[:8]))
        st.rerun()

    # ------------------------------------------- detection and shimming
    st.markdown("**Measurement**")
    with st.expander("Detection region & shimming", expanded=False):
        auto_ok = S["c_on"]
        W(st.checkbox, "detection region = the coil window", "d_auto",
          disabled=not auto_ok,
          help="The part of the sample the coil actually sees. Without a "
               "coil, set it by hand below.")
        manual = not (S["d_auto"] and auto_ok)
        a, b = st.columns(2)
        with a:
            num("bottom height   ", "d_z0", -20.0, 80.0, 0.5,
                disabled=not manual)
            num("radius", "d_r", 0.1, 6.0, 0.1,
                help="Radius of the detected sample cylinder.")
        with b:
            num("top height   ", "d_z1", -20.0, 80.0, 0.5,
                disabled=not manual)
        if not manual:
            z0, z1 = C.coil_window(S)
            st.caption(f"From the coil: **{z0:.1f} to {z1:.1f} mm**.")
        W(st.select_slider, "shimming, up to", "sh_order",
          options=[0, 1, 2, 3, 4], format_func=lambda o: C.SHIM_LABELS[o],
          help="Which correction coils the spectrometer uses to flatten "
               "the field. 'none' only removes the average offset. A real "
               "Bruker room-temperature shim set goes to 4th order and "
               "beyond along z, so 4 is closest to the real magnet; 2 is "
               "pessimistic.")
        a, b = st.columns(2)
        with a:
            num("¹H frequency (MHz)", "f_MHz", 10.0, 1500.0, 10.0)
        with b:
            num("linewidth without the probe (Hz)", "nat", 0.05, 100.0, 0.05,
                help="The width you would get if this probe disturbed "
                     "nothing: the sample's natural width plus whatever the "
                     "magnet itself leaves after shimming. Best estimate: "
                     "the linewidth measured on the same magnet and sample "
                     "with a good commercial probe. The tool adds this "
                     "probe's broadening on top.")
    with st.expander("Compare with a measured spectrum", expanded=False):
        st.caption("Upload a spectrum as two columns (frequency, "
                   "intensity), .csv or .txt. It is centred on its tallest "
                   "point, scaled to 1 and drawn over the simulated line.")
        mu = st.radio("frequency column is in", ["Hz", "ppm"],
                      horizontal=True, key="meas_unit")
        mf = st.file_uploader("measured spectrum", type=["csv", "txt", "dat"],
                              key="meas_file", label_visibility="collapsed")
        ss.meas = None
        if mf is not None:
            try:
                ss.meas = C.parse_spectrum(
                    mf.getvalue().decode("utf-8", "replace"), unit=mu,
                    f_MHz=float(S["f_MHz"]))
            except Exception:
                ss.meas = None
            if ss.meas is None:
                st.error("Could not read two numeric columns from that file.")
            else:
                st.caption(f"Read {len(ss.meas['nu'])} points, measured FWHM "
                           f"**{ss.meas['fwhm']:.1f} Hz**.")


# ---------------------------------------------------------------- preview
COLORS = {"glass": "#9aa7b8", "sample": "#f0b05a", "RF coil": "#c0703a",
          "waveguide": "#b59a3c", "shield": "#7f8c9b",
          "modulation coil": "#a0522d", "custom": "#5aa58a"}


def preview_frame(cfg):
    """(half-width, z_lo, z_hi) of the preview picture for one setup."""
    r, zlo, zhi = C.extent(cfg)
    zhi = cfg["t_tip"] + cfg["t_wall"] + cfg["t_fill"] + 3.0
    for kind, p, _ in C._parts(cfg):
        zhi = max(zhi, p["z1"] + 1.0)
    if cfg["c_on"]:
        zhi = max(zhi, cfg["c_zc"] + cfg["c_len"]/2 + 1.0)
    zlo = max(zlo - 1.0, cfg["t_tip"] - 8.0) if cfg["w_on"] else zlo - 1.0
    return max(min(r + 1.0, 22.0), 6.0), zlo, zhi


def preview(cfg, det, compact=False, frame=None):
    """x-z cut through the axis, drawn with the same builders the solver
    uses, at fine resolution. Instant, no FFT. `frame` forces the picture
    limits, so A and B are drawn on the same scale."""
    half, zlo, zhi = frame or preview_frame(cfg)
    h = max(0.04, 2*half/500)
    x = np.arange(-half + h/2, half, h)
    z = np.arange(zlo + h/2, zhi, h)
    _, masks = C.build_all(cfg, x, np.array([0.0]), z)
    img = np.ones((len(z), len(x), 3)) * np.array(mpl.colors.to_rgb("#f3f5f8"))
    shown = []
    for name in ["glass", "sample", "waveguide", "modulation coil", "shield",
                 "custom", "RF coil"]:
        if name in masks:
            m = masks[name][:, 0, :].T
            if m.any():
                img[m] = mpl.colors.to_rgb(COLORS[name])
                shown.append(name)
    fig, ax = plt.subplots(figsize=(3.6, 5.6) if compact else (6.2, 5.0))
    ax.imshow(img, origin="lower", aspect="equal", interpolation="nearest",
              extent=[x[0]-h/2, x[-1]+h/2, z[0]-h/2, z[-1]+h/2])
    z0, z1, rr = det
    ax.add_patch(plt.Rectangle((-rr, z0), 2*rr, z1 - z0, fill=False,
                               ec=P.INK, lw=1.4, ls=(0, (4, 3))))
    ax.axhline(0.0, color=P.INK, lw=0.8, alpha=0.5)
    ax.text(x[0], 0.0, " z = 0", fontsize=8, color=P.MUTED, va="bottom")
    handles = [mpl.patches.Patch(color=COLORS[n], label=n) for n in shown]
    handles.append(mpl.lines.Line2D([], [], color=P.INK, ls=(0, (4, 3)),
                                    lw=1.4, label="detected region"))
    if compact:
        ax.legend(handles=handles, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, -0.11), fontsize=9, frameon=False)
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                  fontsize=8.5, frameon=False)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("height  (mm)")
    ax.set_title("cut through the axis")
    fig.tight_layout()
    return fig


def show(fig, tight=False):
    """Draw a figure as an image (more robust than st.pyplot across
    Python versions)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170,
                bbox_inches="tight" if tight else None)
    plt.close(fig)
    st.image(buf, width="stretch")


@st.cache_resource(max_entries=4, show_spinner=False)
def solve_cached(sig):
    return C.solve(json.loads(sig))


def draw_measured(ax):
    m = ss.get("meas")
    if m:
        ax.plot(m["nu"], m["y"], color=P.INK, lw=1.3, alpha=0.85,
                label=f"measured ({m['fwhm']:.1f} Hz)")


def measured_span():
    m = ss.get("meas")
    return 4 * m["fwhm"] if m and np.isfinite(m["fwhm"]) else 0.0


def order_table(an, scfg):
    """Linewidth against shim order: the honest error bar on the result."""
    cur = int(scfg["sh_order"])
    rows = ["| shimming up to | leftover field | FWHM | width at 0.55 % |",
            "|---|---|---|---|"]
    for o, ppb, fw, w055 in an["by_order"]:
        b = "**" if o == cur else ""
        rows.append(f"| {b}{C.SHIM_LABELS[o]}{b} | {b}{ppb:.1f} ppb{b} | "
                    f"{b}{fmt(fw, '{:.1f} Hz')}{b} | "
                    f"{fmt(w055, '{:.1f} Hz')} |")
    st.markdown("\n".join(rows))
    st.caption("How the result depends on how well the magnet is shimmed. "
               "A real shim set reaches 4th order and beyond along z, but "
               "is rarely set perfectly, so the measured line usually falls "
               "between the 2nd and 4th order rows.")


def results_figure(sol, an, cfg):
    x, z, j = sol["x"], sol["z"], sol["j"]
    dB, liq = sol["dB"], sol["liq"]
    ref = float(an["raw"].mean())
    v = np.abs(an["raw"] - ref)
    lim = max(1e-4, np.percentile(v, 99) * 1.4)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.7),
                           gridspec_kw={"width_ratios": [0.9, 1.5]})
    d = np.where(liq, dB - ref, np.nan)[:, j, :]
    zl = (cfg["t_tip"] - 1.0, cfg["t_tip"] + cfg["t_wall"] + cfg["t_fill"] + 1)
    ri = cfg["t_OD"] / 2 + 0.5
    m = ax[0].imshow(d.T, origin="lower", aspect="auto", cmap=P.FIELD_CMAP,
                     vmin=-lim, vmax=lim, interpolation="bilinear",
                     extent=[x[0], x[-1], z[0], z[-1]])
    ax[0].add_patch(plt.Rectangle((-an["r"], an["z0"]), 2*an["r"],
                                  an["z1"] - an["z0"], fill=False, ec=P.INK,
                                  lw=1.4, ls=(0, (4, 3))))
    ax[0].set_xlim(-ri, ri)
    ax[0].set_ylim(*zl)
    cb = fig.colorbar(m, ax=ax[0], fraction=0.08, pad=0.04)
    cb.outline.set_visible(False)
    cb.set_label("field change  (ppm)", color=P.MUTED, fontsize=9)
    ax[0].set_title("field in the sample\n(before shimming)")
    ax[0].set_xlabel("x  (mm)")
    ax[0].set_ylabel("height  (mm)")

    nu, sp, wd = an["nu"], an["sp"], an["wd"]
    pk = nu[int(np.argmax(sp))]
    nat = cfg["nat"]
    ideal = (nat/2)**2 / ((nu - pk)**2 + (nat/2)**2)
    ax[1].plot(nu - pk, ideal, color="#9aa7b8", lw=1.4, ls="--",
               label=f"without the probe ({nat:g} Hz)")
    ax[1].plot(nu - pk, sp, color="#2f6fb5", lw=2.2, label="simulated")
    draw_measured(ax[1])
    fw = wd["fwhm"]
    if np.isfinite(fw):
        # walk out from the peak to find the two half-height points
        i = int(np.argmax(sp))
        l = i
        while l > 0 and sp[l] > 0.5:
            l -= 1
        r = i
        while r < len(sp) - 1 and sp[r] > 0.5:
            r += 1
        ax[1].annotate("", xy=(nu[l]-pk, 0.5), xytext=(nu[r]-pk, 0.5),
                       arrowprops=dict(arrowstyle="<->", color=P.INK, lw=1.2))
        ax[1].text(nu[r]-pk, 0.5, f"  FWHM {fw:.1f} Hz", va="center",
                   fontsize=9.5, color=P.INK)
    span = max(6*fw if np.isfinite(fw) else 0, 6*nat, 2.0)
    span = max(span, measured_span())
    ax[1].set_xlim(-span, span)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_xlabel("frequency offset  (Hz)")
    ax[1].set_ylabel("signal")
    ax[1].set_title("the NMR line you would measure")
    ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[1].grid(color=P.GRID, lw=0.8)
    ax[1].set_axisbelow(True)
    fig.tight_layout(w_pad=2.5)
    return fig


GEOM_SKIP = ("d_auto", "d_r", "d_z0", "d_z1", "sh_order", "f_MHz", "nat")
SETUPS = ["A", "B"] if AB else ["A"]


def result_for(e):
    """(sol, an, scfg, stale) for setup e, or None if it was never run.
    Detection, shimming and frequency are cheap, so the live values are
    always used; only geometry changes need a new Run."""
    if e not in ss.solved:
        return None
    cfg = cfg_of(e)
    sig = ss.solved[e]
    old = json.loads(sig)
    scfg = dict(old)
    for k in GEOM_SKIP:
        scfg[k] = cfg[k]
    sol = solve_cached(sig)
    stale = any(old.get(k) != cfg.get(k) for k in cfg if k not in GEOM_SKIP)
    return sol, C.analyse(sol, scfg), scfg, stale


def fmt(v, f):
    return f.format(v) if v is not None and np.isfinite(v) else "n/a"


def numbers(an, scfg):
    wd, rep = an["wd"], an["rep"]
    return {
        "linewidth (FWHM)": (wd["fwhm"], "{:.1f} Hz"),
        "added by the field": (wd["fwhm"] - scfg["nat"], "{:+.1f} Hz"),
        "field spread after shimming": (rep["shim_ppm_rms"] * 1000,
                                        "{:.1f} ppb"),
        "spread before shimming": (rep["raw_ppm_pp"], "{:.3f} ppm"),
        "width at 0.55 %": (wd["w055"], "{:.1f} Hz"),
        "width at 0.11 %": (wd["w011"], "{:.1f} Hz"),
    }


def single_results(sol, an, scfg):
    wd, rep = an["wd"], an["rep"]
    nat = scfg["nat"]
    k = st.columns(3)
    k[0].metric("linewidth (FWHM)", fmt(wd["fwhm"], "{:.1f} Hz"),
                help="Full width at half maximum of the simulated line.")
    k[1].metric("added by the field", fmt(wd["fwhm"] - nat, "{:+.1f} Hz"),
                help="How much wider the line is than in a perfect magnet.")
    k[2].metric("field spread after shimming",
                f"{rep['shim_ppm_rms']*1000:.1f} ppb",
                help="rms of the leftover field over the detected region. "
                     "1 ppb = 0.001 ppm.")
    show(results_figure(sol, an, scfg))
    with st.expander("More results"):
        order_table(an, scfg)
        t = st.columns(4)
        t[0].metric("before shimming", f"{rep['raw_ppm_pp']*1000:.0f} ppb",
                    help="Field spread over the detected region before "
                         "shimming, peak to peak.")
        t[1].metric("after shimming", f"{rep['shim_ppm_pp']*1000:.0f} ppb",
                    help="Field spread after shimming, peak to peak.")
        t[2].metric("width at 0.55 %", fmt(wd["w055"], "{:.1f} Hz"))
        t[3].metric("width at 0.11 %", fmt(wd["w011"], "{:.1f} Hz"))
        st.caption("0.55 % and 0.11 % are the standard NMR lineshape test: "
                   "they show the broad foot of the line that the FWHM hides.")
        f2, a2 = plt.subplots(1, 2, figsize=(7.4, 3.0))
        P.hist_panel(a2[0], an["res"], scfg["f_MHz"])
        a2[0].set_title("leftover field after shimming")
        log_line(a2[1], [(an, "#2f6fb5", None)], nat)
        f2.tight_layout(w_pad=2.5)
        try:
            show(f2)
        except Exception:
            plt.close(f2)
            st.caption("This plot could not be drawn on this server.")
        st.caption(f"{an['n']} voxels detected, voxel {sol['V']} mm, "
                   f"detection {an['z0']:.1f} to {an['z1']:.1f} mm, radius "
                   f"{an['r']:.1f} mm, shimming: "
                   f"{C.SHIM_LABELS[int(scfg['sh_order'])]}.")


def log_line(ax, lines, nat):
    xr = 3 * nat
    for an, col, lab in lines:
        pk = an["nu"][int(np.argmax(an["sp"]))]
        ax.plot(an["nu"] - pk, np.clip(an["sp"], 1e-6, None), color=col,
                lw=1.8, label=lab)
        w11 = an["wd"]["w011"]
        xr = max(xr, 0.8 * (w11 if np.isfinite(w11) else 10 * nat))
    ax.set_xlim(-xr, xr)
    for frac, lab in ((0.5, "50 %"), (0.0055, "0.55 %"), (0.0011, "0.11 %")):
        ax.axhline(frac, color="#c8ccd2", lw=0.8)
        ax.text(xr, frac, f"{lab} ", fontsize=8, color=P.MUTED,
                va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.6)
    ax.set_xlabel("frequency offset  (Hz)")
    ax.set_title("the line, log scale")
    ax.grid(color=P.GRID, lw=0.8)
    if any(l[2] for l in lines):
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def ab_lines(R):
    """Both simulated lines on one plot."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    span = 2.0
    for e in "AB":
        sol, an, scfg, _ = R[e]
        nu, sp, fw = an["nu"], an["sp"], an["wd"]["fwhm"]
        pk = nu[int(np.argmax(sp))]
        ax.plot(nu - pk, sp, color=ABCOL[e], lw=2.2,
                label=f"{e}  (FWHM {fmt(fw, '{:.1f}')} Hz)")
        span = max(span, 6 * fw if np.isfinite(fw) else 0, 6 * scfg["nat"])
    draw_measured(ax)
    span = max(span, measured_span())
    ax.set_xlim(-span, span)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("frequency offset  (Hz)")
    ax.set_ylabel("signal")
    ax.set_title("the NMR line you would measure")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(color=P.GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def ab_maps(R):
    """Field maps of A and B on one colour scale."""
    fig = plt.figure(figsize=(7.4, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.3)
    lim = 1e-4
    for e in "AB":
        an = R[e][1]
        v = np.abs(an["raw"] - an["raw"].mean())
        lim = max(lim, np.percentile(v, 99) * 1.4)
    for i, e in enumerate("AB"):
        sol, an, scfg, _ = R[e]
        ax = fig.add_subplot(gs[0, i])
        x, z, j = sol["x"], sol["z"], sol["j"]
        d = np.where(sol["liq"], sol["dB"] - an["raw"].mean(), np.nan)[:, j, :]
        m = ax.imshow(d.T, origin="lower", aspect="auto", cmap=P.FIELD_CMAP,
                      vmin=-lim, vmax=lim, interpolation="bilinear",
                      extent=[x[0], x[-1], z[0], z[-1]])
        ax.add_patch(plt.Rectangle((-an["r"], an["z0"]), 2*an["r"],
                                   an["z1"] - an["z0"], fill=False, ec=P.INK,
                                   lw=1.2, ls=(0, (4, 3))))
        ri = scfg["t_OD"] / 2 + 0.5
        ax.set_xlim(-ri, ri)
        ax.set_ylim(scfg["t_tip"] - 1, scfg["t_tip"] + scfg["t_wall"]
                    + scfg["t_fill"] + 1)
        ax.set_title(f"field in the sample, {e}", color=ABCOL[e],
                     fontweight="bold")
        ax.set_xlabel("x  (mm)")
        ax.set_ylabel("height  (mm)")
    cb = fig.colorbar(m, cax=fig.add_subplot(gs[0, 2]))
    cb.outline.set_visible(False)
    cb.set_label("field change before shimming (ppm)", color=P.MUTED,
                 fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.14, top=0.88)
    return fig


def ab_results(R):
    NA = numbers(R["A"][1], R["A"][2])
    NB = numbers(R["B"][1], R["B"][2])
    # Headline on the leftover field, not the FWHM: a split line can have a
    # narrow FWHM and still be much worse.
    ra = NA["field spread after shimming"][0]
    rb = NB["field spread after shimming"][0]
    if ra > 0 and rb > 0:
        if abs(rb - ra) / ra < 0.03:
            st.markdown(f"**A and B are equally homogeneous** ({ra:.1f} ppb "
                        "left after shimming).")
        else:
            better, k = ("B", ra / rb) if rb < ra else ("A", rb / ra)
            st.markdown(f"**Setup {better} is more homogeneous: {k:.1f}× "
                        f"less field spread after shimming** "
                        f"(A {ra:.1f} ppb, B {rb:.1f} ppb).")
    rows = ["| | <span style='color:#2f6fb5'>**A**</span> | "
            "<span style='color:#d9822b'>**B**</span> | B − A |",
            "|---|---|---|---|"]
    for k, (va, f) in NA.items():
        vb = NB[k][0]
        dv = vb - va if np.isfinite(va) and np.isfinite(vb) else np.nan
        df = f.replace("{:", "{:+").replace("++", "+") if "+" not in f else f
        rows.append(f"| {k} | {fmt(va, f)} | {fmt(vb, f)} | {fmt(dv, df)} |")
    st.markdown("\n".join(rows), unsafe_allow_html=True)
    show(ab_lines(R))
    show(ab_maps(R))
    with st.expander("More results"):
        for e in "AB":
            st.markdown(f"<b style='color:{ABCOL[e]}'>Setup {e}</b>",
                        unsafe_allow_html=True)
            order_table(R[e][1], R[e][2])
        f2, a2 = plt.subplots(1, 1, figsize=(7.4, 3.0))
        log_line(a2, [(R["A"][1], ABCOL["A"], "A"),
                      (R["B"][1], ABCOL["B"], "B")],
                 max(R["A"][2]["nat"], R["B"][2]["nat"]))
        f2.tight_layout()
        try:
            show(f2)
        except Exception:
            plt.close(f2)
        for e in "AB":
            sol, an, scfg, _ = R[e]
            st.caption(f"{e}: {an['n']} voxels, voxel {sol['V']} mm, "
                       f"detection {an['z0']:.1f} to {an['z1']:.1f} mm, "
                       f"radius {an['r']:.1f} mm, shimming: "
                       f"{C.SHIM_LABELS[int(scfg['sh_order'])]}.")


with right:
    st.subheader("Preview")
    pcols = st.columns(len(SETUPS))
    frame = None
    if AB:                          # same scale for both pictures
        fr = [preview_frame(cfg_of(e)) for e in SETUPS]
        frame = (max(f[0] for f in fr), min(f[1] for f in fr),
                 max(f[2] for f in fr))
    for c, e in zip(pcols, SETUPS):
        with c:
            pcfg = cfg_of(e)
            if AB:
                st.markdown(f"<b style='color:{ABCOL[e]}'>Setup {e}</b>"
                            + (" ✎" if e == E else ""),
                            unsafe_allow_html=True)
            try:
                show(preview(pcfg, C.detection(pcfg), compact=AB, frame=frame),
                     tight=AB)
            except Exception as ex:
                st.error(f"This geometry cannot be drawn: {ex}")

    run = st.button("▶  Run" + (" A and B" if AB else ""), type="primary",
                    width="stretch",
                    help="Computes the field. Takes a second for a bare "
                         "tube, a few seconds for a full probe.")
    if run or ss.get("auto_run"):
        ss.auto_run = False
        with st.spinner("computing the field…"):
            for e in SETUPS:
                sig = json.dumps(cfg_of(e), sort_keys=True)
                ss.solved[e] = sig
                solve_cached(sig)

    R = {e: result_for(e) for e in SETUPS}
    if any(r is None for r in R.values()):
        st.info("Press **Run** to compute the field.")
        st.stop()
    stale = [e for e in SETUPS if R[e][3]]
    if stale:
        who = "the probe" if not AB else "setup " + " and ".join(stale)
        st.warning(f"You changed {who} since the last run. Press **Run** "
                   "to update the results.")
    st.subheader("Result" if not AB else "A vs B")
    for e in SETUPS:
        if R[e][0]["note"]:
            st.info((f"Setup {e}: " if AB else "") + R[e][0]["note"])
    bad = [e for e in SETUPS if R[e][1] is None]
    if bad:
        st.error(("The detection region has no sample in it"
                  + (f" (setup {', '.join(bad)})" if AB else "")
                  + ". Check its heights against the liquid column and the "
                    "tube tip."))
        st.stop()
    if AB:
        ab_results(R)
    else:
        single_results(*R["A"][:3])
