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
# Widget values live in st.session_state["v"], not in the widgets, so a part
# keeps its settings when you switch it off and on again (Streamlit forgets
# the value of a widget that is not on screen).
ss = st.session_state
if "v" not in ss:
    ss.v = dict(C.BLANK)
    ss.custom = []
    ss.preset = "Blank (tube only)"
    ss.auto_run = True
S = ss.v

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


def _push_all():
    """Copy stored values into the widget keys (after a preset or a load)."""
    for k, v in S.items():
        ss["w_" + k] = v
    for p in ss.custom:
        for k, v in p.items():
            if k != "id":
                ss[f"cw_{p['id']}_{k}"] = v


def apply_preset():
    name = ss.preset_pick
    ss.preset = name
    ss.v = dict(C.PRESETS[name][0])
    ss.custom = []
    for k in [k for k in ss.keys() if str(k).startswith(("w_", "cw_"))]:
        del ss[k]
    ss.auto_run = True


def W(widget, label, key, **kw):
    """A widget whose value is remembered in S[key]."""
    wk = "w_" + key
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


def cfg_now():
    c = dict(S)
    c["custom"] = [{k: v for k, v in p.items() if k != "id"} for p in ss.custom]
    return c


# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("B₀ builder")
sb.caption("Build your probe one part at a time and see how it disturbs "
           "the magnetic field in your sample.")

names = list(C.PRESETS)
sb.radio("Start from", names, index=names.index(ss.preset),
         key="preset_pick", on_change=apply_preset,
         help="Loads a complete setup. You can change anything afterwards.")
sb.caption(C.PRESETS[ss.preset][1])

sb.divider()
sb.subheader("Save / load")
sb.download_button("Save this setup (.json)",
                   json.dumps(cfg_now(), indent=1),
                   file_name="b0_setup.json", mime="application/json",
                   width="stretch")
up = sb.file_uploader("Load a setup", type="json",
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
        ss.v = new
        S = ss.v
        ss.custom = [dict(C.CUSTOM_DEFAULT, **p, id=uuid.uuid4().hex[:8])
                     for p in d.get("custom", [])]
        for k in [k for k in ss.keys() if str(k).startswith(("w_", "cw_"))]:
            del ss[k]
        ss.loaded_id = up.file_id
        ss.auto_run = True
        st.rerun()
    except Exception as e:
        sb.error(f"Could not read that file: {e}")

sb.divider()
with sb.expander("Accuracy (advanced)"):
    W(st.select_slider, "voxel size (mm)", "V", options=C.VOXELS,
      help="The cube size the space is cut into. Smaller = more accurate "
           "but slower and more memory. 0.25 mm is a good default.")
    st.caption("Big probes are coarsened automatically if they would not "
               "fit in memory.")
sb.caption("Need sweeps, recommendations or the microwave models? "
           "Run the full app: `python -m streamlit run app.py`")


# ----------------------------------------------------------------- layout
left, right = st.columns([1.0, 1.45], gap="large")

with left:
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
    for p in list(ss.custom):
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
                ss.custom = [q for q in ss.custom if q["id"] != pid]
                st.rerun()
    if st.button("＋ Add a custom part", width="stretch",
                 help="Any cylinder, ring or box of any material: a holder, "
                      "a spacer, a screw, a lens..."):
        ss.custom.append(dict(C.CUSTOM_DEFAULT, id=uuid.uuid4().hex[:8]))
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
        W(st.select_slider, "shimming", "sh_order", options=[0, 1, 2],
          format_func=lambda o: {0: "none", 1: "linear (x, y, z)",
                                 2: "up to 2nd order (z², xy, …)"}[o],
          help="Which correction coils the spectrometer uses to flatten "
               "the field. 'none' only removes the average offset.")
        a, b = st.columns(2)
        with a:
            num("¹H frequency (MHz)", "f_MHz", 10.0, 1500.0, 10.0)
        with b:
            num("natural linewidth (Hz)", "nat", 0.05, 100.0, 0.05,
                help="The width your sample would have in a perfect field.")


# ---------------------------------------------------------------- preview
COLORS = {"glass": "#9aa7b8", "sample": "#f0b05a", "RF coil": "#c0703a",
          "waveguide": "#b59a3c", "shield": "#7f8c9b",
          "modulation coil": "#a0522d", "custom": "#5aa58a"}


def preview(cfg, det):
    """x-z cut through the axis, drawn with the same builders the solver
    uses, at fine resolution. Instant, no FFT."""
    r, zlo, zhi = C.extent(cfg)
    zhi = cfg["t_tip"] + cfg["t_wall"] + cfg["t_fill"] + 3.0
    for kind, p, _ in C._parts(cfg):
        zhi = max(zhi, p["z1"] + 1.0)
    if cfg["c_on"]:
        zhi = max(zhi, cfg["c_zc"] + cfg["c_len"]/2 + 1.0)
    zlo = max(zlo - 1.0, cfg["t_tip"] - 8.0) if cfg["w_on"] else zlo - 1.0
    half = max(min(r + 1.0, 22.0), 6.0)
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
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
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
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=8.5, frameon=False)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("height  (mm)")
    ax.set_title("cut through the axis")
    fig.tight_layout()
    return fig


def show(fig):
    """Fixed-size image, so a narrow probe is not blown up to full width."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    st.image(buf, width="stretch")


@st.cache_resource(max_entries=3, show_spinner=False)
def solve_cached(sig):
    return C.solve(json.loads(sig))


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
               label=f"perfect field ({nat:g} Hz)")
    ax[1].plot(nu - pk, sp, color="#2f6fb5", lw=2.2, label="your probe")
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


with right:
    cfg = cfg_now()
    det = C.detection(cfg)

    st.subheader("Preview")
    try:
        show(preview(cfg, det))
    except Exception as e:
        st.error(f"This geometry cannot be drawn: {e}")

    sig = json.dumps(cfg, sort_keys=True)
    run = st.button("▶  Run", type="primary", width="stretch",
                    help="Computes the field. Takes a second for a bare "
                         "tube, a few seconds for a full probe.")
    if run or ss.get("auto_run"):
        ss.auto_run = False
        ss.solved_sig = sig
        with st.spinner("computing the field…"):
            solve_cached(sig)

    if "solved_sig" in ss:
        scfg = json.loads(ss.solved_sig)
        # detection, shimming and frequency are cheap: always use the live ones
        for k in ("d_auto", "d_r", "d_z0", "d_z1", "sh_order", "f_MHz", "nat"):
            scfg[k] = cfg[k]
        sol = solve_cached(ss.solved_sig)
        geom_keys = [k for k in cfg if k not in
                     ("d_auto", "d_r", "d_z0", "d_z1", "sh_order", "f_MHz",
                      "nat")]
        old = json.loads(ss.solved_sig)
        if any(old.get(k) != cfg.get(k) for k in geom_keys):
            st.warning("You changed the probe since the last run. "
                       "Press **Run** to update the results.")
        an = C.analyse(sol, scfg)
        st.subheader("Result")
        if sol["note"]:
            st.info(sol["note"])
        if an is None:
            st.error("The detection region has no sample in it. Check its "
                     "heights against the liquid column and the tube tip.")
        else:
            wd, rep = an["wd"], an["rep"]
            nat = scfg["nat"]
            k = st.columns(3)
            k[0].metric("linewidth (FWHM)",
                        f"{wd['fwhm']:.1f} Hz" if np.isfinite(wd["fwhm"])
                        else "n/a",
                        help="Full width at half maximum of the simulated line.")
            extra = wd["fwhm"] - nat
            k[1].metric("added by the field", f"{extra:+.1f} Hz",
                        help="How much wider the line is than in a perfect "
                             "magnet.")
            k[2].metric("field spread after shimming",
                        f"{rep['shim_ppm_rms']*1000:.1f} ppb",
                        help="rms of the leftover field over the detected "
                             "region. 1 ppb = 0.001 ppm.")
            st.pyplot(results_figure(sol, an, scfg))

            with st.expander("More results"):
                t = st.columns(4)
                t[0].metric("spread before shimming",
                            f"{rep['raw_ppm_pp']:.3f} ppm",
                            help="Peak to peak, over the detected region.")
                t[1].metric("spread after shimming",
                            f"{rep['shim_ppm_pp']:.4f} ppm",
                            help="Peak to peak.")
                t[2].metric("width at 0.55 %",
                            f"{wd['w055']:.1f} Hz" if np.isfinite(wd["w055"])
                            else "n/a")
                t[3].metric("width at 0.11 %",
                            f"{wd['w011']:.1f} Hz" if np.isfinite(wd["w011"])
                            else "n/a")
                st.caption("0.55 % and 0.11 % are the standard NMR "
                           "lineshape test: they show the broad foot of the "
                           "line that the FWHM hides.")
                f2, a2 = plt.subplots(1, 2, figsize=(7.4, 3.0))
                P.hist_panel(a2[0], an["res"], scfg["f_MHz"])
                a2[0].set_title("leftover field after shimming")
                pk = an["nu"][int(np.argmax(an["sp"]))]
                a2[1].plot(an["nu"] - pk, an["sp"], color="#2f6fb5", lw=1.8)
                w11 = wd["w011"] if np.isfinite(wd["w011"]) else 10*nat
                xr = max(0.8 * w11, 3 * nat)
                a2[1].set_xlim(-xr, xr)
                for frac, lab in ((0.5, "50 %"), (0.0055, "0.55 %"),
                                  (0.0011, "0.11 %")):
                    a2[1].axhline(frac, color="#c8ccd2", lw=0.8)
                    a2[1].text(xr, frac, f"{lab} ", fontsize=8,
                               color=P.MUTED, va="bottom", ha="right")
                a2[1].set_yscale("log")
                a2[1].set_ylim(3e-4, 1.6)
                a2[1].set_xlabel("frequency offset  (Hz)")
                a2[1].set_title("same line, log scale")
                a2[1].grid(color=P.GRID, lw=0.8)
                f2.tight_layout(w_pad=2.5)
                st.pyplot(f2)
                st.caption(f"{an['n']} voxels detected, voxel "
                           f"{sol['V']} mm, detection "
                           f"{an['z0']:.1f} to {an['z1']:.1f} mm, radius "
                           f"{an['r']:.1f} mm.")
    else:
        st.info("Press **Run** to compute the field.")
