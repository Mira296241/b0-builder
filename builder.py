"""
B0 builder: the simple version of the B0 tool.

Start from a preset, add only the parts your probe has, press Run.
Run with:   python -m streamlit run builder.py

The full version with recommendations, microwave models and susceptibility
matching is app.py. Both use the same physics (b0core.py).
"""
import io
import json
import os
import uuid

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

import builder_core as C
from b0core import MATERIALS_PPM, SOURCES
import plotting as P

try:
    import plotly.graph_objects as go
except Exception:                                  # 3D view is optional
    go = None

# ------------------------------------------------ edit before sharing
FEEDBACK_URL = "https://github.com/Mira296241/b0-builder/issues"
AUTHOR = ""      # e.g. "Name, group, institute". Shown on the About page.
CITE = ""        # how to cite the tool. Shown on the About page when set.

P.use_style()
st.set_page_config(page_title="B0 builder", layout="wide")

# ------------------------------------------------------------------ state
# Two setups, A and B. Widget values live in ss.cfg[A or B], not in the
# widgets, so a part keeps its settings when you switch it off and on again
# (Streamlit forgets the value of a widget that is not on screen) and when
# you switch between editing A and B. Widget keys carry the setup letter.
ss = st.session_state
FIRST = "Blank (tube only)"
if "cfg" not in ss:
    ss.cfg = {"A": dict(C.BLANK), "B": dict(C.BLANK)}
    ss.cust = {"A": [], "B": []}
    ss.preset = {"A": FIRST, "B": FIRST}
    ss.solved = {}
    ss.edit = "A"
    ss.auto_run = True
    code = st.query_params.get("s")
    if code:                                   # opened from a share link
        try:
            c0, cu = C.decode_setup(code)
            ss.cfg["A"] = c0
            ss.cust["A"] = [dict(p, id=uuid.uuid4().hex[:8]) for p in cu]
            ss.preset["A"] = None
            ss.from_link = True
        except Exception:
            ss.link_error = True
AB = bool(ss.get("ab_on", False))
if not AB:
    ss.edit = "A"
E = ss.get("edit", "A")
S = ss.cfg[E]
WK = f"w{E}_"                       # widget-key prefix for this setup
ABCOL = {"A": "#2f6fb5", "B": "#d9822b"}

LIQ_NAMES = ("water", "D2O", "toluene", "toluene-d8", "DMSO", "chloroform-d",
             "acetone-d6", "acetonitrile-d3", "benzene-d6", "methanol-d4",
             "THF-d8", "methanol", "ethanol", "glycerol/water 60:40")
BASE_MATS = [m for m in MATERIALS_PPM
             if not m.startswith(("custom", "__"))]
SOLIDS = [m for m in BASE_MATS if m not in ("air", "vacuum") + LIQ_NAMES]
LIQUIDS = [m for m in BASE_MATS if m in LIQ_NAMES]


def mat_label(m):
    if m == C.MATCHED:
        return "matched to the sample"
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
    name = ss.get(f"preset_pick_{e}")
    if not name:
        return
    _forget_widgets(e)
    ss.preset[e] = name
    ss.cfg[e] = dict(C.PRESETS[name][0])
    ss.cust[e] = [dict(q, id=uuid.uuid4().hex[:8])
                  for q in C.PRESET_PARTS.get(name, [])]
    c0 = ss.cfg[e]
    # look through the coil windows when they face y
    ss.cut_plane = ("y" if c0["c_on"] and c0["c_nslots"]
                    and round(c0["c_phase"]) % 180 == 90 else "x")
    ss.auto_run = True


def copy_setup(src, dst):
    _forget_widgets(dst)
    ss.cfg[dst] = dict(ss.cfg[src])
    ss.cust[dst] = [dict(p, id=uuid.uuid4().hex[:8]) for p in ss.cust[src]]
    ss.preset[dst] = ss.preset[src]
    ss.pop(f"preset_pick_{dst}", None)
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
    if m.startswith("custom") or m == C.MATCHED:
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
if ss.pop("from_link", False):
    sb.success("Loaded the setup from the link.")
if ss.pop("link_error", False):
    sb.error("That link could not be read. Starting from a blank setup.")


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
cur = ss.preset[E]
sb.selectbox("Start from" + (f" (setup {E})" if AB else ""), names,
             index=names.index(cur) if cur in names else None,
             key=f"preset_pick_{E}", on_change=apply_preset,
             placeholder="a loaded setup",
             help="Loads a complete setup. You can change anything "
                  "afterwards.")
if cur in C.PRESETS:
    sb.caption(C.PRESETS[cur][1])

sb.divider()
sb.subheader("Save, load, share")
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
        new, cu = C.coerce(json.loads(up.getvalue()))
        _forget_widgets(E)
        ss.cfg[E] = new
        ss.cust[E] = [dict(p, id=uuid.uuid4().hex[:8]) for p in cu]
        ss.preset[E] = None
        ss.pop(f"preset_pick_{E}", None)
        ss.loaded_id = up.file_id
        ss.auto_run = True
        st.rerun()
    except Exception as e:
        sb.error(f"Could not read that file: {e}")

if sb.button("Create a link to this setup", width="stretch",
             help="Packs the whole setup into a web address. Whoever opens "
                  "it sees exactly this probe."):
    ss.link_code = C.encode_setup(cfg_now())
    st.query_params["s"] = ss.link_code
if ss.get("link_code"):
    base = None
    try:
        base = str(st.context.url).split("?")[0]
    except Exception:
        pass
    if base and "localhost" not in base:
        sb.code(f"{base}?s={ss.link_code}", language=None)
        sb.caption("Copy this link and send it.")
    else:
        sb.caption("Done: the address bar of your browser now holds this "
                   "setup. Copy the address from there and send it.")

sb.divider()
with sb.expander("Accuracy (advanced)" + (f", setup {E}" if AB else "")):
    W(st.select_slider, "voxel size (mm)", "V", options=C.VOXELS,
      help="The cube size the space is cut into. Smaller = more accurate "
           "but slower and more memory. 0.25 mm is a good default.")
    mb = C.memory_budget_mb()
    st.caption(f"This server has about **{mb/1000:.1f} GB**. The solver "
               f"keeps its grid under {C.cell_limit()/1e6:.0f} million cells "
               "and coarsens the voxel automatically if a probe would not "
               "fit. On your own computer the limit follows your own RAM.")
    if st.button("Check the solver", width="stretch",
                 help="Solves a case with an exact textbook answer (a "
                      "uniform sphere) at this voxel size and reports the "
                      "error."):
        chk = C.self_check(float(S["V"]))
        st.caption(f"Uniform sphere, voxel {chk['voxel']} mm: field inside "
                   f"flat to **{chk['inside']*100:.1f} %**, dipole field "
                   f"outside right to **{chk['outside']*100:.1f} %** of the "
                   "largest exact value.")
if FEEDBACK_URL:
    sb.link_button("Report a problem or suggest something", FEEDBACK_URL,
                   width="stretch")


# ----------------------------------------------------------------- layout
left, right = st.columns([1.0, 1.45], gap="large")

with left:
    if not ss.get("tour_done"):
        with st.expander("New here? A 30-second tour", expanded=True):
            st.markdown(
                "1. **Start from** a preset in the sidebar, for example "
                "*Standard 5 mm tube*.\n"
                "2. Set your **sample tube** below, then switch on the "
                "**probe parts** your probe has.\n"
                "3. The **preview** on the right redraws as you type. Press "
                "**Run** to compute the field.\n"
                "4. Read the **linewidth** and the simulated NMR line. "
                "*More results* shows how it depends on shimming.\n"
                "5. Try the other tabs: **Sweep** one parameter, look at "
                "the probe in **3D**, or read **About** for the method.\n"
                "6. Want to compare two versions? Switch on **Compare two "
                "setups** in the sidebar.")
            if st.button("Got it, hide this"):
                ss.tour_done = True
                st.rerun()

    if AB:
        st.markdown(f"<h3 style='margin-bottom:0'>Setup <span style='color:"
                    f"{ABCOL[E]}'>{E}</span></h3>", unsafe_allow_html=True)
        st.caption("Switch between A and B in the sidebar.")
    else:
        st.subheader("Your probe")
    st.caption("All heights in mm, measured from **z = 0**, the probe "
               "reference (the waveguide top, unless you move it).")

    # ---------------------------------------------------------- tube
    with st.expander("Sample tube", expanded=True):
        a, b = st.columns(2)
        with a:
            num("outer diameter", "t_OD", 1.0, 12.0, 0.1,
                help="Outside diameter of the tube (mm).")
            num("liquid column", "t_fill", 2.0, 80.0, 1.0,
                help="Height of the liquid (mm).")
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
            T = num("sample temperature (°C)", "t_temp", -20.0, 100.0, 5.0,
                    help="Liquids expand when they warm up, so their "
                         "susceptibility changes (about 0.1 ppm per 10 °C "
                         "for organic solvents). Measured slopes from "
                         "Hoffman 2022, valid for the liquid between 0 and "
                         "70 °C. For a frozen sample use 'custom χ'.")
            if T != C.T_REF:
                note = f"χ at {T:g} °C = **{C.chi_sample(S):+.2f} ppm**"
                if l in C.TEMP_ESTIMATED:
                    note += " (estimated slope)"
                if not 0 <= T <= 70:
                    note += ". Outside 0 to 70 °C this is an extrapolation"
                st.caption(note + ".")
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

    # ----------------------------------------------------------- plugs
    if W(st.toggle, "Susceptibility plugs", "p_on",
         help="Solid plugs inside the tube, below and above the sample, "
              "as in a Shigemi tube. Matched to the sample they hide its "
              "two ends from the coil, so a short sample behaves like a "
              "long one."):
        with st.expander("Plug settings", expanded=False):
            W(st.checkbox, "plug below the sample", "p_bot")
            if S["p_bot"]:
                a, b = st.columns(2)
                with a:
                    num("length", "p_bot_len", 0.5, 40.0, 0.5,
                        help="The sample sits on top of it.")
                with b:
                    mat("material", "p_bot_mat", [C.MATCHED] + SOLIDS)
            W(st.checkbox, "plunger above the sample", "p_top")
            if S["p_top"]:
                a, b = st.columns(2)
                with a:
                    num("length ", "p_top_len", 0.5, 40.0, 0.5)
                with b:
                    mat("material ", "p_top_mat", [C.MATCHED] + SOLIDS)
            st.caption("PMMA (acrylic) is the plastic closest to water. "
                       "'Matched' means exactly the sample's value.")

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
         help="A metal waveguide below the tube. Its top is at z = 0 "
              "unless you move it."):
        with st.expander("Waveguide settings", expanded=False):
            a, b = st.columns(2)
            with a:
                num("inner diameter", "w_id", 1.0, 20.0, 0.1)
                num("length modelled", "w_len", 2.0, 40.0, 1.0,
                    help="Only its top end matters. 20 mm is plenty.")
            with b:
                num("wall", "w_wall", 0.1, 5.0, 0.05)
                num("top height", "w_top", -40.0, 20.0, 0.5,
                    help="0 = at the probe reference. A negative value "
                         "lowers the waveguide away from the sample, "
                         "without moving anything else.")
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

    # ---------------------------------------------------- extra parts
    parts = ss.cust[E]
    if ss.pop("_reset_inc", False):          # set by the buttons below
        ss.pop(WK + "_inc", None)
    if "_pending_sel" in ss:
        ss[WK + "_sel"] = ss.pop("_pending_sel")

    def part_title(q):
        return q.get("name") or f"{q['kind']} · {q['mat']}"

    if parts:
        st.markdown(f"**Extra parts ({len(parts)})**")
        groups = list(dict.fromkeys(part_title(q) for q in parts))
        gk = WK + "_inc"
        if gk not in ss:
            ss[gk] = [g for g in groups
                      if any(q.get("on", True) for q in parts
                             if part_title(q) == g)]
        ss[gk] = [g for g in ss[gk] if g in groups]
        inc = st.multiselect(
            "included in the calculation", groups, key=gk,
            help="Remove a part here to see what it does to the field. "
                 "It stays in the list, so you can put it back.")
        for q in parts:
            q["on"] = part_title(q) in inc
        ids = [q["id"] for q in parts]
        sk = WK + "_sel"
        if ss.get(sk) not in ids + [None]:
            ss.pop(sk, None)
        pid = st.selectbox(
            "edit a part", ids, index=None, key=sk,
            placeholder="choose a part to edit its size or material",
            format_func=lambda i: next(
                f"{part_title(q)}  ·  z {q['z0']:g} to {q['z1']:g}"
                for q in parts if q["id"] == i))
        p = next((q for q in parts if q["id"] == pid), None)
        if p is not None:
            with st.container(border=True):
                def cw(widget, label, k, **kw):
                    key = f"cw_{pid}_{k}"
                    if key not in ss:
                        ss[key] = p[k]
                    p[k] = widget(label, key=key, **kw)
                    return p[k]

                def cnum(label, k, lo, hi, step):
                    return cw(st.number_input, label, k, min_value=float(lo),
                              max_value=float(hi), step=float(step))

                cw(st.text_input, "name", "name",
                   help="Parts with the same name are shown, coloured and "
                        "switched on and off together.")
                cw(st.radio, "shape", "kind",
                   options=["cylinder / ring", "box"], horizontal=True)
                opts = SOLIDS if p["mat"] in SOLIDS else SOLIDS + [p["mat"]]
                m = cw(st.selectbox, "material", "mat", options=opts,
                       format_func=mat_label)
                source_note(m)
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
                        cnum("inner radius (0 = solid)", "r_in", 0.0, 20.0,
                             0.1)
                        cnum("centre x", "cx", -20.0, 20.0, 0.1)
                    with b:
                        cnum("outer radius", "r_out", 0.05, 20.0, 0.1)
                        cnum("centre y", "cy", -20.0, 20.0, 0.1)
                with a:
                    cnum("bottom height", "z0", -40.0, 80.0, 0.05)
                with b:
                    cnum("top height", "z1", -40.0, 80.0, 0.05)
                cnum("fill fraction", "fill", 0.05, 1.0, 0.05)
                if st.button("Remove this part", key=f"rm_{pid}"):
                    ss.cust[E] = [q for q in parts if q["id"] != pid]
                    ss["_reset_inc"] = True
                    st.rerun()
    if st.button("＋ Add a part", width="stretch",
                 help="Any cylinder, ring or box of any material: a holder, "
                      "a spacer, a screw, a lens..."):
        new_id = uuid.uuid4().hex[:8]
        ss.cust[E].append(dict(C.CUSTOM_DEFAULT, name="my part", id=new_id))
        ss["_reset_inc"] = True
        ss["_pending_sel"] = new_id
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
        W(st.checkbox, "smooth coil edges (more realistic)", "d_soft",
          help="A real coil does not stop seeing the sample at a sharp "
               "line: its sensitivity fades out beyond the window. With "
               "this on, the sample is weighted by a simple coil profile "
               "(flat inside the window, fading over the edge width "
               "outside) instead of a sharp-edged box. Shimming then also "
               "optimises the weighted signal.")
        if S["d_soft"]:
            num("edge width (mm)", "d_edge", 0.2, 10.0, 0.1,
                help="How far beyond the window the coil still sees "
                     "sample. About half the coil radius is typical.")
        W(st.select_slider, "shimming, up to", "sh_order",
          options=[0, 1, 2, 3, 4], format_func=lambda o: C.SHIM_LABELS[o],
          help="Which correction coils the spectrometer uses to flatten "
               "the field. 'none' only removes the average offset. A real "
               "Bruker room-temperature shim set goes to 4th order and "
               "beyond along z, so 4 is closest to the real magnet; 2 is "
               "pessimistic.")
        a, b = st.columns(2)
        with a:
            num("magnet: ¹H frequency (MHz)", "f_MHz", 10.0, 1500.0, 10.0,
                help="The size of the magnet, quoted the usual way.")
            W(st.selectbox, "observed nucleus", "nuc",
              options=list(C.NUCLEI), format_func=lambda n: C.NUC_LABEL[n],
              help="Linewidths in Hz scale with the nucleus frequency: the "
                   "same field error gives a 4 times narrower ¹³C line "
                   "than ¹H.")
        with b:
            num("linewidth without the probe (Hz)", "nat", 0.05, 100.0, 0.05,
                help="The width you would get if this probe disturbed "
                     "nothing: the sample's natural width plus whatever the "
                     "magnet itself leaves after shimming. Best estimate: "
                     "the linewidth measured on the same magnet and sample "
                     "with a good commercial probe. The tool adds this "
                     "probe's broadening on top.")
            st.caption(f"{C.NUC_LABEL[S['nuc']]} at "
                       f"**{C.f_obs(S):.1f} MHz**")
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
                    f_MHz=C.f_obs(S))
            except Exception:
                ss.meas = None
            if ss.meas is None:
                st.error("Could not read two numeric columns from that file.")
            else:
                st.caption(f"Read {len(ss.meas['nu'])} points, measured FWHM "
                           f"**{ss.meas['fwhm']:.1f} Hz**.")


# ------------------------------------------------------------- figures
def colour(label):
    return COLORS.get(label) or C.PART_COLORS.get(label) or COLORS["custom"]


BASE_ORDER = ["glass", "sample", "plug", "waveguide", "modulation coil",
              "shield"]

COLORS = {"glass": "#9aa7b8", "sample": "#f0b05a", "RF coil": "#c0703a",
          "waveguide": "#b59a3c", "shield": "#7f8c9b",
          "modulation coil": "#a0522d", "custom": "#5aa58a",
          "plug": "#b9a3dd"}


def preview_frame(cfg):
    """(half-width, z_lo, z_hi) of the preview picture for one setup."""
    r, zlo, zhi = C.extent(cfg)
    zhi = C.liquid_span(cfg)[1] + 3.0
    if cfg["p_on"] and cfg["p_top"]:
        zhi += cfg["p_top_len"]
    for kind, p, _ in C._parts(cfg):
        zhi = max(zhi, p["z1"] + 1.0)
    if cfg["c_on"]:
        zhi = max(zhi, cfg["c_zc"] + cfg["c_len"]/2 + 1.0)
    if cfg["w_on"]:
        zlo = max(zlo - 1.0, min(cfg["t_tip"], cfg["w_top"]) - 8.0)
    else:
        zlo = zlo - 1.0
    return max(min(r + 1.0, 22.0), 6.0), zlo, zhi


def preview(cfg, det, compact=False, frame=None):
    """x-z cut through the axis, drawn with the same builders the solver
    uses, at fine resolution. Instant, no FFT. `frame` forces the picture
    limits, so A and B are drawn on the same scale."""
    half, zlo, zhi = frame or preview_frame(cfg)
    h = max(0.04, 2*half/500)
    x = np.arange(-half + h/2, half, h)
    z = np.arange(zlo + h/2, zhi, h)
    along_y = ss.get("cut_plane", "x") == "y"
    if along_y:
        # look along x instead: turn the probe a quarter turn about z
        cfg = dict(cfg, c_phase=cfg["c_phase"] - 90.0, custom=[
            dict(q, cx=q["cy"], cy=-q["cx"], sx=q["sy"], sy=q["sx"])
            for q in cfg.get("custom", [])])
    _, masks = C.build_all(cfg, x, np.array([0.0]), z)
    img = np.ones((len(z), len(x), 3)) * np.array(mpl.colors.to_rgb("#f3f5f8"))
    shown = []
    extra = [k for k in masks if k not in BASE_ORDER + ["RF coil"]]
    for name in extra + BASE_ORDER + ["RF coil"]:
        if name in masks:
            m = masks[name][:, 0, :].T
            if m.any():
                img[m] = mpl.colors.to_rgb(colour(name))
                shown.append(name)
    shown = [n for n in BASE_ORDER + ["RF coil"] + extra if n in shown]
    fig, ax = plt.subplots(figsize=(3.6, 5.6) if compact else (6.2, 5.0))
    ax.imshow(img, origin="lower", aspect="equal", interpolation="nearest",
              extent=[x[0]-h/2, x[-1]+h/2, z[0]-h/2, z[-1]+h/2])
    z0, z1, rr = det
    ax.add_patch(plt.Rectangle((-rr, z0), 2*rr, z1 - z0, fill=False,
                               ec=P.INK, lw=1.4, ls=(0, (4, 3))))
    ax.axhline(0.0, color=P.INK, lw=0.8, alpha=0.5)
    ax.text(x[0], 0.0, " z = 0", fontsize=8, color=P.MUTED, va="bottom")
    handles = [mpl.patches.Patch(color=colour(n), label=n) for n in shown]
    handles.append(mpl.lines.Line2D([], [], color=P.INK, ls=(0, (4, 3)),
                                    lw=1.4, label="detected region"))
    if compact:
        ax.legend(handles=handles, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, -0.11), fontsize=9, frameon=False)
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                  fontsize=8.5, frameon=False)
    ax.set_xlabel("y  (mm)" if along_y else "x  (mm)")
    ax.set_ylabel("height  (mm)")
    ax.set_title("cut through the axis")
    fig.tight_layout()
    return fig


def fig_bytes(fig, fmt="png", tight=False):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=170,
                bbox_inches="tight" if tight else None)
    return buf.getvalue()


def show(fig, tight=False, pdf=False):
    """Draw a figure as an image (more robust than st.pyplot across Python
    versions). Returns (png, pdf) bytes for the download buttons."""
    png = fig_bytes(fig, "png", tight)
    out = (png, fig_bytes(fig, "pdf", tight) if pdf else None)
    plt.close(fig)
    st.image(png, width="stretch")
    return out


@st.cache_resource(max_entries=4, show_spinner=False)
def solve_cached(sig):
    return C.solve(json.loads(sig))


def fmt(v, f):
    return f.format(v) if v is not None and np.isfinite(v) else "n/a"


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


def field_map(ax, sol, an, scfg, lim):
    x, z, j = sol["x"], sol["z"], sol["j"]
    d = np.where(sol["liq"], sol["dB"] - an["raw"].mean(), np.nan)[:, j, :]
    m = ax.imshow(d.T, origin="lower", aspect="auto", cmap=P.FIELD_CMAP,
                  vmin=-lim, vmax=lim, interpolation="bilinear",
                  extent=[x[0], x[-1], z[0], z[-1]])
    ax.add_patch(plt.Rectangle((-an["r"], an["z0"]), 2*an["r"],
                               an["z1"] - an["z0"], fill=False, ec=P.INK,
                               lw=1.3, ls=(0, (4, 3))))
    ri = scfg["t_OD"] / 2 + 0.5
    zl0, zl1 = C.liquid_span(scfg)
    ax.set_xlim(-ri, ri)
    ax.set_ylim(zl0 - scfg["t_wall"] - 1.0, zl1 + 1.0)
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("height  (mm)")
    return m


def map_limit(an):
    v = np.abs(an["raw"] - an["raw"].mean())
    return max(1e-4, np.percentile(v, 99) * 1.4)


def results_figure(sol, an, cfg):
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.7),
                           gridspec_kw={"width_ratios": [0.9, 1.5]})
    m = field_map(ax[0], sol, an, cfg, map_limit(an))
    cb = fig.colorbar(m, ax=ax[0], fraction=0.08, pad=0.04)
    cb.outline.set_visible(False)
    cb.set_label("field change  (ppm)", color=P.MUTED, fontsize=9)
    ax[0].set_title("field in the sample\n(before shimming)")

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
    ax[1].set_title(f"the {C.NUC_LABEL[cfg['nuc']]} NMR line you would "
                    "measure")
    ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[1].grid(color=P.GRID, lw=0.8)
    ax[1].set_axisbelow(True)
    fig.tight_layout(w_pad=2.5)
    return fig


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
    lim = max(map_limit(R[e][1]) for e in "AB")
    for i, e in enumerate("AB"):
        sol, an, scfg, _ = R[e]
        ax = fig.add_subplot(gs[0, i])
        m = field_map(ax, sol, an, scfg, lim)
        ax.set_title(f"field in the sample, {e}", color=ABCOL[e],
                     fontweight="bold")
    cb = fig.colorbar(m, cax=fig.add_subplot(gs[0, 2]))
    cb.outline.set_visible(False)
    cb.set_label("field change before shimming (ppm)", color=P.MUTED,
                 fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.14, top=0.88)
    return fig


# ----------------------------------------------------------- downloads
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


def numbers_csv(R):
    """Every number on screen plus the full setup, one column per setup."""
    es = list(R)
    out = ["quantity," + ",".join(es)]

    def row(name, vals):
        out.append(name + "," + ",".join(str(v) for v in vals))

    row("FWHM (Hz)", [f"{R[e][1]['wd']['fwhm']:.4g}" for e in es])
    row("width at 0.55 % (Hz)", [f"{R[e][1]['wd']['w055']:.4g}" for e in es])
    row("width at 0.11 % (Hz)", [f"{R[e][1]['wd']['w011']:.4g}" for e in es])
    row("leftover field rms (ppb)",
        [f"{R[e][1]['rep']['shim_ppm_rms']*1000:.4g}" for e in es])
    row("leftover field peak to peak (ppb)",
        [f"{R[e][1]['rep']['shim_ppm_pp']*1000:.4g}" for e in es])
    row("spread before shimming peak to peak (ppm)",
        [f"{R[e][1]['rep']['raw_ppm_pp']:.4g}" for e in es])
    row("observed frequency (MHz)", [f"{R[e][1]['f']:.4f}" for e in es])
    row("voxel (mm)", [R[e][0]["V"] for e in es])
    row("voxels detected", [R[e][1]["n"] for e in es])
    for o in range(5):
        row(f"FWHM with shimming to order {o} (Hz)",
            [f"{R[e][1]['by_order'][o][2]:.4g}" for e in es])
        row(f"leftover rms with shimming to order {o} (ppb)",
            [f"{R[e][1]['by_order'][o][1]:.4g}" for e in es])
    out.append("")
    out.append("setup parameter," + ",".join(es))
    for k in C.BLANK:
        row(k, [str(R[e][2][k]).replace(",", ";") for e in es])
    for e in es:
        for i, p in enumerate(R[e][2].get("custom", [])):
            out.append(f"custom part {i+1} ({e}),"
                       + json.dumps(p).replace(",", ";"))
    return "\n".join(out)


def line_csv(R):
    out = ["setup,frequency_offset_Hz,signal"]
    for e in R:
        an = R[e][1]
        nu, sp = an["nu"], an["sp"]
        nu = nu - nu[int(np.argmax(sp))]
        keep = np.where(sp > 1e-4)[0]
        sl = slice(max(keep[0] - 5, 0), keep[-1] + 6)
        nu, sp = nu[sl], sp[sl]
        step = max(1, len(nu) // 3000)
        out += [f"{e},{a:.4f},{b:.6f}" for a, b in zip(nu[::step], sp[::step])]
    return "\n".join(out)


def download_row(R, png, pdf, tag):
    st.caption("Download")
    c = st.columns(4)
    c[0].download_button("figure (PNG)", png, "b0_result.png", "image/png",
                         key=f"dl_png_{tag}", width="stretch")
    c[1].download_button("figure (PDF)", pdf, "b0_result.pdf",
                         "application/pdf", key=f"dl_pdf_{tag}",
                         width="stretch")
    c[2].download_button("numbers (CSV)", numbers_csv(R), "b0_numbers.csv",
                         "text/csv", key=f"dl_num_{tag}", width="stretch",
                         help="Every number shown, the result for each "
                              "shim order, and the full setup.")
    c[3].download_button("NMR line (CSV)", line_csv(R), "b0_line.csv",
                         "text/csv", key=f"dl_line_{tag}", width="stretch",
                         help="The simulated spectrum: frequency offset "
                              "and signal.")


# ------------------------------------------------------------- results
GEOM_SKIP = ("d_auto", "d_r", "d_z0", "d_z1", "d_soft", "d_edge",
             "sh_order", "f_MHz", "nuc", "nat")
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


def single_results(R):
    sol, an, scfg, _ = R["A"]
    wd, rep = an["wd"], an["rep"]
    nat = scfg["nat"]
    k = st.columns(3)
    k[0].metric("linewidth (FWHM)", fmt(wd["fwhm"], "{:.1f} Hz"),
                help="Full width at half maximum of the simulated line.")
    k[1].metric("added by the field", fmt(wd["fwhm"] - nat, "{:+.1f} Hz"),
                help="How much wider the line is than without the probe.")
    k[2].metric("field spread after shimming",
                f"{rep['shim_ppm_rms']*1000:.1f} ppb",
                help="rms of the leftover field over the detected region. "
                     "1 ppb = 0.001 ppm.")
    png, pdf = show(results_figure(sol, an, scfg), pdf=True)
    download_row(R, png, pdf, "one")
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
        P.hist_panel(a2[0], an["res"], an["f"])
        a2[0].set_title("leftover field after shimming")
        log_line(a2[1], [(an, "#2f6fb5", None)], nat)
        f2.tight_layout(w_pad=2.5)
        try:
            show(f2)
        except Exception:
            plt.close(f2)
            st.caption("This plot could not be drawn on this server.")
        st.caption(f"{an['n']} voxels detected, voxel {sol['V']} mm, "
                   f"detection {an['z0']:.1f} to {an['z1']:.1f} mm"
                   + (" (smooth edges)" if an["soft"] else "")
                   + f", radius {an['r']:.1f} mm, shimming: "
                   f"{C.SHIM_LABELS[int(scfg['sh_order'])]}.")


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
    png, pdf = show(ab_lines(R), pdf=True)
    download_row(R, png, pdf, "ab")
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


def tab_result():
    st.subheader("Preview")
    if "cut_plane" not in ss:
        ss.cut_plane = "x"
    st.radio("cut plane", ["x", "y"], key="cut_plane", horizontal=True,
             format_func=lambda v: f"through the {v} axis",
             label_visibility="collapsed")
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
        if run:
            ss.tour_done = True
        try:
            with st.spinner("computing the field…"):
                for e in SETUPS:
                    sig = json.dumps(cfg_of(e), sort_keys=True)
                    solve_cached(sig)
                    ss.solved[e] = sig
        except Exception as ex:
            st.error(f"This setup could not be computed: {ex}")
            return

    try:
        R = {e: result_for(e) for e in SETUPS}
    except Exception as ex:
        st.error(f"The result could not be analysed: {ex}")
        return
    if any(r is None for r in R.values()):
        st.info("Press **Run** to compute the field.")
        return
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
        return
    if AB:
        ab_results(R)
    else:
        single_results(R)


# --------------------------------------------------------------- sweep
def sweep_params(c):
    """(key, label, lowest, highest, default half-range) that make sense
    for this setup."""
    out = [("t_tip", "tube tip height (mm)", -20.0, 40.0, 5.0),
           ("t_fill", "liquid column (mm)", 2.0, 80.0, 15.0),
           ("t_OD", "tube outer diameter (mm)", 1.0, 12.0, 1.0),
           ("t_wall", "tube wall thickness (mm)", 0.05, 2.0, 0.2)]
    if not c["t_liquid"].startswith("custom"):
        out.append(("t_temp", "sample temperature (°C)", -20.0, 100.0, 20.0))
    if c["w_on"]:
        out += [("w_top", "waveguide top height (mm)", -40.0, 20.0, 10.0),
                ("w_id", "waveguide inner diameter (mm)", 1.0, 20.0, 1.5),
                ("w_wall", "waveguide wall (mm)", 0.1, 5.0, 0.6)]
    if c["c_on"]:
        out += [("c_zc", "coil centre height (mm)", -20.0, 80.0, 5.0),
                ("c_r", "coil inner radius (mm)", 1.0, 15.0, 1.5),
                ("c_len", "coil length (mm)", 1.0, 60.0, 8.0)]
    if c["p_on"] and c["p_bot"]:
        out.append(("p_bot_len", "bottom plug length (mm)", 0.5, 40.0, 5.0))
    if c["p_on"] and c["p_top"]:
        out.append(("p_top_len", "top plunger length (mm)", 0.5, 40.0, 5.0))
    if not (c["d_auto"] and c["c_on"]):
        out += [("d_z0", "detected region bottom (mm)", -20.0, 80.0, 5.0),
                ("d_z1", "detected region top (mm)", -20.0, 80.0, 5.0)]
    out.append(("d_r", "detected radius (mm)", 0.1, 6.0, 0.5))
    return out


def sweep_figure(sw):
    rows = sw["rows"]
    v = [r["value"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for a, key, lab, col in ((ax[0], "fwhm", "linewidth, FWHM  (Hz)",
                              "#2f6fb5"),
                             (ax[1], "ppb", "field spread after shimming  "
                              "(ppb)", "#c0703a")):
        y = [r[key] for r in rows]
        a.plot(v, y, "-o", color=col, lw=2, ms=5)
        a.axvline(sw["current"], color=P.MUTED, lw=1, ls=(0, (4, 3)))
        a.set_xlabel(sw["label"])
        a.set_title(lab, fontsize=10.5)
        a.set_ylim(bottom=0)
        a.grid(color=P.GRID, lw=0.8)
        a.set_axisbelow(True)
    ax[0].text(sw["current"], ax[0].get_ylim()[1], " now", fontsize=8.5,
               color=P.MUTED, va="top")
    fig.tight_layout(w_pad=2.0)
    return fig


def tab_sweep():
    c = cfg_now()
    st.subheader("Sweep one parameter")
    st.caption("Vary one thing, keep everything else, and see how the "
               "linewidth follows. The usual questions: how far does the "
               "waveguide have to be, how thin should the wall be, where "
               "should the tube sit?"
               + (f" This sweeps **setup {E}**." if AB else ""))
    prm = sweep_params(c)
    labels = {k: lab for k, lab, *_ in prm}
    key = st.selectbox("parameter", [k for k, *_ in prm],
                       format_func=lambda k: labels[k], key=f"sw_key_{E}")
    _, label, lo, hi, half = next(p for p in prm if p[0] == key)
    cur = float(c[key])
    a, b, d = st.columns(3)
    v0 = a.number_input("from", lo, hi, float(np.clip(cur - half, lo, hi)),
                        key=f"sw_from_{E}_{key}")
    v1 = b.number_input("to", lo, hi, float(np.clip(cur + half, lo, hi)),
                        key=f"sw_to_{E}_{key}")
    nmax = C.max_sweep_steps()
    n = d.number_input("steps", 3, nmax, min(7, nmax), key=f"sw_n_{E}",
                       help=None if nmax >= 15 else
                       "Limited to 7 on this server: each step is a full "
                       "solve, and free servers throttle long runs. Run the "
                       "app on your own computer for longer sweeps.")
    st.caption(f"Now: **{cur:g}**. Each step is a full field calculation, "
               "so a big probe takes a few seconds per step.")
    if st.button("▶  Run the sweep", type="primary", width="stretch"):
        if v1 <= v0:
            st.error("'to' has to be larger than 'from'.")
        else:
            bar = st.progress(0.0, text="sweeping…")
            rows = C.sweep(c, key, np.linspace(v0, v1, int(n)),
                           progress=lambda f: bar.progress(f, "sweeping…"))
            bar.empty()
            ss.sweep = dict(setup=E, key=key, label=label, rows=rows,
                            current=cur, order=int(c["sh_order"]),
                            nuc=c["nuc"])
    sw = ss.get("sweep")
    if not sw:
        return
    st.markdown(f"**{sw['label']}**"
                + (f", setup {sw['setup']}" if AB else "")
                + f" · shimming: {C.SHIM_LABELS[sw['order']]} · "
                f"{C.NUC_LABEL[sw['nuc']]}")
    ok = [r for r in sw["rows"] if np.isfinite(r["ppb"])]
    if not ok:
        st.error("No step gave a result: the detected region had no sample "
                 "in it, or the geometry was impossible.")
        return
    best = min(ok, key=lambda r: r["ppb"])
    st.markdown(f"Most homogeneous at **{best['value']:g}**: "
                f"{best['ppb']:.1f} ppb left, FWHM "
                f"{fmt(best['fwhm'], '{:.1f}')} Hz.")
    png, pdf = show(sweep_figure(sw), pdf=True)
    st.dataframe(
        [{sw["label"]: round(r["value"], 4),
          "FWHM (Hz)": round(r["fwhm"], 2),
          "width at 0.55 % (Hz)": round(r["w055"], 1),
          "spread after shimming (ppb)": round(r["ppb"], 2),
          "spread before shimming (ppm)": round(r["raw"], 4)}
         for r in sw["rows"]], hide_index=True, width="stretch")
    csv = "\n".join(
        [f"{sw['label']},FWHM_Hz,width_0.55pct_Hz,leftover_rms_ppb,"
         "raw_pp_ppm"]
        + [f"{r['value']:.5g},{r['fwhm']:.4g},{r['w055']:.4g},"
           f"{r['ppb']:.4g},{r['raw']:.4g}" for r in sw["rows"]])
    c3 = st.columns(3)
    c3[0].download_button("figure (PNG)", png, "b0_sweep.png", "image/png",
                          width="stretch")
    c3[1].download_button("figure (PDF)", pdf, "b0_sweep.pdf",
                          "application/pdf", width="stretch")
    c3[2].download_button("table (CSV)", csv, "b0_sweep.csv", "text/csv",
                          width="stretch")
    if len(ok) < len(sw["rows"]):
        st.caption("Empty rows: no sample in the detected region at that "
                   "value, or an impossible geometry.")


# ------------------------------------------------------------- 3D view
CAD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "frascan_cad.npz")
CAD_TIP = 1.0        # height of the tube tip above the waveguide in the CAD
CAD_COL = {"copper parts": ("#c0703a", 1.0), "brass parts": ("#b59a3c", 1.0),
           "PEEK parts": ("#d8cfa6", 0.9), "lid (SLA resin)": ("#c9c9d4", 0.12),
           "glass former": ("#bcd3e6", 0.3),
           "chip capacitors": ("#6b5b95", 1.0)}


@st.cache_resource(show_spinner=False)
def load_cad():
    """The real probe-head meshes exported from the Fusion assembly, if the
    file is there. Heights are moved to the probe frame and everything
    more than 24 mm below the waveguide top is cut off."""
    if not os.path.exists(CAD_FILE):
        return None
    d = np.load(CAD_FILE)
    out = {}
    for k in d.files:
        if k.endswith("|V"):
            g = k[:-2]
            V = d[k].astype(float).copy()
            T = d[g + "|T"]
            V[:, 2] += CAD_TIP
            T = T[V[T][:, :, 2].max(axis=1) > -24.0]
            V[:, 2] = np.maximum(V[:, 2], -24.0)
            out[g] = (V, T)
    return out


def fig3d(cfg, cut=True, cad=None):
    """The probe as rotatable surfaces. Drawn from the same numbers the
    solver uses; a picture, not the voxel model. With `cad`, the probe
    parts are the real CAD meshes instead (tube and sample stay ours)."""
    phi = np.linspace(0, 1.5*np.pi if cut else 2*np.pi, 73)
    tr, seen = [], set()

    def surf(x, y, z, part, op):
        col = colour(part)
        tr.append(go.Surface(
            x=x, y=y, z=z, surfacecolor=np.zeros_like(z),
            colorscale=[[0, col], [1, col]], showscale=False, opacity=op,
            name=part, legendgroup=part, showlegend=part not in seen,
            hoverinfo="name",
            lighting=dict(ambient=0.75, diffuse=0.5, specular=0.05)))
        seen.add(part)

    def cyl(r, z0, z1, part, op, cx=0.0, cy=0.0):
        Pm, Zm = np.meshgrid(phi, [z0, z1])
        surf(cx + r*np.cos(Pm), cy + r*np.sin(Pm), Zm, part, op)

    def disc(r0, r1, zc, part, op, cx=0.0, cy=0.0):
        Pm, Rm = np.meshgrid(phi, [r0, r1])
        surf(cx + Rm*np.cos(Pm), cy + Rm*np.sin(Pm),
             np.full(Pm.shape, float(zc)), part, op)

    def ring(r0, r1, z0, z1, part, op, cx=0.0, cy=0.0):
        cyl(r1, z0, z1, part, op, cx, cy)
        if r0 > 0:
            cyl(r0, z0, z1, part, op, cx, cy)
        disc(r0, r1, z0, part, op, cx, cy)
        disc(r0, r1, z1, part, op, cx, cy)

    def cap(r, zc, depth, part, op):
        t = np.linspace(0, np.pi/2, 14)
        Pm, Tm = np.meshgrid(phi, t)
        surf(r*np.cos(Tm)*np.cos(Pm), r*np.cos(Tm)*np.sin(Pm),
             zc - depth*np.sin(Tm), part, op)

    # ---- tube, sample, plugs
    ro, ri = cfg["t_OD"]/2, cfg["t_OD"]/2 - cfg["t_wall"]
    tip = cfg["t_tip"]
    zl0, zl1 = C.liquid_span(cfg)
    pb = cfg["p_on"] and cfg["p_bot"]
    pt = cfg["p_on"] and cfg["p_top"]
    ztop = zl1 + (cfg["p_top_len"] + 4 if pt else 8)
    low = "plug" if pb else "sample"          # what sits on the tube floor
    if cfg["t_bottom"] == "round":
        zj = tip + cfg["t_botdepth"]
        cap(ro, zj, cfg["t_botdepth"], "glass", 0.22)
        cyl(ro, zj, ztop, "glass", 0.22)
        cap(ri, zj, max(cfg["t_botdepth"] - cfg["t_wall"], 0.05), low, 0.95)
        floor = zj
    else:
        disc(0, ro, tip, "glass", 0.22)
        cyl(ro, tip, ztop, "glass", 0.22)
        floor = tip + cfg["t_wall"]
        disc(0, ri, floor, low, 0.95)
    if pb:
        cyl(ri, floor, zl0, "plug", 0.95)
    cyl(ri, max(zl0, floor), zl1, "sample", 0.95)
    disc(0, ri, zl1, "sample", 0.95)
    if pt:
        cyl(ri, zl1, zl1 + cfg["p_top_len"], "plug", 0.95)
        disc(0, ri, zl1 + cfg["p_top_len"], "plug", 0.95)

    if cad:
        for g, (V, T) in cad.items():
            col, op = CAD_COL.get(g, ("#999999", 1.0))
            if cut:                       # same quarter removed as above
                c = V[T].mean(axis=1)
                T = T[~((c[:, 0] > 0) & (c[:, 1] < 0))]
            tr.append(go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2], i=T[:, 0], j=T[:, 1],
                k=T[:, 2], color=col, opacity=op, name=g, showlegend=True,
                hoverinfo="name", flatshading=False,
                lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1,
                              roughness=0.7)))

    # ---- RF coil: a sheet with its slots or windows cut out
    if cfg["c_on"] and not cad:
        pf = np.linspace(phi[0], phi[-1], 241)
        bot = cfg["c_zc"] - cfg["c_len"]/2
        zz = np.linspace(bot, bot + cfg["c_len"], 121)
        Pm, Zm = np.meshgrid(pf, zz)
        keep = np.ones(Pm.shape, bool)
        n = int(cfg["c_nslots"])
        if n > 0 and cfg["c_slotdeg"] > 0:
            deg = np.degrees(Pm) % 360.0
            for i in range(n):
                c0 = cfg["c_phase"] + 360.0*i/n
                d = np.abs((deg - c0 + 180.0) % 360.0 - 180.0)
                if not cfg["c_win"]:
                    keep &= ~(d <= cfg["c_slotdeg"]/2)
                else:
                    w0 = bot + cfg["c_winz0"]
                    sl = np.degrees(cfg["c_slit"] / cfg["c_r"])
                    keep &= ~(((d <= cfg["c_slotdeg"]/2) & (Zm >= w0)
                               & (Zm <= w0 + cfg["c_winh"]))
                              | ((d <= sl/2) & (Zm < w0)))
        r = cfg["c_r"]
        surf(r*np.cos(Pm), r*np.sin(Pm), np.where(keep, Zm, np.nan),
             "RF coil", 1.0)

    # ---- waveguide (its lower part is cut short for the picture)
    if cfg["w_on"] and not cad:
        wt = cfg["w_top"]
        ring(cfg["w_id"]/2, cfg["w_id"]/2 + cfg["w_wall"],
             wt - min(cfg["w_len"], 12.0), wt, "waveguide", 1.0)

    # ---- shields, modulation coil, custom parts
    for kind, p, label in ([] if cad else C._parts(cfg)):
        op = {"shield": 0.55, "modulation coil": 1.0}.get(label, 0.9)
        if p["r_out"] > 12 if kind != "box" else False:
            op = min(op, 0.55)          # big housings: keep the inside visible
        if kind == "annulus":
            ring(p["r_in"], p["r_out"], p["z0"], p["z1"], label, op)
        elif kind == "cyl":
            ring(p["r_in"], p["r_out"], p["z0"], p["z1"], label, op,
                 p["cx"], p["cy"])
        else:
            x0, x1 = p["cx"] - p["sx"]/2, p["cx"] + p["sx"]/2
            y0, y1 = p["cy"] - p["sy"]/2, p["cy"] + p["sy"]/2
            X = [x0, x1, x1, x0, x0, x1, x1, x0]
            Y = [y0, y0, y1, y1, y0, y0, y1, y1]
            Z = [p["z0"]]*4 + [p["z1"]]*4
            tr.append(go.Mesh3d(
                x=X, y=Y, z=Z, color=colour(label), opacity=op, name=label,
                i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
                j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
                k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
                legendgroup=label, showlegend=label not in seen,
                hoverinfo="name", flatshading=True))
            seen.add(label)

    # ---- detected region, as an outline
    z0, z1, rr = C.detection(cfg)
    t = np.linspace(0, 2*np.pi, 60)
    xs, ys, zs = [], [], []
    for zc in (z0, z1):
        xs += list(rr*np.cos(t)) + [None]
        ys += list(rr*np.sin(t)) + [None]
        zs += [zc]*len(t) + [None]
    for a in (0, np.pi/2, np.pi, 1.5*np.pi):
        xs += [rr*np.cos(a)]*2 + [None]
        ys += [rr*np.sin(a)]*2 + [None]
        zs += [z0, z1, None]
    tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                           line=dict(color="#1c2530", width=4, dash="dash"),
                           name="detected region", hoverinfo="name"))

    fig = go.Figure(tr)
    ax = dict(backgroundcolor="#f3f5f8", gridcolor="#dfe3e8",
              zerolinecolor="#c8ccd2")
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.6)"),
        scene=dict(aspectmode="data",
                   xaxis=dict(title="x (mm)", **ax),
                   yaxis=dict(title="y (mm)", **ax),
                   zaxis=dict(title="height (mm)", **ax),
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5))))
    return fig


def tab_3d():
    st.subheader("3D view" + (f" of setup {E}" if AB else ""))
    if go is None:
        st.info("The 3D view needs the plotly package. Install it with "
                "`pip install plotly` and restart the app.")
        return
    cut = st.toggle("cut-away (remove a quarter to see inside)", value=True,
                    key="cut3d")
    cad = None
    if ss.preset[E] == "FRASCAN II" and load_cad():
        if st.toggle("show the real CAD parts", value=True, key="cad3d",
                     help="On: the probe head as drawn in Fusion (every "
                          "screw and capacitor). Off: the simplified probe "
                          "the calculation actually uses."):
            cad = load_cad()
    st.caption("Drag to rotate, scroll to zoom, click a name in the legend "
               "to hide that part."
               + (" The tube and sample follow your settings; the rest is "
                  "the CAD. The calculation uses the simplified parts "
                  "(switch the CAD off to see them)." if cad else ""))
    try:
        st.plotly_chart(fig3d(cfg_now(), cut, cad), width="stretch")
    except Exception as ex:
        st.error(f"This geometry cannot be drawn in 3D: {ex}")


# --------------------------------------------------------------- about
ABOUT = """
### What this tool does

Every material in a magnet, glass, liquid, copper, even air, is very
slightly magnetic. Each part therefore bends the magnetic field around it a
little. In NMR and DNP that matters: if the field is not the same everywhere
in the sample, the line gets broad or splits.

This tool computes that disturbance for a probe you describe (tube, sample,
coil, waveguide, shields, anything else), removes what the shim coils could
correct, and shows the NMR line you would be left with.

### How it works

1. Space is cut into small cubes (voxels, 0.25 mm by default). Each cube
   gets the magnetic susceptibility χ of the material in it. Thin parts
   such as coil foil are handled so that χ × thickness is right.
2. The field change follows from one formula, evaluated with a fast Fourier
   transform:  **ΔB/B₀ = FT⁻¹[ FT(χ) · (1/3 − k_z²/k²) ]**.
   This is the standard method from MRI (Salomir 2003, Marques & Bowtell
   2005). It is exact to first order in χ, and χ is about 10⁻⁵, so the
   neglected part is about 10⁻¹⁰.
3. Over the part of the sample the coil sees, the tool fits and removes
   the field shapes a shim set can produce (up to the order you choose).
4. What is left is turned into a spectrum: each voxel resonates at its own
   frequency, and the line is their sum, broadened by the linewidth you
   would have without the probe.

### What it can and cannot tell you

**It can:** compare designs (which tube, which wall, how far the metal
should be, which material to use), show which part causes the damage, and
give the linewidth a design adds.

**It cannot:**
- know your magnet. Its own imperfections, and how well it was shimmed, are
  not modelled. Put the linewidth you measure with a good commercial probe
  in *linewidth without the probe*; the tool adds this probe on top.
- know how your shims were really set. It assumes the best possible setting
  up to the chosen order. *More results* shows every order, which is the
  honest error bar.
- model magic-angle spinning. Spinning averages most of these effects away,
  so a static calculation would be misleading for a MAS rotor.
- model ferromagnetic or strongly magnetic parts, radiation damping, or
  anything about the RF and microwave fields themselves.

### How far to trust it

- **The solver** is checked against exact solutions (a sphere, cylinders
  along and across the field). Press *Check the solver* in the sidebar to
  run the sphere at your voxel size: the error is about 0.5 %.
- **The materials** decide the accuracy in practice. Values marked
  *measured* come from Wapler et al. 2014 and Hoffman 2022 and are good to
  about 0.01 ppm. Values marked *estimate* (brass, silver, sapphire...) are
  handbook numbers: measure your own material if the result depends on it.
- **Comparison with measured spectra** is in progress. You can overlay your
  own spectrum under *Compare with a measured spectrum*.

### Sources of the material values

- F. Wapler et al., *Magnetic properties of materials for MR engineering,
  micro-MR and beyond*, J. Magn. Reson. 242 (2014) 233.
- R. Hoffman, *Magnetic susceptibility measurement by NMR: 2. The magnetic
  susceptibility of NMR solvents and their chemical shifts*, J. Magn.
  Reson. 335 (2022) 107105 (also the temperature dependence).
- R. Salomir, B. de Senneville, C. Moonen, Concepts Magn. Reson. B 19
  (2003) 26; J. Marques, R. Bowtell, Concepts Magn. Reson. B 25 (2005) 65.
"""


def tab_about():
    st.markdown(ABOUT)
    if CITE:
        st.markdown("### How to cite\n\n" + CITE)
    if AUTHOR or FEEDBACK_URL:
        st.markdown("### Contact")
        if AUTHOR:
            st.markdown(AUTHOR)
        if FEEDBACK_URL:
            st.markdown(f"Found a problem or missing a feature? "
                        f"[Tell us here]({FEEDBACK_URL}).")


with right:
    t_res, t_sweep, t_3d, t_about = st.tabs(
        ["Result", "Sweep", "3D view", "About"])
    with t_res:
        tab_result()
    with t_sweep:
        tab_sweep()
    with t_3d:
        tab_3d()
    with t_about:
        tab_about()
