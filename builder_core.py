"""
Engine behind builder.py, the simple "build your probe" app.

It reuses the validated geometry builders and analysis in b0core.py. Three
things are new here:

1. A config is a plain dict (and can be saved as JSON). Every height is in
   ONE frame: z = 0 is the probe reference, which is the waveguide top when
   a waveguide is present. The tube tip sits at cfg["t_tip"]. This is the
   same frame b0core uses internally, so nothing is converted.
2. The background (the 'surroundings' material) is subtracted before the
   FFT. A uniform medium only shifts the field uniformly, which the lock
   removes, so this changes nothing physical, but it removes the fake
   surface at the edge of the computation box.
3. A faster, lower-memory FFT (real-to-complex, multithreaded when scipy is
   installed). Same maths as b0core.field_shift_ppm.
"""
import base64
import json
import threading
import zlib
from math import erf

import numpy as np

from b0core import (MATERIALS_PPM, SOURCES, build_tube, build_coil,
                    build_cwg, build_annulus, width_at)

# build_all writes per-setup entries into the shared materials table, so
# two people on the same server must not build at the same moment.
_LOCK = threading.Lock()

# ------------------------------------------------- extra materials
# All from Wapler et al. 2014, Table 1: measured relative to water, here
# on the same water reference (-9.032) as the entries already in b0core.
_W14 = "Wapler et al., J Magn Reson 242 (2014) 233, arXiv:1403.4760, Table 1"
for _n, _d in (("PMMA (acrylic)", -0.023), ("polyimide (Kapton)", 0.117),
               ("polystyrene", -0.141), ("polycarbonate", -0.210),
               ("nylon (PA6)", -0.446), ("polypropylene", -0.523),
               ("polyethylene (HDPE)", -0.635), ("PVC", -1.673),
               ("FR4 circuit board", 5.29)):
    MATERIALS_PPM.setdefault(_n, round(-9.032 + _d, 2))
    SOURCES.setdefault(_n, ("measured", _W14 + f": {_d:+.3f} relative to "
                            "water at -9.032"))

MATERIALS_PPM.setdefault("SLA resin", -9.0)
SOURCES.setdefault("SLA resin", ("estimate", "acrylate photopolymer, taken "
                   "close to PMMA; not measured"))
MATERIALS_PPM.setdefault("ceramic (chip capacitor)", -18.0)
SOURCES.setdefault("ceramic (chip capacitor)", (
    "estimate", "GUESS, taken as alumina. What matters is the dielectric and "
    "the metal terminations, not the capacitance. Between -30 and 0 ppm the "
    "predicted FRASCAN II line moves from 61 to 19 Hz, so ask the maker or "
    "measure with and without them."))

MATCHED = "matched to the sample"

# ------------------------------------------------------ temperature
# Volume susceptibility against temperature, measured 0 to 70 °C by
# Hoffman, J Magn Reson 335 (2022) 107105, Table 4. Slope in ppm per °C
# (positive: less diamagnetic when warmer, because the liquid expands).
# The table values in b0core are his 25 °C values.
T_REF = 25.0
TEMP_SLOPE = {
    "acetone-d6": 8.0e-3, "acetonitrile-d3": 9.4e-3, "benzene-d6": 9.6e-3,
    "chloroform-d": 12.1e-3, "DMSO": 7.4e-3, "methanol-d4": 7.8e-3,
    "THF-d8": 10.2e-3, "toluene-d8": 8.6e-3,
    # protiated: the deuterated slope (same expansion)
    "toluene": 8.6e-3, "methanol": 7.8e-3,
    # estimates from the thermal expansion coefficient
    "ethanol": 7.8e-3, "glycerol/water 60:40": 4.0e-3,
}
TEMP_ESTIMATED = ("ethanol", "glycerol/water 60:40")
# density of water, g/cm3 (its mass susceptibility barely changes)
_WATER_T = np.array([0, 4, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100.])
_WATER_RHO = np.array([0.99984, 0.99997, 0.99970, 0.99910, 0.99821, 0.99705,
                       0.99565, 0.99222, 0.98804, 0.98320, 0.97778, 0.97182,
                       0.96535, 0.95840])


def chi_sample(cfg):
    """Sample susceptibility (ppm) at the chosen temperature."""
    name = cfg["t_liquid"]
    if name == "custom (sample)":
        return float(cfg["t_liquid_chi"])
    chi = MATERIALS_PPM[name]
    T = float(cfg.get("t_temp", T_REF))
    if T == T_REF:
        return chi
    if name == "water":
        rho = np.interp([T, T_REF], _WATER_T, _WATER_RHO)
        return float(chi * rho[0] / rho[1])
    if name == "D2O":                      # Hoffman: -3.2e-3 T + 69e-6 T²
        f = lambda t: -3.2e-3 * t + 69e-6 * t * t
        return float(chi + f(T) - f(T_REF))
    return float(chi + TEMP_SLOPE.get(name, 0.0) * (T - T_REF))


def chi_tube(cfg):
    if cfg["t_glass"] == "custom (tube)":
        return float(cfg["t_glass_chi"])
    return MATERIALS_PPM[cfg["t_glass"]]


def chi_plug(cfg, which):
    m = cfg[f"p_{which}_mat"]
    return chi_sample(cfg) if m == MATCHED else MATERIALS_PPM[m]


# ----------------------------------------------------------- nuclei
# resonance frequency relative to 1H
NUCLEI = {"1H": 1.0, "19F": 0.940940, "31P": 0.404808, "13C": 0.251450,
          "2H": 0.153506, "15N": 0.101368}
NUC_LABEL = {"1H": "¹H", "19F": "¹⁹F", "31P": "³¹P", "13C": "¹³C",
             "2H": "²H", "15N": "¹⁵N"}


def f_obs(cfg):
    """Observed frequency in MHz: the magnet's 1H frequency times the
    nucleus ratio."""
    return float(cfg["f_MHz"]) * NUCLEI[cfg.get("nuc", "1H")]

try:
    from scipy import fft as _fft
    _KW = {"workers": -1}
except Exception:                                    # numpy fallback
    _fft = np.fft
    _KW = {}

VOXELS = [0.15, 0.2, 0.25, 0.3, 0.4]
CELL_LIMIT = 72e6          # padded FFT cells; about 1.2 GB peak at float64


# --------------------------------------------------------------- presets

BLANK = dict(
    # sample tube
    t_OD=5.0, t_wall=0.5, t_fill=40.0, t_glass="borosilicate",
    t_liquid="water", t_out="air", t_tip=0.0, t_bottom="round",
    t_botdepth=2.0, t_punt=2.0, t_men=0.0,
    t_glass_chi=-11.0, t_liquid_chi=-9.0, t_temp=25.0,
    # susceptibility plugs (Shigemi style)
    p_on=False, p_bot=True, p_bot_len=8.0, p_bot_mat=MATCHED,
    p_top=True, p_top_len=10.0, p_top_mat=MATCHED,
    # RF coil
    c_on=False, c_r=3.5, c_t=50.0, c_len=18.0, c_zc=20.0, c_mat="copper",
    c_nslots=2, c_slotdeg=90.0, c_phase=90.0, c_win=False, c_winz0=3.0,
    c_winh=12.0, c_slit=1.0,
    # waveguide
    w_on=False, w_id=5.0, w_wall=1.25, w_len=20.0, w_mat="brass", w_top=0.0,
    # RF shields
    rs_on=False, rs_bot=True, rs_side=True, rs_mat="copper",
    rs_b_rin=8.0, rs_b_rout=17.925, rs_b_z0=-0.74, rs_b_z1=-0.55,
    rs_s_rin=9.0, rs_s_rout=9.1, rs_s_z0=4.95, rs_s_z1=25.05,
    # modulation coil
    m_on=False, m_rin=9.1, m_rout=12.7, m_z0=5.0, m_z1=24.55, m_fill=0.75,
    m_mat="copper",
    # detection and shimming
    d_auto=True, d_r=1.5, d_z0=10.0, d_z1=25.0, d_soft=False, d_edge=2.0,
    sh_order=2, f_MHz=500.0, nuc="1H", nat=0.5,
    # numerics
    V=0.25,
)

STANDARD = dict(BLANK, t_wall=0.4, t_liquid="chloroform-d",
                c_on=True, c_r=3.5, c_t=50.0, c_len=18.0, c_zc=20.0,
                d_r=1.8)

# FRASCAN II, read from the Fusion assembly NMR_Probe_v617 (STEP export,
# 9 Nov 2026). z = 0 is the waveguide top; the CAD has the tube tip 1.0 mm
# above it. Materials from J. Dubsky's thesis, section 2.3.
FRASCAN = dict(BLANK,
    t_OD=4.98, t_wall=0.39, t_fill=34.0, t_glass="fused quartz",
    t_liquid="toluene", t_tip=1.0, t_bottom="round", t_botdepth=2.45,
    c_on=True, c_r=5.0, c_t=100.0, c_len=39.0, c_zc=15.0, c_nslots=2,
    c_slotdeg=90.0, c_phase=90.0, c_win=True, c_winz0=12.0, c_winh=15.0,
    c_slit=1.2,
    w_on=True, w_id=5.0, w_wall=1.25, w_len=20.0, w_mat="brass",
    rs_on=True, rs_bot=True, rs_side=True,
    rs_b_rin=8.0, rs_b_rout=17.93, rs_b_z0=-0.75, rs_b_z1=-0.55,
    rs_s_rin=9.0, rs_s_rout=9.1, rs_s_z0=4.95, rs_s_z1=25.05,
    m_on=True, m_rin=9.1, m_rout=12.6, m_z0=5.0, m_z1=24.55, m_fill=0.75,
    d_auto=True, d_r=1.5, d_edge=2.5,
)


def _ring(name, mat, r0, r1, z0, z1):
    return dict(name=name, on=True, kind="cylinder / ring", mat=mat,
                r_in=r0, r_out=r1, cx=0.0, cy=0.0, sx=4.0, sy=4.0,
                z0=z0, z1=z1, fill=1.0)


def _box(name, mat, cx, cy, sx, sy, z0, z1):
    return dict(name=name, on=True, kind="box", mat=mat, r_in=0.0,
                r_out=1.0, cx=cx, cy=cy, sx=sx, sy=sy, z0=z0, z1=z1,
                fill=1.0)


_SP, _LID = "modulation coil spool (PEEK)", "lid (SLA resin)"
_CAP = "chip capacitors (χ unknown)"
# Parts of the probe head that are not one of the standard parts above.
# Bodies of revolution are cut into rings; heights in the probe frame.
FRASCAN_PARTS = [
    _ring("RF coil former (glass)", "borosilicate", 4.55, 5.0, -18.25, 46.75),
    _ring("RF coil ground", "copper", 5.0, 5.1, -18.25, -6.25),
    _ring(_SP, "PEEK", 8.0, 17.93, -0.55, 2.45),
    _ring(_SP, "PEEK", 8.0, 12.6, 2.45, 4.95),
    _ring(_SP, "PEEK", 8.0, 9.0, 4.95, 25.05),
    _ring(_SP, "PEEK", 8.0, 12.6, 25.05, 34.24),
    _ring(_SP, "PEEK", 6.5, 12.6, 34.24, 37.24),
    _ring(_SP, "PEEK", 5.0, 12.6, 37.24, 37.95),
    _ring(_SP, "PEEK", 5.0, 9.0, 37.95, 46.75),
    # The lid sits 15 mm above the detected sample and changes the line by
    # 0.3 Hz out of 19, so it is off by default. Switch it on to check.
    dict(_ring(_LID, "SLA resin", 9.35, 17.93, 37.95, 55.25), on=False),
    dict(_ring(_LID, "SLA resin", 4.5, 9.35, 46.8, 51.2), on=False),
    dict(_ring(_LID, "SLA resin", 2.7, 9.35, 51.2, 55.5), on=False),
    _ring("electronics support (PEEK)", "PEEK", 10.05, 17.93, -3.25, -0.75),
    dict(_box(_CAP, "ceramic (chip capacitor)", 0.0, 6.09, 3.0, 2.02, 4.98, 7.5), on=False),
    dict(_box(_CAP, "ceramic (chip capacitor)", 0.0, -6.09, 3.0, 2.02, 4.98, 7.5), on=False),
]

TUBE3 = dict(BLANK, t_OD=3.0, t_wall=0.29, t_liquid="chloroform-d",
             c_on=True, c_r=2.2, c_t=50.0, c_len=16.0, c_zc=20.0,
             d_r=1.0, d_edge=1.2)

# 5 mm Shigemi: flat bottom, 8 mm matched glass below the sample and a
# matched plunger above, so the coil only ever sees matched material.
SHIGEMI = dict(BLANK, t_wall=0.4, t_bottom="flat", t_liquid="D2O",
               t_fill=18.0, p_on=True, p_bot=True, p_bot_len=8.0,
               p_top=True, p_top_len=15.0,
               c_on=True, c_r=3.5, c_t=50.0, c_len=18.0, c_zc=17.4,
               d_r=1.8)

QUARTZ4 = dict(BLANK, t_OD=4.0, t_wall=0.5, t_glass="fused quartz",
               t_liquid="toluene", t_fill=30.0, d_auto=False, d_r=1.3,
               d_z0=8.0, d_z1=23.0)

PRESETS = {
    "Blank (tube only)": (BLANK, "A tube of water in a perfect magnet. "
                          "Add the parts your probe has."),
    "Standard 5 mm tube": (STANDARD, "5 mm tube, CDCl₃, a plain copper "
                           "coil. Real commercial coils use "
                           "susceptibility-matched wire, so this is a "
                           "pessimistic version."),
    "3 mm tube": (TUBE3, "3 mm tube (wall 0.29 mm), CDCl₃, a small copper "
                  "coil."),
    "Shigemi tube (5 mm)": (SHIGEMI, "Flat bottom, only 18 mm of D₂O "
                            "between a matched glass bottom and a matched "
                            "plunger. Switch the plugs off to see what "
                            "they are for."),
    "4 mm quartz tube (EPR / DNP)": (QUARTZ4, "4 mm fused quartz tube with "
                                     "toluene, no coil: set the detected "
                                     "region by hand."),
    "FRASCAN II": (FRASCAN, "Our DNP probe, read from the Fusion CAD "
                   "(v617): slotted coil with windows on a glass former, "
                   "brass waveguide, RF shields, modulation coil on its "
                   "PEEK spool, lid, and the two chip capacitors at the "
                   "coil."),
}
# extra parts that come with a preset
PRESET_PARTS = {"FRASCAN II": FRASCAN_PARTS}
# colours for named extra parts (anything else gets the 'custom' colour)
PART_COLORS = {"RF coil former (glass)": "#bcd3e6",
               "RF coil ground": "#c98a5a",
               _SP: "#d8cfa6", _LID: "#c9c9d4",
               "electronics support (PEEK)": "#b9ad82",
               _CAP: "#6b5b95"}

CUSTOM_DEFAULT = dict(name="", on=True,
                      kind="cylinder / ring", mat="PTFE", r_in=0.0,  # a small PTFE lens below the tube
                      r_out=2.0, cx=0.0, cy=0.0, sx=4.0, sy=4.0,
                      z0=-6.0, z1=-1.0, fill=1.0)


# ---------------------------------------------------------- materials

def _register(cfg):
    """Per-setup materials, under names b0core can look up."""
    MATERIALS_PPM["__sample"] = chi_sample(cfg)
    MATERIALS_PPM["__tube"] = chi_tube(cfg)
    MATERIALS_PPM["__plug_bot"] = chi_plug(cfg, "bot")
    MATERIALS_PPM["__plug_top"] = chi_plug(cfg, "top")


def liquid_span(cfg):
    """(z_bottom, z_top) of the liquid column on the axis."""
    z0 = cfg["t_tip"] + cfg["t_wall"]
    if cfg["p_on"] and cfg["p_bot"]:
        z0 += cfg["p_bot_len"]
    return z0, z0 + cfg["t_fill"]


# ----------------------------------------------------------- geometry

def coil_window(cfg):
    """Detection region the coil defines: its window if it has one,
    otherwise its full length. (z0, z1) in the probe frame."""
    bot = cfg["c_zc"] - cfg["c_len"] / 2
    if cfg["c_win"]:
        return bot + cfg["c_winz0"], bot + cfg["c_winz0"] + cfg["c_winh"]
    return bot, bot + cfg["c_len"]


def detection(cfg):
    if cfg["d_auto"] and cfg["c_on"]:
        z0, z1 = coil_window(cfg)
    else:
        z0, z1 = cfg["d_z0"], cfg["d_z1"]
    return float(z0), float(z1), float(cfg["d_r"])


def _parts(cfg):
    """Every enabled metal or plastic part as (kind, kwargs, label)."""
    out = []
    if cfg["rs_on"] and cfg["rs_bot"]:
        out.append(("annulus", dict(r_in=cfg["rs_b_rin"], r_out=cfg["rs_b_rout"],
                    z0=cfg["rs_b_z0"], z1=cfg["rs_b_z1"],
                    material=cfg["rs_mat"]), "shield"))
    if cfg["rs_on"] and cfg["rs_side"]:
        out.append(("annulus", dict(r_in=cfg["rs_s_rin"], r_out=cfg["rs_s_rout"],
                    z0=cfg["rs_s_z0"], z1=cfg["rs_s_z1"],
                    material=cfg["rs_mat"]), "shield"))
    if cfg["m_on"]:
        out.append(("annulus", dict(r_in=cfg["m_rin"], r_out=cfg["m_rout"],
                    z0=cfg["m_z0"], z1=cfg["m_z1"], fill=cfg["m_fill"],
                    material=cfg["m_mat"]), "modulation coil"))
    for p in cfg.get("custom", []):
        if not p.get("on", True):
            continue
        label = p.get("name") or "custom"
        if p["kind"] == "box":
            out.append(("box", dict(cx=p["cx"], cy=p["cy"], sx=p["sx"],
                        sy=p["sy"], z0=p["z0"], z1=p["z1"], fill=p["fill"],
                        material=p["mat"]), label))
        else:
            out.append(("cyl", dict(cx=p["cx"], cy=p["cy"], r_in=p["r_in"],
                        r_out=p["r_out"], z0=p["z0"], z1=p["z1"],
                        fill=p["fill"], material=p["mat"]), label))
    return out


def _overlap(c, h, a, b):
    """Fraction of each cell [c-h/2, c+h/2] that lies inside [a, b]."""
    return np.clip((np.minimum(c + h/2, b) - np.maximum(c - h/2, a)) / h, 0, 1)


def build_box(x, y, z, *, cx, cy, sx, sy, z0, z1, material, outside,
              fill=1.0):
    """Axis-aligned block, exact partial volume. Returns (chi_add, mask)."""
    h = x[1] - x[0]
    dchi = (MATERIALS_PPM[material] - MATERIALS_PPM[outside]) * fill
    f = (_overlap(x, h, cx - sx/2, cx + sx/2)[:, None, None]
         * _overlap(y, h, cy - sy/2, cy + sy/2)[None, :, None]
         * _overlap(z, h, z0, z1)[None, None, :])
    return f * dchi, f > 0.25


def build_cyl(x, y, z, *, cx, cy, r_in, r_out, z0, z1, material, outside,
              fill=1.0, subsample=5):
    """Cylinder or ring, possibly off axis, still parallel to B0."""
    h = x[1] - x[0]
    dchi = (MATERIALS_PPM[material] - MATERIALS_PPM[outside]) * fill
    s = subsample
    offs = ((np.arange(s) + 0.5) / s - 0.5) * h
    fr = np.zeros((x.size, y.size))
    for ox in offs:
        for oy in offs:
            r = np.hypot(x[:, None] + ox - cx, y[None, :] + oy - cy)
            fr += (r >= r_in) & (r <= r_out)
    fr /= s * s
    f = fr[:, :, None] * _overlap(z, h, z0, z1)[None, None, :]
    return f * dchi, f > 0.25


def extent(cfg):
    """(radial half-width, z_lo, z_hi) that holds every enabled part."""
    r = cfg["t_OD"] / 2
    zlo = cfg["t_tip"] - 1.0
    zhi = liquid_span(cfg)[1] + 20.0
    if cfg["p_on"] and cfg["p_top"]:
        zhi = max(zhi, liquid_span(cfg)[1] + cfg["p_top_len"] + 5.0)
    if cfg["c_on"]:
        r = max(r, cfg["c_r"] + cfg["c_t"] * 1e-3)
        zlo = min(zlo, cfg["c_zc"] - cfg["c_len"]/2)
        zhi = max(zhi, cfg["c_zc"] + cfg["c_len"]/2)
    if cfg["w_on"]:
        r = max(r, cfg["w_id"]/2 + cfg["w_wall"])
        zlo = min(zlo, cfg["w_top"] - cfg["w_len"])
    for kind, p, _ in _parts(cfg):
        if kind == "box":
            r = max(r, abs(p["cx"]) + p["sx"]/2, abs(p["cy"]) + p["sy"]/2)
        elif kind == "annulus":
            r = max(r, p["r_out"])
        else:
            r = max(r, np.hypot(p["cx"], p["cy"]) + p["r_out"])
        zlo, zhi = min(zlo, p["z0"]), max(zhi, p["z1"])
    return r, zlo, zhi


def choose_grid(cfg):
    """Grid that holds everything, at the requested voxel if it fits in
    memory, otherwise the next coarser one. Returns (x, y, z, V, note)."""
    r, zlo, zhi = extent(cfg)
    half = min(r + 1.5, 22.0)
    half = max(half, cfg["t_OD"]/2 + 4.0)
    zlo, zhi = zlo - 1.5, zhi + 1.5
    V = float(cfg["V"])
    note = ""
    for v in [V] + [w for w in VOXELS if w > V]:
        cells = (2 * 2*half/v) ** 2 * (2 * (zhi - zlo)/v)
        if cells <= CELL_LIMIT:
            if v != V:
                note = (f"Voxel raised from {V} to {v} mm to fit in memory "
                        f"(the probe is big). Results are slightly coarser.")
            V = v
            break
    else:
        V = VOXELS[-1]
        note = "Probe is very big: running at the coarsest voxel (0.4 mm)."
    dx = V
    x = np.arange(-half + dx/2, half, dx)
    z = np.arange(zlo + dx/2, zhi, dx)
    return x, x.copy(), z, V, note


def build_all(cfg, x, y, z):
    """Full susceptibility map (ppm, background removed) and the masks the
    plots need. Works on any grid, including a single y = 0 slice."""
    with _LOCK:
        return _build_all(cfg, x, y, z)


def _build_all(cfg, x, y, z):
    _register(cfg)
    out = cfg["t_out"]
    h = x[1] - x[0]
    plug_b = cfg["p_on"] and cfg["p_bot"]
    plug_t = cfg["p_on"] and cfg["p_top"]
    zl0, zl1 = liquid_span(cfg)
    kw = dict(tube_OD=cfg["t_OD"], wall=cfg["t_wall"], d_cwg=cfg["t_tip"],
              fill_h=zl1 - (cfg["t_tip"] + cfg["t_wall"]),
              bottom=cfg["t_bottom"],
              punt_r=cfg["t_punt"], bot_depth=cfg["t_botdepth"],
              glass="__tube", liquid="__sample", outside=out,
              men_depth=0.0 if plug_t else cfg["t_men"])
    chi, liq, gls = build_tube(x, y, z, **kw)
    masks = {"glass": gls}
    liq0 = liq                      # the bore filled up to the liquid top
    if plug_b:
        # the bottom of the bore, up to zl0, is solid plug instead of liquid
        ov = _overlap(z, h, -1e9, zl0)[None, None, :]
        chi = chi + (MATERIALS_PPM["__plug_bot"]
                     - MATERIALS_PPM["__sample"]) * liq * ov
        masks["plug"] = liq & (ov > 0.5)
        liq = liq & (ov <= 0.0)
    if plug_t:
        # fills the bore above the liquid. Same voxel rule as the bore in
        # build_tube (voxel centre inside the inner radius), so the plunger
        # and the glass wall neither overlap nor leave a gap.
        ri = cfg["t_OD"]/2 - cfg["t_wall"]
        bore = (np.hypot(x[:, None], y[None, :]) <= ri)[:, :, None]
        f = (bore * _overlap(z, h, zl1, zl1 + cfg["p_top_len"])[None, None, :]
             * ~liq0)             # never on top of a liquid voxel
        chi = chi + (MATERIALS_PPM["__plug_top"] - MATERIALS_PPM[out]) * f
        pm = f > 0.5
        masks["plug"] = masks.get("plug", np.zeros_like(pm)) | pm
    masks["sample"] = liq
    if cfg["c_on"]:
        win = (dict(win_z0=cfg["c_winz0"], win_h=cfg["c_winh"],
                    slit_mm=cfg["c_slit"]) if cfg["c_win"] else None)
        cchi, cm = build_coil(x, y, z, coil_r=cfg["c_r"],
                              thickness_um=cfg["c_t"], length=cfg["c_len"],
                              z_centre=cfg["c_zc"], n_slots=int(cfg["c_nslots"]),
                              slot_deg=cfg["c_slotdeg"],
                              slot_phase=cfg["c_phase"],
                              material=cfg["c_mat"], window=win)
        chi = np.where(cm, cchi, chi)
        masks["RF coil"] = cm
    if cfg["w_on"]:
        wchi, wm = build_cwg(x, y, z, inner_d=cfg["w_id"], wall=cfg["w_wall"],
                             length=cfg["w_len"], z_top=cfg["w_top"],
                             material=cfg["w_mat"], outside=out)
        chi = chi + wchi
        masks["waveguide"] = wm
    for kind, p, label in _parts(cfg):
        if kind == "annulus":
            pc, pm = build_annulus(x, y, z, outside=out, **p)
        elif kind == "box":
            pc, pm = build_box(x, y, z, outside=out, **p)
        else:
            pc, pm = build_cyl(x, y, z, outside=out, **p)
        chi = chi + pc
        masks[label] = masks.get(label, np.zeros_like(pm)) | pm
    chi = chi - MATERIALS_PPM[out]          # background removed
    return chi, masks


# ---------------------------------------------------------------- field

def field_fast(chi, V, pad=2):
    """dBz/B0 in ppm. Same maths as b0core.field_shift_ppm, with a real FFT
    in single precision and the kernel applied slab by slab, so the padded
    k-space never needs a second full-size array. Single precision halves
    the memory and the time; against double it differs by about 3e-6 ppm
    on the full FRASCAN II probe, a thousand times below the shimmed
    residual."""
    f4 = np.float32
    n = chi.shape
    N = tuple(int(s * pad) for s in n)
    F = _fft.rfftn(chi.astype(f4, copy=False), s=N, **_KW)
    ky = np.fft.fftfreq(N[1], d=V)[:, None].astype(f4)
    kz = np.fft.rfftfreq(N[2], d=V)[None, :].astype(f4)
    kx = np.fft.fftfreq(N[0], d=V)
    kyz2 = ky**2 + kz**2
    kz2 = kz**2
    third = f4(1.0 / 3.0)
    for i in range(N[0]):
        k2 = f4(kx[i]**2) + kyz2
        with np.errstate(divide="ignore", invalid="ignore"):
            D = third - kz2 / k2
        if i == 0:
            D[0, 0] = 0.0
        F[i] *= D
    out = _fft.irfftn(F, s=N, **_KW)
    return np.ascontiguousarray(out[:n[0], :n[1], :n[2]], dtype=f4)


def solve(cfg):
    """The expensive step: geometry + FFT. Returns a dict the app caches."""
    x, y, z, V, note = choose_grid(cfg)
    chi, masks = build_all(cfg, x, y, z)
    dB = field_fast(chi, V)
    return dict(x=x, y=y, z=z, V=V, note=note, dB=dB, liq=masks["sample"],
                j=len(y)//2)


# ------------------------------------------------------------- shimming

SHIM_LABELS = {0: "none", 1: "linear (x, y, z)",
               2: "2nd order (z², xy, …)", 3: "3rd order (z³, …)",
               4: "4th order (z⁴, …)"}


def shim_basis(X, Y, Z, mask, order):
    """What a shim set up to `order` can produce: every real solid harmonic
    r^l P_l^m(cos θ) {cos, sin}(m φ) with l <= order, sampled on the mask.
    2l+1 functions per order (25 up to 4th). For order <= 2 this spans
    exactly the same space as b0core.shim_matrix. Centred on the detection
    region and scaled, only for numerical conditioning: the space itself
    does not change under a shift."""
    from numpy.polynomial import Legendre
    xa, ya, za = (np.asarray(a[mask], float) for a in (X, Y, Z))
    xa = xa - xa.mean(); ya = ya - ya.mean(); za = za - za.mean()
    L = max(np.sqrt(xa**2 + ya**2 + za**2).max(), 1e-9)
    xa, ya, za = xa / L, ya / L, za / L
    r = np.sqrt(xa**2 + ya**2 + za**2)
    ct = np.where(r > 0, za / np.where(r > 0, r, 1), 1.0)
    st_ = np.sqrt(np.clip(1 - ct**2, 0, None))
    ph = np.arctan2(ya, xa)
    cols = [np.ones_like(xa)]
    for l in range(1, order + 1):
        rl = r**l
        Pl = Legendre.basis(l)
        for m in range(0, l + 1):
            Plm = (Pl.deriv(m) if m else Pl)(ct) * st_**m
            cols.append(rl * Plm * np.cos(m * ph))
            if m > 0:
                cols.append(rl * Plm * np.sin(m * ph))
    return np.vstack(cols).T


def shim_any(dB, X, Y, Z, mask, order, w=None):
    """Residual after the best shim set up to `order` (0 to 4). With
    weights w (one per voxel of the mask) the fit minimises the weighted
    spread, which is what shimming on the signal does."""
    A = shim_basis(X, Y, Z, mask, order)
    b = np.asarray(dB[mask], float)
    if w is None:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    return b - A @ coef


def lineshape_w(res_ppm, f_MHz, nat_Hz=1.0, w=None, n=2001):
    """b0core.lineshape with a weight per voxel (the coil sensitivity).
    With w=None it is the same calculation. Returns (nu, spec, widths)."""
    hz = np.asarray(res_ppm, float).ravel() * f_MHz
    lo, hi = float(hz.min()), float(hz.max())
    span = hi - lo
    pad = max(40.0 * nat_Hz, 0.3 * span, 1e-6)
    full = span + 2 * pad
    n = int(max(n, 10 * full / max(nat_Hz, 1e-9))) | 1
    n = min(n, 200001)
    nu = np.linspace(lo - pad, hi + pad, n)
    d = nu[1] - nu[0]
    edges = np.concatenate([nu - d / 2, [nu[-1] + d / 2]])
    hist, _ = np.histogram(hz, bins=edges, weights=w)
    half = max(nat_Hz / 2.0, 1e-9)
    t = nu - nu[n // 2]
    kern = half**2 / (t**2 + half**2)
    if w is None:
        N, sd = hz.size, float(hz.std())
    else:                       # effective sample size and weighted spread
        N = float(w.sum())**2 / float((w**2).sum())
        mu = float((w * hz).sum() / w.sum())
        sd = float(np.sqrt((w * (hz - mu)**2).sum() / w.sum()))
    h = 1.06 * sd * N ** (-0.2) if (N > 1 and sd > 0) else 0.0
    if h > d:
        g = np.exp(-0.5 * (t / h) ** 2)
        kern = np.convolve(kern, g / g.sum(), mode="same")
    spec = np.convolve(hist.astype(float), kern, mode="same")
    m = spec.max()
    spec = spec / m if m > 0 else spec
    return nu, spec, {"fwhm": width_at(nu, spec, 0.5),
                      "w055": width_at(nu, spec, 0.0055),
                      "w011": width_at(nu, spec, 0.0011)}


def coil_weight(z, z0, z1, edge):
    """Signal weight along z for a coil whose window is z0..z1: its B1 is
    flat inside and falls smoothly to zero over about `edge` mm outside.
    A 90° pulse set at the centre excites sin(90°·b) and is received with
    b, so the weight is b·sin(90°·b). A simple model of a real coil, not
    a field calculation."""
    S = np.vectorize(lambda u: 0.5 * (1 + erf(1.2 * u)))
    e = max(edge, 1e-6)
    b = S((z - z0) / e) * S((z1 - z) / e)
    return b * np.sin(np.pi / 2 * b)


def analyse(sol, cfg):
    """Cheap step: detection region, shimming, lineshape. Re-run freely."""
    x, y, z, dB, liq = sol["x"], sol["y"], sol["z"], sol["dB"], sol["liq"]
    z0, z1, r = detection(cfg)
    soft = bool(cfg.get("d_soft"))
    if soft:
        wz = coil_weight(z, z0, z1, float(cfg["d_edge"]))
    else:
        wz = ((z >= z0) & (z <= z1)).astype(float)
    R2 = x[:, None]**2 + y[None, :]**2
    act = (R2[:, :, None] <= r*r) & (wz > 0.01)[None, None, :] & liq
    n = int(act.sum())
    if n < 50:
        return None
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij", sparse=True)
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    w = np.broadcast_to(wz[None, None, :], dB.shape)[act] if soft else None
    f = f_obs(cfg)
    order = int(cfg["sh_order"])

    def wstd(v):
        if w is None:
            return float(v.std())
        mu = (w * v).sum() / w.sum()
        return float(np.sqrt((w * (v - mu)**2).sum() / w.sum()))

    raw = np.asarray(dB[act], float)
    # how the answer depends on the shim order: the honest error bar on
    # "what would I measure", since nobody knows how well a real shim set
    # was actually set.
    by_order, keep = [], None
    for o in range(5):
        ro = shim_any(dB, X, Y, Z, act, o, w)
        nu_o, sp_o, wd_o = lineshape_w(ro, f, cfg["nat"], w)
        by_order.append((o, wstd(ro) * 1000, wd_o["fwhm"], wd_o["w055"]))
        if o == order:
            keep = (ro, nu_o, sp_o, wd_o)
    res, nu, sp, wd = keep
    rep = {"raw_ppm_pp": float(raw.max() - raw.min()),
           "raw_ppm_rms": wstd(raw),
           "shim_ppm_pp": float(res.max() - res.min()),
           "shim_ppm_rms": wstd(res),
           "shim_Hz_rms": wstd(res) * f}
    return dict(rep=rep, res=res, raw=raw, w=w, nu=nu, sp=sp, wd=wd, n=n,
                z0=z0, z1=z1, r=r, by_order=by_order, f=f, soft=soft)


def sweep(cfg, key, values, progress=None):
    """Solve the setup once per value of one parameter. Returns a list of
    dicts: value, fwhm (Hz), w055 (Hz), ppb (leftover rms), raw (ppm p-p).
    A value with no sample in the detected region gives nan."""
    rows = []
    for i, v in enumerate(values):
        c = dict(cfg)
        c[key] = float(v)
        try:
            an = analyse(solve(c), c)
        except Exception:            # an impossible geometry at this value
            an = None
        if an is None:
            rows.append(dict(value=float(v), fwhm=np.nan, w055=np.nan,
                             ppb=np.nan, raw=np.nan))
        else:
            rows.append(dict(value=float(v), fwhm=an["wd"]["fwhm"],
                             w055=an["wd"]["w055"],
                             ppb=an["rep"]["shim_ppm_rms"] * 1000,
                             raw=an["rep"]["raw_ppm_pp"]))
        if progress:
            progress((i + 1) / len(values))
    return rows


# ------------------------------------------------------- share links

def coerce(d):
    """A full, type-safe setup from a possibly partial or older dict."""
    new = dict(BLANK)
    for k, v in d.items():
        if k in BLANK:
            d0 = BLANK[k]
            try:
                new[k] = (bool(v) if isinstance(d0, bool) else
                          float(v) if isinstance(d0, float) else
                          int(v) if isinstance(d0, int) else str(v))
            except (TypeError, ValueError):
                pass
    custom = []
    for p in d.get("custom", []) or []:
        q = dict(CUSTOM_DEFAULT)
        q.update({k: p[k] for k in CUSTOM_DEFAULT if k in p})
        custom.append(q)
    return new, custom


def encode_setup(cfg):
    """Compact text for a URL: only what differs from the blank setup."""
    d = {k: v for k, v in cfg.items() if k in BLANK and BLANK[k] != v}
    if cfg.get("custom"):
        d["custom"] = cfg["custom"]
    raw = json.dumps(d, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


def decode_setup(code):
    pad = "=" * (-len(code) % 4)
    raw = zlib.decompress(base64.urlsafe_b64decode(code + pad))
    return coerce(json.loads(raw))


# ----------------------------------------------------------- self check

def self_check(V=0.25):
    """Solve a case with an exact answer on the same solver and voxel:
    a uniform sphere (radius 3 mm, chi = 1 ppm) in empty space.
    Exact: zero field change inside (with the Lorentz correction), and
    outside a dipole, (chi/3)(a/r)^3 (3cos²θ - 1).
    Returns the worst errors, as a fraction of the largest exact value."""
    a, chi0, half = 3.0, 1.0, 9.0
    x = np.arange(-half + V/2, half, V)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij", sparse=True)
    s = 4
    offs = ((np.arange(s) + 0.5) / s - 0.5) * V
    f = np.zeros((x.size,) * 3, np.float32)
    for ox in offs:                       # partial-volume sphere
        for oy in offs:
            for oz in offs:
                f += ((X+ox)**2 + (Y+oy)**2 + (Z+oz)**2) <= a*a
    f /= s**3
    dB = field_fast(f * chi0, V).astype(float)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R = np.broadcast_to(R, dB.shape)
    Zb = np.broadcast_to(Z, dB.shape)
    exact = np.where(R > a, chi0/3 * (a/np.maximum(R, 1e-9))**3
                     * (3*(Zb/np.maximum(R, 1e-9))**2 - 1), 0.0)
    scale = 2 * chi0 / 3                  # largest exact value (on the pole)
    inside = R < a - 0.8
    outside = (R > a + 0.8) & (R < 2*a)
    return dict(inside=float(np.abs(dB[inside] - dB[inside].mean()).max()
                             / scale),
                outside=float(np.abs((dB - dB[inside].mean() - exact)
                                     [outside]).max() / scale),
                voxel=V, n=int(x.size))


# ----------------------------------------------------- measured spectra

def parse_spectrum(text, unit="Hz", f_MHz=500.0):
    """Two columns of numbers (frequency, intensity) from a text or CSV
    file, any separator, header lines ignored. Returns frequency in Hz
    centred on the tallest point and intensity scaled to 1, baseline (the
    outer 5 % at each end) removed. None if it cannot be read."""
    rows = []
    for line in text.splitlines():
        for ch in ",;\t":
            line = line.replace(ch, " ")
        parts = line.split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    if len(rows) < 8:
        return None
    a = np.array(rows)
    a = a[np.argsort(a[:, 0])]
    nu, y = a[:, 0], a[:, 1]
    if unit == "ppm":
        nu = nu * f_MHz
    k = max(2, len(y) // 20)
    y = y - np.median(np.r_[y[:k], y[-k:]])
    if y.max() <= 0:
        return None
    y = y / y.max()
    nu = nu - nu[int(np.argmax(y))]
    # width at half height, by walking out from the peak
    i = int(np.argmax(y)); l = i; r = i
    while l > 0 and y[l] > 0.5:
        l -= 1
    while r < len(y) - 1 and y[r] > 0.5:
        r += 1
    def cross(i0, i1):
        if y[i0] == y[i1]:
            return nu[i0]
        return nu[i0] + (0.5 - y[i0]) * (nu[i1] - nu[i0]) / (y[i1] - y[i0])
    fwhm = cross(r - 1, r) - cross(l, l + 1) if r > l + 1 else np.nan
    return dict(nu=nu, y=y, fwhm=float(fwhm))
