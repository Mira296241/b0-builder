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
import numpy as np

from b0core import (MATERIALS_PPM, build_tube, build_coil, build_cwg,
                    build_annulus, shim, report, lineshape_report)

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
    t_glass_chi=-11.0, t_liquid_chi=-9.0,
    # RF coil
    c_on=False, c_r=3.5, c_t=50.0, c_len=18.0, c_zc=20.0, c_mat="copper",
    c_nslots=2, c_slotdeg=90.0, c_phase=90.0, c_win=False, c_winz0=3.0,
    c_winh=12.0, c_slit=1.0,
    # waveguide
    w_on=False, w_id=5.0, w_wall=1.25, w_len=20.0, w_mat="brass",
    # RF shields
    rs_on=False, rs_bot=True, rs_side=True, rs_mat="copper",
    rs_b_rin=8.0, rs_b_rout=17.925, rs_b_z0=-0.74, rs_b_z1=-0.55,
    rs_s_rin=9.0, rs_s_rout=9.1, rs_s_z0=4.95, rs_s_z1=25.05,
    # modulation coil
    m_on=False, m_rin=9.1, m_rout=12.7, m_z0=5.0, m_z1=24.55, m_fill=0.75,
    m_mat="copper",
    # detection and shimming
    d_auto=True, d_r=1.5, d_z0=10.0, d_z1=25.0,
    sh_order=2, f_MHz=500.0, nat=0.5,
    # numerics
    V=0.25,
)

STANDARD = dict(BLANK, t_wall=0.4, t_liquid="chloroform-d",
                c_on=True, c_r=3.5, c_t=50.0, c_len=18.0, c_zc=20.0,
                d_r=1.8)

FRASCAN = dict(BLANK,
    t_OD=5.0, t_wall=0.5, t_fill=45.0, t_glass="fused quartz",
    t_liquid="toluene", t_tip=0.5, t_bottom="round", t_botdepth=2.0,
    c_on=True, c_r=5.0, c_t=100.0, c_len=39.0, c_zc=15.0, c_nslots=2,
    c_slotdeg=88.0, c_phase=20.0, c_win=True, c_winz0=12.0, c_winh=15.0,
    c_slit=1.2,
    w_on=True, w_id=5.0, w_wall=1.25, w_len=20.0, w_mat="brass",
    rs_on=True, rs_bot=True, rs_side=True,
    m_on=True,
    d_auto=True, d_r=1.5,
)

PRESETS = {
    "Blank (tube only)": (BLANK, "A tube of water in a perfect magnet. "
                          "Add the parts your probe has."),
    "Standard NMR probe": (STANDARD, "5 mm tube, CDCl₃, a plain copper "
                           "coil. Real commercial coils use "
                           "susceptibility-matched wire, so this is a "
                           "pessimistic version."),
    "FRASCAN II": (FRASCAN, "Our DNP probe, from the Fusion CAD (Oct 2026): "
                   "slotted coil with windows, brass waveguide, RF "
                   "shields, modulation coil."),
}

CUSTOM_DEFAULT = dict(kind="cylinder / ring", mat="PTFE", r_in=0.0,  # a small PTFE lens below the tube
                      r_out=2.0, cx=0.0, cy=0.0, sx=4.0, sy=4.0,
                      z0=-6.0, z1=-1.0, fill=1.0)


# ---------------------------------------------------------- materials

def chi_of(name, cfg):
    """Susceptibility in ppm, including the two custom entries."""
    if name == "custom (tube)":
        return float(cfg["t_glass_chi"])
    if name == "custom (sample)":
        return float(cfg["t_liquid_chi"])
    return MATERIALS_PPM[name]


def _register_custom(cfg):
    MATERIALS_PPM["custom (tube)"] = float(cfg["t_glass_chi"])
    MATERIALS_PPM["custom (sample)"] = float(cfg["t_liquid_chi"])


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
        if p["kind"] == "box":
            out.append(("box", dict(cx=p["cx"], cy=p["cy"], sx=p["sx"],
                        sy=p["sy"], z0=p["z0"], z1=p["z1"], fill=p["fill"],
                        material=p["mat"]), "custom"))
        else:
            out.append(("cyl", dict(cx=p["cx"], cy=p["cy"], r_in=p["r_in"],
                        r_out=p["r_out"], z0=p["z0"], z1=p["z1"],
                        fill=p["fill"], material=p["mat"]), "custom"))
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
    zhi = cfg["t_tip"] + cfg["t_wall"] + cfg["t_fill"] + 20.0
    if cfg["c_on"]:
        r = max(r, cfg["c_r"] + cfg["c_t"] * 1e-3)
        zlo = min(zlo, cfg["c_zc"] - cfg["c_len"]/2)
        zhi = max(zhi, cfg["c_zc"] + cfg["c_len"]/2)
    if cfg["w_on"]:
        r = max(r, cfg["w_id"]/2 + cfg["w_wall"])
        zlo = min(zlo, -cfg["w_len"])
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
    _register_custom(cfg)
    out = cfg["t_out"]
    kw = dict(tube_OD=cfg["t_OD"], wall=cfg["t_wall"], d_cwg=cfg["t_tip"],
              fill_h=cfg["t_fill"], bottom=cfg["t_bottom"],
              punt_r=cfg["t_punt"], bot_depth=cfg["t_botdepth"],
              glass=cfg["t_glass"], liquid=cfg["t_liquid"], outside=out,
              men_depth=cfg["t_men"])
    chi, liq, gls = build_tube(x, y, z, **kw)
    masks = {"glass": gls, "sample": liq}
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
                             length=cfg["w_len"], material=cfg["w_mat"],
                             outside=out)
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
    """dBz/B0 in ppm. Same as b0core.field_shift_ppm, real FFT, slab-wise
    kernel so the padded k-space never needs a second full-size array."""
    n = chi.shape
    N = tuple(int(s * pad) for s in n)
    F = _fft.rfftn(chi, s=N, **_KW)
    ky = np.fft.fftfreq(N[1], d=V)[:, None]
    kz = np.fft.rfftfreq(N[2], d=V)[None, :]
    kx = np.fft.fftfreq(N[0], d=V)
    kyz2 = ky**2 + kz**2
    kz2 = kz**2
    for i in range(N[0]):
        k2 = kx[i]**2 + kyz2
        with np.errstate(divide="ignore", invalid="ignore"):
            D = 1.0/3.0 - kz2 / k2
        if i == 0:
            D[0, 0] = 0.0
        F[i] *= D
    out = _fft.irfftn(F, s=N, **_KW)
    return np.ascontiguousarray(out[:n[0], :n[1], :n[2]])


def solve(cfg):
    """The expensive step: geometry + FFT. Returns a dict the app caches."""
    x, y, z, V, note = choose_grid(cfg)
    chi, masks = build_all(cfg, x, y, z)
    dB = field_fast(chi, V)
    return dict(x=x, y=y, z=z, V=V, note=note, dB=dB, liq=masks["sample"],
                j=len(y)//2)


def analyse(sol, cfg):
    """Cheap step: detection region, shimming, lineshape. Re-run freely."""
    x, y, z, dB, liq = sol["x"], sol["y"], sol["z"], sol["dB"], sol["liq"]
    z0, z1, r = detection(cfg)
    R2 = x[:, None]**2 + y[None, :]**2
    act = (R2[:, :, None] <= r*r) & (z >= z0)[None, None, :] \
        & (z <= z1)[None, None, :] & liq
    n = int(act.sum())
    if n < 50:
        return None
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij", sparse=True)
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    order = int(cfg["sh_order"])
    rep = report(dB, act, X, Y, Z, f_MHz=cfg["f_MHz"], order=order)
    res = shim(dB, X, Y, Z, act, order=order)
    nu, sp, wd = lineshape_report(res, cfg["f_MHz"], nat_Hz=cfg["nat"])
    return dict(rep=rep, res=res, raw=dB[act], nu=nu, sp=sp, wd=wd,
                n=n, z0=z0, z1=z1, r=r)
