"""
B0 homogeneity from magnetic susceptibility.

Method: k-space (Fourier) convolution with the dipole kernel.
    dBz(r)/B0  =  IFFT[ FFT(chi) * (1/3 - kz^2/k^2) ]
Standard in MRI: Salomir et al. 2003; Marques & Bowtell 2005.
Valid to first order in chi, which is exact enough for chi ~ 1e-5.
"""

import numpy as np

# Volume magnetic susceptibility, SI, in ppm (dimensionless x 1e-6).
# Every value carries a source in SOURCES below. "measured" means taken
# directly from a measurement paper; "estimate" means a literature or
# handbook value that has NOT been checked against a measurement here.
MATERIALS_PPM = {
    "vacuum":              0.0,
    "air":                 0.36,
    # ---- tube and probe materials
    "fused quartz":      -11.27,
    "borosilicate":      -11.06,
    "sapphire":          -18.0,
    "alumina":           -18.0,
    "zirconia":          -13.0,
    "PTFE":              -10.28,
    "PEEK":               -9.33,
    "POM (Delrin)":       -9.37,
    "Vespel":             -9.02,
    "copper":             -9.63,
    "silver":            -24.0,
    "aluminium 7075":     13.80,
    "brass":             -13.0,
    # ---- solvents and matrices
    "water":              -9.04,
    "D2O":                -8.99,
    "toluene":            -7.8,
    "toluene-d8":         -7.95,
    "DMSO":               -7.80,
    "chloroform-d":       -9.28,
    "acetone-d6":         -5.74,
    "acetonitrile-d3":    -6.70,
    "benzene-d6":         -7.51,
    "methanol-d4":        -6.76,
    "THF-d8":             -8.01,
    "methanol":           -6.6,
    "ethanol":            -7.2,
    "glycerol/water 60:40": -8.9,
}

_W14 = "Wapler et al., J Magn Reson 242 (2014) 233, arXiv:1403.4760"
_H22 = "Hoffman, J Magn Reson 335 (2022) 107105, Table 3"
SOURCES = {
    "fused quartz": ("measured", _W14 + ", Table 4, Schott Lithosil quartz: "
                     "-2.24 relative to water at -9.032"),
    "borosilicate": ("measured", _W14 + ", Table 4, Schott Borofloat 33: "
                     "-2.024 relative to water at -9.032"),
    "PTFE":         ("measured", _W14 + ", Table 7"),
    "PEEK":         ("measured", _W14 + ", Table 7"),
    "POM (Delrin)": ("measured", _W14 + ", Table 7, POM white"),
    "Vespel":       ("measured", _W14 + ", Table 7, from their ref. [9]"),
    "water":        ("measured", _H22 + " text: -9.043 and -9.041"),
    "D2O":          ("measured", _H22),
    "toluene-d8":   ("measured", _H22),
    "DMSO":         ("measured", _H22 + ", DMSO-d6; protiated is within 1 %"),
    "chloroform-d": ("measured", _H22),
    "acetone-d6":   ("measured", _H22),
    "acetonitrile-d3": ("measured", _H22),
    "benzene-d6":   ("measured", _H22),
    "methanol-d4":  ("measured", _H22),
    "THF-d8":       ("measured", _H22),
    "toluene":      ("derived", _H22 + ": protiated molar value 830.7 over "
                     "molar volume; uncertain by about 1 %"),
    "air":          ("estimate", "O2 paramagnetism, handbook value"),
    "copper":       ("estimate", "handbook value"),
    "aluminium 7075": ("measured",
                     "Schonzart et al., J Magn Reson 364 (2024) 107704: "
                     "chi = 1.38e-5 SI, the rod used for their experimental "
                     "validation"),
    "silver":       ("estimate", "handbook value"),
    "brass":        ("estimate", "between copper (-9.6) and zinc (-15.7); real "
                     "brass can be slightly magnetic from iron impurities. "
                     "MEASURE or ask the manufacturer before trusting it"),
    "sapphire":     ("estimate", "handbook value, anisotropic in reality"),
    "alumina":      ("estimate", "handbook value"),
    "zirconia":     ("estimate", "handbook value"),
    "methanol":     ("estimate", "handbook value; see methanol-d4"),
    "ethanol":      ("estimate", "handbook value"),
    "glycerol/water 60:40": ("estimate", "volume-weighted mix, not measured"),
    "vacuum":       ("exact", "zero by definition"),
}


def dipole_kernel(shape, voxel):
    """(1/3 - kz^2/k^2) on the FFT grid. DC term set to 0 (a uniform
    offset is not observable: it is absorbed by the field/frequency lock)."""
    nx, ny, nz = shape
    dx, dy, dz = voxel
    kx = np.fft.fftfreq(nx, d=dx)[:, None, None]
    ky = np.fft.fftfreq(ny, d=dy)[None, :, None]
    kz = np.fft.fftfreq(nz, d=dz)[None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    with np.errstate(divide="ignore", invalid="ignore"):
        D = 1.0/3.0 - (kz**2)/k2
    D[0, 0, 0] = 0.0
    return D


def field_shift_ppm(chi_ppm, voxel, pad=2):
    """chi_ppm: 3D array of susceptibility in ppm. Returns dBz/B0 in ppm,
    same shape. Zero-padded by `pad` to suppress wrap-around."""
    n = np.array(chi_ppm.shape)
    N = (n * pad).astype(int)
    big = np.zeros(N, dtype=np.float64)
    s = tuple(slice(0, n[i]) for i in range(3))
    big[s] = chi_ppm
    D = dipole_kernel(tuple(N), voxel)
    out = np.fft.ifftn(np.fft.fftn(big) * D).real
    return out[s]


# ---------------------------------------------------------------- geometry

def grid(xy_half, z_lo, z_hi, voxel):
    """Cell-centred coordinate arrays."""
    dx = dy = dz = voxel
    x = np.arange(-xy_half + dx/2, xy_half, dx)
    y = np.arange(-xy_half + dy/2, xy_half, dy)
    z = np.arange(z_lo + dz/2, z_hi, dz)
    return x, y, z


def build_tube(x, y, z, *, tube_OD=4.0, wall=0.5, d_cwg=5.0, fill_h=30.0,
               bottom="round", punt_r=2.0, bot_depth=2.0,
               glass="fused quartz", liquid="toluene", outside="air",
               men_depth=0.0,
               insert=None, inner_tube=None, return_regions=False):
    """Susceptibility map (ppm) of tube + liquid. z in mm, origin at the CWG
    aperture, tube axis along z (which is B0)."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    ro, ri = tube_OD/2, tube_OD/2 - wall

    if bottom == "round":                      # hemispherical, tip at d_cwg
        zj = d_cwg + bot_depth
        outer = ((R <= ro) & (Z >= zj)) | ((R**2 + ((Z-zj)*ro/bot_depth)**2 <= ro**2) & (Z < zj))
        bi = max(bot_depth - wall, 1e-6)
        inner_floor = zj - bi*np.sqrt(np.clip(1-(R/ri)**2, 0, None))
        cavity = (R <= ri) & (Z >= inner_floor)
    elif bottom == "flat":
        outer = (R <= ro) & (Z >= d_cwg)
        cavity = (R <= ri) & (Z >= d_cwg + wall)
    elif bottom == "punt":                     # dome carved up into the bottom
        dome_o = (R <= punt_r) & (Z <= d_cwg + np.sqrt(np.clip(punt_r**2 - R**2, 0, None)))
        outer = (R <= ro) & (Z >= d_cwg) & ~dome_o
        dome_i = (R <= punt_r) & (Z <= d_cwg + wall + np.sqrt(np.clip(punt_r**2 - R**2, 0, None)))
        cavity = (R <= ri) & (Z >= d_cwg + wall) & ~dome_i
    else:
        raise ValueError(bottom)

    # ---- liquid surface, flat or concave (men_depth > 0 dips on the axis)
    zfloor = d_cwg + wall
    ztop = zfloor + fill_h
    if men_depth > 0:
        surf = ztop - men_depth*(1.0 - np.clip((R/ri)**2, 0, 1))
    else:
        surf = np.full(R.shape, ztop)
    liquid_m = cavity & (Z <= surf)
    glass_m = outer & ~cavity

    # ---- optional coaxial inner tube (glass annulus, liquid inside too)
    inner_m = np.zeros(X.shape, bool)
    if inner_tube:
        i_ro = inner_tube["OD"]/2
        i_ri = i_ro - inner_tube["wall"]
        i_z0 = inner_tube["z_bottom"]
        col = (Z >= i_z0) & cavity
        inner_m = col & (R <= i_ro) & (R > i_ri)
        glass_m = glass_m | inner_m
        liquid_m = liquid_m & ~inner_m

    # ---- optional insert resting in the bottom
    ins_m = np.zeros(X.shape, bool)
    if insert:
        if insert["kind"] == "ball":
            Rb = insert["radius"]
            zc = insert.get("z_centre", zfloor + Rb)
            ins_m = (X**2 + Y**2 + (Z-zc)**2 <= Rb**2)
        else:                                   # domed-top plug
            rp = min(insert.get("radius", ri*0.97), ri)
            h  = insert["height"]
            d  = insert.get("dome", 0.0)
            z0 = insert.get("z_bottom", zfloor)
            body = (R <= rp) & (Z >= z0) & (Z <= z0 + h)
            if d > 0:
                dome = (R <= rp) & (Z > z0 + h) & \
                       (Z <= z0 + h + d*np.sqrt(np.clip(1-(R/rp)**2, 0, None)))
                body = body | dome
            ins_m = body
        ins_m &= cavity
        liquid_m = liquid_m & ~ins_m

    chi = np.full(X.shape, MATERIALS_PPM[outside], dtype=np.float64)
    chi[glass_m] = MATERIALS_PPM[glass]
    chi[liquid_m] = MATERIALS_PPM[liquid]
    if inner_tube:
        chi[inner_m] = MATERIALS_PPM[inner_tube.get("material", glass)]
    if insert:
        chi[ins_m] = MATERIALS_PPM[insert.get("material", "fused quartz")]
    if return_regions:
        tube_only = glass_m & ~inner_m
        regions = {
            "tube":       (tube_only, MATERIALS_PPM[glass]),
            "sample":     (liquid_m, MATERIALS_PPM[liquid]),
            "surroundings": (~(glass_m | liquid_m | ins_m),
                             MATERIALS_PPM[outside]),
        }
        if insert and ins_m.any():
            regions["insert"] = (
                ins_m, MATERIALS_PPM[insert.get("material", "fused quartz")])
        if inner_tube and inner_m.any():
            regions["inner tube"] = (
                inner_m, MATERIALS_PPM[inner_tube.get("material", glass)])
        return chi, liquid_m, (glass_m | ins_m), regions
    return chi, liquid_m, (glass_m | ins_m)


# ---------------------------------------------------------------- analysis

def shim_matrix(X, Y, Z, mask, order=2):
    """Columns the shim set can produce, sampled over the active volume."""
    pts = [np.ones(mask.sum())]
    xa, ya, za = X[mask], Y[mask], Z[mask]
    if order >= 1:
        pts += [xa, ya, za]
    if order >= 2:
        pts += [xa*ya, xa*za, ya*za, xa**2 - ya**2, 2*za**2 - xa**2 - ya**2]
    return np.vstack(pts).T


def shim(dB, X, Y, Z, mask, order=2, coef=None):
    """Remove the part a shim set could remove: fit and subtract low-order
    polynomials over the active volume. Returns the residual.

    order=None     no shimming at all, the raw field over the mask.
    coef=<array>   do not refit: subtract the shim currents given, which is
                   what a frozen-shim experiment does. Get them from
                   shim_coef on the unperturbed geometry.
    """
    if order is None:
        return dB[mask]
    A = shim_matrix(X, Y, Z, mask, order=order)
    if coef is None:
        coef, *_ = np.linalg.lstsq(A, dB[mask], rcond=None)
    return dB[mask] - A @ np.asarray(coef, float)


def shim_coef(dB, X, Y, Z, mask, order=2):
    """The shim currents that best flatten this field. Fit once on the
    unperturbed sample, then pass to shim(..., coef=...) for every
    perturbed case, exactly as the shims are frozen on the bench."""
    A = shim_matrix(X, Y, Z, mask, order=order)
    coef, *_ = np.linalg.lstsq(A, dB[mask], rcond=None)
    return coef


def report(dB, mask, X, Y, Z, f_MHz=500.0, order=2, coef=None):
    raw = dB[mask]
    res = shim(dB, X, Y, Z, mask, order=order, coef=coef)
    return {
        "raw_ppm_pp":  float(raw.max() - raw.min()),
        "raw_ppm_rms": float(raw.std()),
        "shim_ppm_pp": float(res.max() - res.min()),
        "shim_ppm_rms": float(res.std()),
        "shim_Hz_rms": float(res.std() * f_MHz),
        "shim_Hz_pp":  float((res.max() - res.min()) * f_MHz),
    }


def scan_windows(dB, X, Y, Z, liq, r_act, win_len, *, step=0.5,
                 order=2, f_MHz=500.0):
    """Slide an active volume of fixed length along z and shim at each
    position. The field dB is computed once and reused, so this costs
    almost nothing compared with the FFT that produced it.

    Returns a list of (z_bottom, z_top, rms_ppm, rms_Hz), empty if the
    window never fits inside the liquid."""
    R = np.sqrt(X**2 + Y**2)
    base = (R <= r_act) & liq
    if not base.any():
        return []
    zl = Z[base]
    z_lo, z_hi = float(zl.min()), float(zl.max())
    out = []
    zc = z_lo
    while zc + win_len <= z_hi + 1e-9:
        act = base & (Z >= zc) & (Z <= zc + win_len)
        if act.sum() >= 50:
            r = shim(dB, X, Y, Z, act, order=order)
            out.append((zc, zc + win_len, float(r.std()),
                        float(r.std() * f_MHz)))
        zc += step
    return out


def build_coil(x, y, z, *, coil_r=5.0, thickness_um=50.0, length=15.0,
               z_centre=15.0, n_slots=2, slot_deg=30.0, slot_phase=90.0,
               material="copper", window=None):
    """Slotted-tube resonator: a thin conducting cylinder with n axial slots.

    Real foil is tens of microns, far below any practical voxel. For a thin
    sheet the perturbation depends on the product chi x thickness, so the
    foil is placed in a single voxel layer with chi scaled to preserve it.
    Returns (chi_ppm_contribution, mask)."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    V = float(x[1] - x[0])

    t_mm = thickness_um * 1e-3
    layers = max(1, int(round(t_mm / V)))
    t_model = layers * V
    chi_eff = MATERIALS_PPM[material] * (t_mm / t_model)

    shell = (R >= coil_r) & (R < coil_r + t_model)
    axial = (np.abs(Z - z_centre) <= length/2)
    mask = shell & axial

    if n_slots > 0 and slot_deg > 0:
        phi = np.degrees(np.arctan2(Y, X)) % 360.0
        keep = np.ones_like(mask)
        z_bot = z_centre - length/2
        for i in range(n_slots):
            c = slot_phase + 360.0 * i / n_slots
            d = np.abs((phi - c + 180.0) % 360.0 - 180.0)
            if window is None:
                # full-length axial slots
                keep &= ~(d <= slot_deg/2)
            else:
                # FRASCAN-style cutout: a window of height win_h starting
                # win_z0 above the coil bottom, fed by a narrow slit of
                # arc width slit_mm running from the coil bottom up to it
                w0 = z_bot + window["win_z0"]
                in_win = (d <= slot_deg/2) & (Z >= w0) & (Z <= w0 + window["win_h"])
                slit_deg = np.degrees(window["slit_mm"] / coil_r)
                in_slit = (d <= slit_deg/2) & (Z >= z_bot) & (Z < w0)
                keep &= ~(in_win | in_slit)
        mask &= keep

    chi = np.zeros(X.shape)
    chi[mask] = chi_eff
    return chi, mask


# ------------------------------------------------------ probe-head parts

def build_annulus(x, y, z, *, r_in, r_out, z0, z1, material="copper",
                  outside="air", fill=1.0, subsample=5):
    """A coaxial ring or shell of metal: r_in..r_out, z0..z1 (internal z,
    where 0 is the waveguide top). Used for the RF shields and the
    modulation coil. Partial volume in r and exact overlap in z, so parts
    thinner than a voxel (0.1 mm shield, 0.19 mm disc) keep the right
    chi x thickness. `fill` < 1 for a winding of round wire.
    Returns (chi_add_ppm, mask)."""
    dchi = (MATERIALS_PPM[material] - MATERIALS_PPM[outside]) * fill
    h = x[1] - x[0]
    s = int(max(1, subsample))
    offs = (np.arange(s) + 0.5) / s - 0.5
    fr = np.zeros((x.size, y.size))
    for ox in offs * h:
        for oy in offs * h:
            r = np.hypot(x[:, None] + ox, y[None, :] + oy)
            fr += (r >= r_in) & (r <= r_out)
    fr /= s ** 2
    dz = z[1] - z[0]
    fz = np.clip((np.minimum(z + dz/2, z1) - np.maximum(z - dz/2, z0)) / dz, 0, 1)
    f3 = fr[:, :, None] * fz[None, None, :]
    return f3 * dchi, f3 > 0.25


# --------------------------------------------------------- waveguide

def build_cwg(x, y, z, *, inner_d=5.0, wall=1.0, length=20.0,
              z_top=0.0, material="brass", outside="air", subsample=3):
    """The corrugated waveguide (CWG) end below the tube: a metal tube on
    the axis, its top face at z_top (the launcher aperture, z = 0 in this
    tool's frame, so the tube tip sits d_cwg above it), running down for
    `length` mm. Corrugations are ignored: they are far smaller than the
    distance to the sample and average out.

    Returns (chi_add_ppm, mask): a fill-fraction-weighted CONTRIBUTION to
    add to the tube map, like build_rod. Its top rim is the dominant
    source, exactly as for the rod's end face, which is why the gap
    between the CWG and the sample matters so much.
    """
    dchi = MATERIALS_PPM[material] - MATERIALS_PPM[outside]
    ri, ro = inner_d / 2.0, inner_d / 2.0 + wall
    hx = x[1] - x[0]
    s = int(max(1, subsample))
    offs = (np.arange(s) + 0.5) / s - 0.5
    frac = np.zeros((x.size, y.size))
    for ox in offs * hx:
        for oy in offs * hx:
            r = np.hypot(x[:, None] + ox, y[None, :] + oy)
            frac += (r >= ri) & (r <= ro)
    frac /= s ** 2
    zf = np.clip((np.minimum(z + (z[1]-z[0])/2, z_top)
                  - np.maximum(z - (z[1]-z[0])/2, z_top - length))
                 / (z[1]-z[0]), 0, 1)
    f3 = frac[:, :, None] * zf[None, None, :]
    return f3 * dchi, f3 > 0.5


# ------------------------------------------------------- external rod

def build_rod(x, y, z, *, material="aluminium 7075", diameter=3.0,
              length=40.0, distance=6.0, z_end=15.0, phi_deg=0.0,
              vertical=True, extends="up", outside="air", subsample=3):
    """A rod of known susceptibility standing outside the tube.

    This is the perturber of the Schonzart validation experiment
    (J Magn Reson 364 (2024) 107704): shim on the bare sample, freeze the
    shims, bring a rod of known chi up beside the tube, and compare the
    lineshape with the simulation. Only the rod height was fitted there.

    Why the height is the whole story: a long rod PARALLEL to B0 is
    magnetised along its own axis, so div(M) vanishes everywhere except at
    its two end faces. An infinitely long vertical rod therefore produces
    NO field outside itself at all. What the sample sees is the field of
    the nearer end face, a disc of magnetic surface charge sigma = M. That
    is why the distance from that end face to the sample is the sensitive
    parameter, and why the effect dies as 1/d^2 once d exceeds the radius.
    Turn `vertical` off and the rod lies along y, perpendicular to B0; then
    it is a line dipole and the external field is far stronger, falling
    only as 1/rho^2.

    Geometry, in the frame the rest of the tool uses (z = 0 at the CWG
    aperture, tube tip at d_cwg, B0 along +z):
        distance : rod axis to tube axis, mm
        phi_deg  : azimuth of the rod around the tube
        z_end    : z of the rod end FACE that faces the sample. This is
                   the one number the experiment sweeps.
        extends  : which way the rest of the rod goes from that face,
                   "up" or "down". "up" is a rod lowered into a vertical
                   bore from above, its lower tip at z_end. The two are
                   mirror images and give the same magnitude, so this only
                   decides whether the rod BODY runs alongside the sample
                   or away from it.
        length   : rod length. Make it long enough that the far end is
                   irrelevant, which the validation script checks.

    Returns (chi_add_ppm, mask). chi_add is a fill-fraction-weighted
    susceptibility CONTRIBUTION, to be added to the tube's map, not
    written over it. Partial volume matters here: the end face is the
    entire source of the field, and a hard in/out test staircases it into
    a few percent error on exactly the number being validated.
    """
    chi_rod = MATERIALS_PPM[material] - MATERIALS_PPM[outside]
    a = diameter / 2.0
    ph = np.radians(phi_deg)
    cx, cy = distance * np.cos(ph), distance * np.sin(ph)

    if vertical:
        up = 1.0 if str(extends).lower().startswith("u") else -1.0
        ax = np.array([0.0, 0.0, up])
        base = np.array([cx, cy, z_end])
    else:
        ax = np.array([0.0, 1.0, 0.0])          # lies along y, _|_ B0
        base = np.array([cx, -length / 2.0, z_end])

    hx, hy, hz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    s = int(max(1, subsample))
    offs = (np.arange(s) + 0.5) / s - 0.5
    frac = np.zeros((x.size, y.size, z.size))
    for ox in offs * hx:
        X = (x + ox)[:, None, None] - base[0]
        for oy in offs * hy:
            Y = (y + oy)[None, :, None] - base[1]
            for oz in offs * hz:
                Z = (z + oz)[None, None, :] - base[2]
                t = X * ax[0] + Y * ax[1] + Z * ax[2]
                perp2 = X * X + Y * Y + Z * Z - t * t
                frac += (t >= 0) & (t <= length) & (perp2 <= a * a)
    frac /= s ** 3
    return frac * chi_rod, frac > 0.5


# ------------------------------------------------------------- lineshape

def lineshape(res_ppm, f_MHz, nat_Hz=1.0, n=2001):
    """Turn the residual field distribution into the line a spectrometer
    would actually show.

    Every voxel in the active volume resonates at its own offset,
    nu = (dB/B0) * f0. The histogram of those offsets IS the inhomogeneous
    lineshape, to first order. Convolving it with the natural Lorentzian
    gives the observed line.

    The grid is refined until the Lorentzian spans at least ~10 samples,
    otherwise a narrow natural line falls between bins and the width comes
    out as noise.

    Returns (nu_Hz, intensity) with the peak normalised to 1.
    """
    hz = np.asarray(res_ppm, dtype=float).ravel() * f_MHz
    lo, hi = float(hz.min()), float(hz.max())
    span = hi - lo
    # 20x the natural width so the Lorentzian foot is still on the
    # grid at 0.11 % of the peak, which is where it is measured.
    pad = max(40.0 * nat_Hz, 0.3 * span, 1e-6)
    full = span + 2 * pad
    n = int(max(n, 10 * full / max(nat_Hz, 1e-9))) | 1
    n = min(n, 200001)
    nu = np.linspace(lo - pad, hi + pad, n)
    d = nu[1] - nu[0]
    edges = np.concatenate([nu - d / 2, [nu[-1] + d / 2]])
    hist, _ = np.histogram(hz, bins=edges)
    half = max(nat_Hz / 2.0, 1e-9)
    t = nu - nu[n // 2]
    kern = half**2 / (t**2 + half**2)

    # The voxels are a finite sample of a continuous field, so the raw
    # histogram carries sampling noise that looks like structure in the
    # line but is not. Smooth it with the Silverman bandwidth, which is
    # the finest detail this many voxels can actually support. It goes to
    # zero for a uniform field, so an analytic Lorentzian is unaffected.
    N = hz.size
    sd = float(hz.std())
    h = 1.06 * sd * N ** (-0.2) if (N > 1 and sd > 0) else 0.0
    if h > d:
        g = np.exp(-0.5 * (t / h) ** 2)
        kern = np.convolve(kern, g / g.sum(), mode="same")

    spec = np.convolve(hist.astype(float), kern, mode="same")
    m = spec.max()
    return nu, (spec / m if m > 0 else spec), h


def _shape_only(res_ppm, f_MHz, nat_Hz=1.0, n=2001):
    nu, sp, _ = lineshape(res_ppm, f_MHz, nat_Hz=nat_Hz, n=n)
    return nu, sp


def fwhm(nu, spec):
    """Full width at half maximum of the MAIN peak, by walking outwards
    from the maximum to the first crossing of half height on each side.
    Taking the outermost crossings instead would measure side structure,
    not the line. nan if the line does not fall to half on both sides."""
    spec = np.asarray(spec, dtype=float)
    p = int(np.argmax(spec))
    h = spec[p] / 2.0

    def edge(step):
        i = p
        while 0 < i < len(spec) - 1 and spec[i] > h:
            i += step
        if spec[i] > h:
            return None
        j = i - step
        if spec[j] == spec[i]:
            return nu[i]
        return nu[i] + (h - spec[i]) * (nu[j] - nu[i]) / (spec[j] - spec[i])

    left, right = edge(-1), edge(+1)
    if left is None or right is None:
        return float("nan")
    return float(abs(right - left))


def width_at(nu, spec, frac):
    """Width of the main peak at a given fraction of its height, walking
    outwards from the maximum.

    NMR shim quality is quoted exactly this way: the standard lineshape
    test reports the width at 50 %, 0.55 % and 0.11 % of the peak. The
    low ones catch the broad foot that a single FWHM number hides, which
    is precisely the part susceptibility mismatch produces.
    """
    spec = np.asarray(spec, dtype=float)
    p = int(np.argmax(spec))
    h = spec[p] * frac

    def edge(step):
        i = p
        while 0 < i < len(spec) - 1 and spec[i] > h:
            i += step
        if spec[i] > h:
            return None
        j = i - step
        if spec[j] == spec[i]:
            return nu[i]
        return nu[i] + (h - spec[i]) * (nu[j] - nu[i]) / (spec[j] - spec[i])

    a, b = edge(-1), edge(+1)
    return float("nan") if a is None or b is None else float(abs(b - a))


def lineshape_report(res_ppm, f_MHz, nat_Hz=1.0):
    """The three standard lineshape numbers, in Hz, plus the sampling
    resolution. If that resolution is comparable with the natural width,
    the line is limited by the voxel count and a finer voxel is needed
    before the numbers mean anything."""
    nu, sp, h = lineshape(res_ppm, f_MHz, nat_Hz=nat_Hz)
    return nu, sp, {
        "fwhm": width_at(nu, sp, 0.5),
        "w055": width_at(nu, sp, 0.0055),
        "w011": width_at(nu, sp, 0.0011),
        "resolution": h,
    }


def region_basis(x, y, z, kw, voxel, X, Y, Z, act, order=2, pad=2):
    """One field solve per material region, each at unit susceptibility,
    shimmed and restricted to the active volume.

    The method is linear in chi, so the residual of ANY material
    combination is a weighted sum of these. Changing a material therefore
    costs no new FFT, and the best-matched susceptibility for any one
    region follows in closed form.

    Returns {name: (residual_vector, chi_now_ppm)}.
    """
    _, _, _, regions = build_tube(x, y, z, return_regions=True, **kw)
    out = {}
    for name, (mask, chi_now) in regions.items():
        if not mask.any():
            continue
        d = field_shift_ppm(mask.astype(float), voxel, pad=pad)
        out[name] = (shim(d, X, Y, Z, act, order=order), chi_now)
    return out


def breakdown(basis):
    """How much each region contributes to the residual, in ppm rms, and
    the total. This is the first thing to look at: matching a material
    can only ever help with the region that dominates."""
    tot = None
    rows = []
    for name, (r, chi) in basis.items():
        v = chi * r
        tot = v if tot is None else tot + v
        rows.append((name, chi, float(v.std())))
    rows.sort(key=lambda t: -t[2])
    return rows, float(tot.std())


def match_region(basis, target, order=2):
    """The susceptibility of one region that minimises the total residual,
    in closed form.

    This is what a Shigemi tube does: match the glass (or the plug) to the
    solvent so the two cancel. It only pays off when that region actually
    dominates, which `breakdown` tells you.

    Returns (chi_best_ppm, residual_there, residual_now).
    """
    if target not in basis:
        return float("nan"), float("nan"), float("nan")
    rT, chi_now = basis[target]
    rest = None
    for name, (r, chi) in basis.items():
        if name == target:
            continue
        v = chi * r
        rest = v if rest is None else rest + v
    if rest is None:
        rest = np.zeros_like(rT)
    denom = float(rT @ rT)
    if denom <= 0:
        return float("nan"), float("nan"), float((rest).std())
    chi_best = -float(rT @ rest) / denom
    return (chi_best,
            float((chi_best * rT + rest).std()),
            float((chi_now * rT + rest).std()))


def residual_with(basis, overrides=None):
    """Residual rms with some regions' susceptibilities replaced."""
    overrides = overrides or {}
    tot = None
    for name, (r, chi) in basis.items():
        v = overrides.get(name, chi) * r
        tot = v if tot is None else tot + v
    return float(tot.std())


def nearest_materials(chi_target, k=3, exclude=("vacuum",)):
    """Real materials closest to a target susceptibility."""
    items = [(m, v) for m, v in MATERIALS_PPM.items() if m not in exclude]
    items.sort(key=lambda t: abs(t[1] - chi_target))
    return items[:k]
