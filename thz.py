"""
Reduced microwave models for the sample tube at THz-EPR / DNP frequencies.

Two things are computed, each exact in its own limit:

1. STACK  (1D transfer matrix, normal incidence)
   The beam travels up the tube axis through air / bottom glass / sample.
   Gives the reflected power, the standing wave, and |H|^2 in the sample.
   Exact for an infinitely wide beam and flat interfaces.
   Does NOT know about curvature, focusing or diffraction.

2. BEAM   (Gaussian beam through ABCD matrices)
   Gives where the bottom focuses the beam and how wide it is there.
   Exact in the paraxial limit for a smooth surface.
   Does NOT know about reflections, interference or standing waves.

Neither replaces a full-wave solver. Together they scan thousands of
designs per second so that only the survivors are worth an HFSS run.

Sign convention for a surface radius R: positive when the centre of
curvature lies DOWNSTREAM (on the far side from the incoming beam).
A tube bottom bulging down towards the beam therefore has R > 0 and
converges; a punt carved upwards has R < 0 and diverges.
"""

import numpy as np

C_MM_GHZ = 299.792458          # c in mm*GHz

# Relative permittivity and loss tangent near 330 GHz.
# quartz and toluene are set to match the wavelengths used in the HFSS
# model (lambda_quartz = 0.446 mm, lambda_toluene = 0.58 mm at 330 GHz).
# REPLACE with the measured values before publishing anything.
MW_MATERIALS = {
    "vacuum":        (1.000, 0.0),
    "air":           (1.000, 0.0),
    "fused quartz":  (4.150, 1.0e-4),
    "borosilicate":  (4.600, 7.0e-3),
    "sapphire":      (9.400, 1.0e-5),
    "alumina":       (9.700, 3.0e-4),
    "zirconia":     (23.000, 2.0e-3),
    "PTFE":          (2.060, 2.0e-4),
    "PEEK":          (3.200, 5.0e-3),
    "Vespel":        (3.000, 6.0e-3),
    "toluene":       (2.450, 1.2e-2),
    "water":         (4.500, 5.6e-1),     # very lossy at 330 GHz
    "glycerol/water 60:40": (5.200, 4.0e-1),
    "DMSO":          (4.000, 2.0e-1),
    "ethanol":       (3.000, 3.0e-1),
    "methanol":      (3.300, 3.5e-1),
}
# metals are handled as a perfect mirror, not as a dielectric
METALS = {"copper", "silver"}


def index(name):
    """Complex refractive index, n - i*k, from eps' and tan(delta)."""
    e, td = MW_MATERIALS.get(name, (1.0, 0.0))
    eps = e * (1 - 1j * td)
    return np.sqrt(eps)


# --------------------------------------------------------------- 1. stack

def stack(f_GHz, layers, n_in="air", n_out="air"):
    """Transfer matrix for normal incidence.

    layers: list of (material_name, thickness_mm), in the order the beam
            meets them. Semi-infinite media on both sides.

    Returns dict with r, t, R (reflected power), T (transmitted power),
    A (absorbed), and the per-layer forward/backward amplitudes needed
    to reconstruct the field.
    """
    lam = C_MM_GHZ / f_GHz
    ns = [index(n_in)] + [index(m) for m, _ in layers] + [index(n_out)]
    ds = [0.0] + [d for _, d in layers] + [0.0]

    def I(a, b):                       # interface a -> b
        r = (ns[a] - ns[b]) / (ns[a] + ns[b])
        t = 2 * ns[a] / (ns[a] + ns[b])
        return np.array([[1, r], [r, 1]], dtype=complex) / t

    def L(j):
        # Maps the amplitudes at the END of layer j back to its START,
        # which is the direction the product below runs in. Getting this
        # the wrong way round is invisible without loss (the two differ
        # only by a conjugation) and gives R > 1 the moment the sample
        # absorbs anything.
        b = 2 * np.pi * ns[j] * ds[j] / lam
        return np.array([[np.exp(1j * b), 0], [0, np.exp(-1j * b)]])

    S = I(0, 1)
    for j in range(1, len(ns) - 1):
        S = S @ L(j) @ I(j, j + 1)

    r = S[1, 0] / S[0, 0]
    t = 1.0 / S[0, 0]

    # amplitudes at the START of each layer, normalised to unit incidence
    amps = [np.array([1.0 + 0j, r])]
    M = I(0, 1)
    amps.append(np.linalg.solve(M, amps[0]))
    for j in range(1, len(ns) - 1):
        M = L(j) @ I(j, j + 1)
        amps.append(np.linalg.solve(M, amps[-1]))

    Rp = abs(r) ** 2
    Tp = float((ns[-1].real / ns[0].real) * abs(t) ** 2)
    return dict(r=r, t=t, R=Rp, T=Tp, A=max(0.0, 1 - Rp - Tp),
                ns=ns, ds=ds, amps=amps, lam=lam)


def stack_fields(sol, j, zrel):
    """|E|^2 and |H|^2 inside layer j, at depth zrel (mm) from its start.

    The two interfere with OPPOSITE sign: a node of E is an antinode of H.
    B1 is magnetic, so |H|^2 is the one that matters for EPR and DNP.
    """
    n = sol["ns"][j]
    A, B = sol["amps"][j]
    b = 2 * np.pi * n / sol["lam"]
    f, g = A * np.exp(-1j * b * zrel), B * np.exp(1j * b * zrel)
    E = f + g
    H = n * (f - g)
    return np.abs(E) ** 2, np.abs(H) ** 2


def axial_layers(*, bottom="round", wall=0.5, punt_r=2.0, bot_depth=2.0,
                 fill_h=30.0, glass="fused quartz", liquid="toluene"):
    """The material stack ALONG THE AXIS, from the beam upwards, plus the
    z of the first interface relative to the tube tip.

    On the axis both the round bottom and the punt present the same glass
    path: one wall thickness. The whole difference between them is
    curvature, which is the beam model, not this one. That is worth
    knowing, and it is why the wall sweep and the bottom sweep are
    separate questions.
    """
    if bottom == "flat":
        t_glass, z0 = wall, 0.0
    elif bottom == "round":
        t_glass, z0 = wall, 0.0
    else:                                   # punt: air pocket first
        t_glass, z0 = wall, punt_r
    return [(glass, t_glass), (liquid, fill_h)], z0


def axial_profile(f_GHz, *, n_pts=4000, **kw):
    """|H|^2 along the axis inside the sample, and the reflected power.

    Returns (z_mm_from_tube_tip, H2, solution_dict).
    """
    layers, z0 = axial_layers(**kw)
    sol = stack(f_GHz, layers, n_in="air", n_out="air")
    fill = layers[1][1]
    zr = np.linspace(0, fill, n_pts)
    _, H2 = stack_fields(sol, 2, zr)          # layer 2 = the sample
    z = z0 + layers[0][1] + zr
    return z, H2, sol


# ---------------------------------------------------------------- 2. beam

def bottom_radii(*, bottom="round", tube_OD=4.0, wall=0.5,
                 punt_r=2.0, bot_depth=2.0):
    """On-axis radii of curvature of the two bottom surfaces, (outer, inner).

    The round bottom is an ellipsoid of semi-axes (ro, bot_depth), whose
    on-axis radius of curvature is ro^2 / bot_depth. The punt is a sphere,
    carved the other way, so both radii are -punt_r.
    """
    ro, ri = tube_OD / 2, tube_OD / 2 - wall
    if bottom == "flat":
        return np.inf, np.inf
    if bottom == "round":
        bi = max(bot_depth - wall, 1e-6)
        return ro**2 / bot_depth, ri**2 / bi
    return -punt_r, -punt_r


def _q_prop(q, d):
    """Free space over a geometric distance d. The medium enters only
    through the width formula, not here."""
    return q + d


def _q_refract(q, R, n1, n2):
    """Spherical dielectric interface, ABCD = [[1, 0], [(n1-n2)/(n2 R), n1/n2]].
    R = inf is a flat interface, which still changes q because the index
    changes."""
    C = 0.0 if not np.isfinite(R) else (n1 - n2) / (n2 * R)
    D = n1 / n2
    return q / (C * q + D)


def _width(q, n, lam):
    """Beam radius from q in a medium of index n."""
    im = np.imag(1.0 / q)
    if im >= 0:
        return np.nan
    return float(np.sqrt(-lam / (np.pi * n * im)))


def beam_through_bottom(f_GHz, w0_mm, *, d_cwg=5.0, bottom="round",
                        tube_OD=4.0, wall=0.5, punt_r=2.0, bot_depth=2.0,
                        glass="fused quartz", liquid="toluene",
                        z_max=40.0, n_pts=800):
    """Gaussian beam from a waist w0 at z=0 (the launcher aperture), up
    through the tube bottom and into the sample.

    Returns (z, w_mm, z_focus). z_focus is where the beam is narrowest
    inside the sample, or nan if it never forms a waist there.
    """
    lam = C_MM_GHZ / f_GHz
    n_g, n_l = index(glass).real, index(liquid).real
    R1, R2 = bottom_radii(bottom=bottom, tube_OD=tube_OD, wall=wall,
                          punt_r=punt_r, bot_depth=bot_depth)

    # vertex positions on the axis
    if bottom == "punt":
        z1 = d_cwg + punt_r                 # air pocket, then glass
    else:
        z1 = d_cwg
    z2 = z1 + wall

    q0 = 1j * np.pi * w0_mm**2 / lam        # q at the waist, in air

    z = np.linspace(0, z_max, n_pts)
    w = np.empty_like(z)
    for i, zi in enumerate(z):
        q, n = q0, 1.0
        if zi <= z1:
            q = _q_prop(q, zi)
        else:
            q = _q_prop(q, z1)
            q = _q_refract(q, R1, 1.0, n_g); n = n_g
            if zi <= z2:
                q = _q_prop(q, zi - z1)
            else:
                q = _q_prop(q, wall)
                q = _q_refract(q, R2, n_g, n_l); n = n_l
                q = _q_prop(q, zi - z2)
        w[i] = _width(q, n, lam)

    inside = z > z2
    zf = np.nan
    if inside.any():
        wi = w[inside]
        k = int(np.argmin(wi))
        if 0 < k < len(wi) - 1:             # a real interior minimum
            zf = float(z[inside][k])
    return z, w, zf
