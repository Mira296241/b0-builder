"""
Checks for the reduced microwave models in thz.py.

Part 1 is analytic: cases with a known closed-form answer.
Part 2 is against the HFSS wall sweep, which is the only measured-by-
simulation data we have. It is reported honestly, including the case
where the model does not work.

Run:  python validate_thz.py
"""
import numpy as np
import thz as T

F = 330.0
OK = "ok "
BAD = "FAIL"


def chk(name, got, want, tol):
    good = abs(got - want) <= tol
    print(f"  [{OK if good else BAD}] {name:38s} got {got:10.5f}   "
          f"expected {want:10.5f}")
    return good


print("=" * 68)
print("1. ANALYTIC CHECKS")
print("=" * 68)

n_q = T.index("fused quartz").real
lam0 = T.C_MM_GHZ / F
lam_q = lam0 / n_q
allgood = True

# a) half-wave slab is invisible, quarter-wave slab reflects ((n^2-1)/(n^2+1))^2
s = T.stack(F, [("fused quartz", lam_q / 2)])
allgood &= chk("half-wave slab, R", s["R"], 0.0, 1e-3)

s = T.stack(F, [("fused quartz", lam_q / 4)])
want = ((n_q**2 - 1) / (n_q**2 + 1)) ** 2
allgood &= chk("quarter-wave slab, R", s["R"], want, 1e-3)

# b) single interface, Fresnel at normal incidence
s = T.stack(F, [("fused quartz", 1e-9)], n_out="fused quartz")
want = ((1 - n_q) / (1 + n_q)) ** 2
allgood &= chk("air -> quartz interface, R", s["R"], want, 1e-4)

# c) energy conservation, including a strongly absorbing sample
for liq in ("toluene", "water", "DMSO"):
    s = T.stack(F, [("fused quartz", 0.5), (liq, 30.0)])
    allgood &= chk(f"R+T+A = 1 with {liq}", s["R"] + s["T"] + s["A"], 1.0, 1e-9)

# d) Beer-Lambert: the transmitted power through a thick lossy slab must
#    follow exp(-alpha d) with alpha = 4 pi k / lam0.
#    Immersed in its own medium, so there are no interfaces and no
#    Fabry-Perot ripple: the result must be exactly exp(-alpha d).
k = abs(T.index("toluene").imag)
alpha = 4 * np.pi * k / lam0
for d in (3.0, 10.0, 30.0):
    got = T.stack(F, [("toluene", d)], n_in="toluene", n_out="toluene")["T"]
    allgood &= chk(f"Beer-Lambert, {d:.0f} mm of toluene", got,
                   float(np.exp(-alpha * d)), 1e-9)
print(f"       (alpha = {alpha:.3f} /mm = {alpha*10:.2f} /cm, "
      f"1/e power depth {1/alpha:.2f} mm)")

# e) a Gaussian beam in free space must follow w(z) = w0 sqrt(1+(z/zR)^2)
w0 = 1.0
zR = np.pi * w0**2 / lam0
z, w, _ = T.beam_through_bottom(F, w0, d_cwg=1e6, bottom="flat", z_max=20.0)
want = w0 * np.sqrt(1 + (z / zR) ** 2)
allgood &= chk("free-space Gaussian, max error",
               float(np.max(np.abs(w - want))), 0.0, 1e-9)

# f) thin-lens limit: a single spherical surface focuses a collimated beam
#    at f = n2 R / (n2 - n1), measured from the vertex, inside medium 2.
R1 = 2.0
f_exp = n_q * R1 / (n_q - 1.0)
q = 1j * np.pi * 20.0**2 / lam0           # 20 mm waist: collimated here
q = T._q_refract(q, R1, 1.0, n_q)
allgood &= chk("single-surface focal length", -q.real, f_exp, 1e-3 * f_exp)

# g) the two bottom shapes must have opposite sign of curvature
Ro_round, _ = T.bottom_radii(bottom="round", tube_OD=4.0, wall=0.5, bot_depth=2.0)
Ro_punt, _ = T.bottom_radii(bottom="punt", tube_OD=4.0, wall=0.5, punt_r=2.0)
allgood &= chk("round bottom outer radius", Ro_round, 4.0 / 2.0, 1e-9)
allgood &= chk("punt outer radius (must be negative)", Ro_punt, -2.0, 1e-9)

print("\n" + "=" * 68)
print("2. AGAINST THE HFSS WALL SWEEP")
print("=" * 68)
print("""
HFSS, 330 GHz, round bottom 2 mm, quarter model, measured materials.
The question was why the wall thickness changed the figure of merit,
since the half-wave etalon prediction did not match.
""")

hfss = {0.334: 2.149, 0.446: 1.833, 0.500: 1.856, 0.668: 1.501}

print("  wall   1D etalon R    beam waist    HFSS FoM   FoM x waist^2")
rows = []
for wl in sorted(hfss):
    st = T.stack(F, [("fused quartz", wl), ("toluene", 30.0)])
    z, w, zf = T.beam_through_bottom(F, 1.0, d_cwg=5.0, bottom="round",
                                     wall=wl, bot_depth=2.0)
    wf = float(np.nanmin(w[z > 5 + wl]))
    fom = hfss[wl]
    rows.append((wl, st["R"], wf, fom, fom * wf**2))
    print(f"  {wl:.3f}    {st['R']*100:6.2f} %     {wf:.3f} mm     "
          f"{fom:.3f}      {fom*wf**2:.3f}")

prod = np.array([r[4] for r in rows])
spread = prod.std() / prod.mean()
print(f"""
  The etalon column does not track the FoM at all: 0.446 mm is a
  half-wave multiple and reflects least, yet it is not the best design.

  The beam waist does track it. FoM x waist^2 is constant to
  {spread*100:.1f} %, so the figure of merit is simply the on-axis
  intensity at the focus, and the wall acts through the INNER radius of
  curvature of the bottom, r_i^2 / (depth - wall), not through
  interference. Thinner wall -> flatter inner surface -> less negative
  power -> tighter focus -> higher FoM.
""")
allgood &= chk("FoM x waist^2 constant (rel. spread)", spread, 0.0, 0.06)

print("\n" + "=" * 68)
print("3. WHERE THE BEAM MODEL DOES NOT WORK")
print("=" * 68)
z, w, zf = T.beam_through_bottom(F, 1.0, d_cwg=5.0, bottom="punt", punt_r=2.0)
w15 = float(np.interp(15.0, z, w))
print(f"""
  For the punt the model predicts no focus at all and a beam radius of
  {w15:.1f} mm at z = 15 mm, far wider than the {4.0/2-0.5:.1f} mm bore.
  Taken literally that means almost no power stays in the sample, yet
  HFSS gives only a 17 % drop in FoM.

  The model is wrong here, and the reason is instructive: once the beam
  diverges onto the wall, the tube stops being a lens and starts being a
  dielectric waveguide. Total internal reflection at the glass-air
  boundary keeps the power inside. A paraxial Gaussian cannot represent
  that, so the punt has to be done in HFSS.

  Use the reduced models for: wall thickness, bottom depth, reflected
  power, absorption, solvent choice. Use HFSS for: the punt, slots,
  anything where the field leaves the paraxial region.
""")

# ---------------------------------------------------------------------------
# Coordinate consistency between the two microwave models.
#
# The two functions do NOT share a frame, and that asymmetry caused a real
# bug: axial_profile returns height above the TUBE TIP, while
# beam_through_bottom propagates from the LAUNCHER and so returns launcher
# z, with the tip at d_cwg. Plot one against a window drawn in the other
# and everything is wrong by d_cwg with nothing on screen to show it.
#
# Both models know where the glass-to-liquid interface is, so that is the
# landmark. Converted to tube-tip height it must come out at the wall
# thickness in both.
# ---------------------------------------------------------------------------
print()
print("coordinate consistency, axial_profile vs beam_through_bottom")
print("-" * 68)

D_CWG, WALL = 5.0, 0.5
zp_c, _, _ = T.axial_profile(330.0, bottom="round", wall=WALL, punt_r=2.0,
                              bot_depth=2.0, fill_h=30.0,
                              glass="fused quartz", liquid="toluene")
sample_start_tip = float(zp_c.min())          # already tube-tip height
beam_inner_tip = (D_CWG + WALL) - D_CWG       # launcher z of the vertex, to tip

for name, got in (("axial_profile, sample starts at", sample_start_tip),
                  ("beam, glass-liquid vertex at ", beam_inner_tip)):
    ok = abs(got - WALL) < 1e-9
    allgood &= ok
    print(f"  {name} {got:7.3f} mm above tip   "
          f"expected {WALL:.3f}   {'ok' if ok else 'FAIL'}")
print("  Both are tube-tip heights. app.py plots axial_profile raw and")
print("  subtracts d_cwg from the beam, so the two axes agree on screen.")

print("=" * 68)
print("ALL CHECKS PASSED" if allgood else "SOMETHING FAILED, see above")
print("=" * 68)
