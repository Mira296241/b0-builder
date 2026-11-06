"""Validation of the external rod against analytic results.

Same rule as validate.py: only field DIFFERENCES are observable, so every
test compares two points, never an absolute level.

The rod is the perturber of the Schonzart validation experiment
(J Magn Reson 364 (2024) 107704). Two exact limits bracket it.

1. Rod PERPENDICULAR to B0, long. Across the rod it is a 2D line dipole of
   moment pi a^2 M per unit length, so outside it

        dB/B0 = (chi/2) (a/rho)^2 cos 2phi,   phi measured from B0

   positive along B0 from the rod, negative beside it, falling as 1/rho^2.

2. Rod PARALLEL to B0, semi-infinite, flat end face. M lies along the rod,
   so div M vanishes everywhere except on that face: a uniformly charged
   disc, sigma = M. On the axis, a distance d beyond the face,

        dB/B0 = (chi/2) [ 1 - d / sqrt(d^2 + a^2) ]  ->  chi a^2 / (4 d^2)

   This is the regime the experiment works in, and it is why the rod
   HEIGHT was Schonzart's only fit parameter: the body of a vertical rod
   does nothing, the near end face does everything.

3. That claim, checked directly: the spread of the field over an active
   volume beside the middle of a long vertical rod, against the spread
   beside its end. Reference free, so no offset can hide in it.

NOTE ON BOX WIDTH, which test 2 measures. The FFT is periodic, so copies of
the rod sit beside the real one at the padded box pitch and their end
fields leak onto the axis. The tube itself does not suffer from this (its
field is weak and smooth and shimming removes what leaks), but a rod of
chi = +13 ppm sitting off axis does. Widen the box when the rod is on.
"""
import numpy as np
from b0core import field_shift_ppm, build_rod, MATERIALS_PPM

MAT = "aluminium 7075"
DCHI = MATERIALS_PPM[MAT] - MATERIALS_PPM["air"]


def ax(half, V):
    """voxel centres, odd count so that 0 is a centre and the rod end face
    at z_end=0 lands on a voxel boundary... to within half a voxel, which
    the partial-volume fill in build_rod then resolves."""
    n = int(round(2 * half / V)) | 1
    return (np.arange(n) - (n - 1) / 2) * V


def solve(x, y, z, **kw):
    chi, _ = build_rod(x, y, z, material=MAT, outside="air", subsample=4, **kw)
    return field_shift_ppm(chi, (x[1]-x[0], y[1]-y[0], z[1]-z[0]), pad=2)


# ---------------------------------------------------------------- test 1
print("1. rod perpendicular to B0, external field")
print("   dB = (chi/2)(a/rho)^2 cos 2phi,  chi = %+.2f ppm" % DCHI)
print("   referenced to the opposite azimuth, which doubles the signal\n")

V, a = 0.2, 1.0
x = z = ax(8.0, V)
y = ax(30.0, 0.5)
dB = solve(x, y, z, diameter=2*a, length=56.0, distance=0.0, z_end=0.0,
           vertical=False)
jy = y.size // 2
plane = dB[:, jy, :]                      # x-z plane through the rod centre


def pt(px, pz):
    return float(plane[int(np.argmin(abs(x-px))), int(np.argmin(abs(z-pz)))])


print(f"   {'rho (mm)':>9}{'exact':>11}{'computed':>11}{'err %':>9}")
for rho in (2.0, 3.0, 4.0, 5.0):
    ex = DCHI * (a / rho) ** 2            # along B0 minus across B0
    got = pt(0.0, rho) - pt(rho, 0.0)
    print(f"   {rho:>9.1f}{ex:>11.4f}{got:>11.4f}{abs(got-ex)/abs(ex)*100:>9.2f}")

# ---------------------------------------------------------------- test 2
print("\n2. rod parallel to B0, on the axis beyond its end face")
print("   dB(d) - dB(10 mm),  dB = (chi/2)[1 - d/sqrt(d^2+a^2)],  a = 1.5 mm")
print("   repeated for three box half-widths: the error is periodic images,")
print("   not the method, and it falls as the box grows\n")

V, a, L = 0.25, 1.5, 30.0


def axis_line(half, V, a, L):
    x = y = ax(half, V)
    z = np.arange(int(round((-L-4)/V)), int(round(12/V)) + 1) * V
    # the rod body runs DOWN from its face at z = 0, into the negative z
    # the grid provides; the test points d > 0 sit beyond that face.
    dB = solve(x, y, z, diameter=2*a, length=L, distance=0.0, z_end=0.0,
               vertical=True, extends="down")
    return z, dB[x.size//2, y.size//2, :]


def ana(d, a):
    return (DCHI / 2) * (1 - d / np.hypot(d, a))


ds = (1.0, 2.0, 3.0, 5.0, 8.0)
rows = {}
for half in (5.0, 8.0, 12.0):
    z, line = axis_line(half, V, a, L)
    rows[half] = [np.interp(d, z, line) - np.interp(10.0, z, line) for d in ds]

print(f"   {'d (mm)':>8}{'exact':>10}" + "".join(f"{'half=%g'%h:>12}" for h in rows))
for i, d in enumerate(ds):
    ex = ana(d, a) - ana(10.0, a)
    cells = "".join(f"{rows[h][i]:>12.4f}" for h in rows)
    print(f"   {d:>8.1f}{ex:>10.4f}{cells}")
print(f"   {'err %':>8}{'':>10}" + "".join(
    f"{np.mean([abs(rows[h][i]-(ana(d,a)-ana(10.,a)))/abs(ana(d,a)-ana(10.,a))*100 for i,d in enumerate(ds)]):>12.2f}"
    for h in rows))

# ---------------------------------------------------------------- test 3
print("\n3. the body of a vertical rod does nothing, only its end does")
print("   spread of dB over a 15 mm x 1.5 mm active volume beside a 90 mm rod\n")

V, a = 0.3, 1.5
x = y = ax(12.0, V)
z = ax(60.0, V)
dB = solve(x, y, z, diameter=2*a, length=90.0, distance=4.0, z_end=45.0,
           vertical=True)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
R = np.hypot(X, Y)
for label, zc in (("beside the middle of the rod", 0.0),
                  ("straddling the top end face", 45.0)):
    m = (R <= 1.5) & (np.abs(Z - zc) <= 7.5)
    v = dB[m]
    print(f"   {label:<30}{'p-p':>6} {np.ptp(v):8.4f} ppm"
          f"   rms {v.std():8.4f} ppm")

# ---------------------------------------------------------------- test 4
print("\n4. voxel convergence, test 2 at d = 2 mm, box half-width 10 mm")
a, L = 1.5, 30.0
ex = ana(2.0, a) - ana(10.0, a)
print(f"   {'voxel':>9}{'computed':>11}{'err %':>9}")
for V in (0.4, 0.3, 0.25, 0.2):
    z, line = axis_line(10.0, V, a, L)
    got = np.interp(2.0, z, line) - np.interp(10.0, z, line)
    print(f"   {V:>9.2f}{got:>11.4f}{abs(got-ex)/abs(ex)*100:>9.2f}")

# ---------------------------------------------------------------- test 5
print("\n5. the two mounting directions are mirror images")
print("   same near-face distance, rod body above vs below. The near face")
print("   is the only source, so both must give the same magnitude.\n")

V, a, L = 0.3, 1.5, 50.0
x = y = ax(12.0, V)
z = ax(60.0, V)


def line(ext):
    chi, _ = build_rod(x, y, z, material=MAT, diameter=2*a, length=L,
                       distance=0.0, z_end=0.0, vertical=True, extends=ext,
                       outside="air", subsample=4)
    dB = field_shift_ppm(chi, (V, V, V), pad=2)
    return dB[x.size//2, y.size//2, :]


up, dn = line("up"), line("down")
print(f"   {'d (mm)':>8}{'exact':>10}{'tip up':>10}{'tip down':>10}")
for d in (1.5, 3.0, 5.0):
    ex = ana(d, a) - ana(d + 20, a)
    ku = np.interp(-d, z, up) - np.interp(-d - 20, z, up)
    kd = np.interp(d, z, dn) - np.interp(d + 20, z, dn)
    print(f"   {d:>8.1f}{ex:>10.4f}{ku:>10.4f}{kd:>10.4f}")
