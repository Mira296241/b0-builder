"""Validation against analytic results.

Only field DIFFERENCES are observable: a uniform offset is absorbed by the
field/frequency lock, and the FFT method fixes the box mean arbitrarily.
Every test compares inside minus far-outside.

Ellipsoid of uniform chi, demagnetising factor N along B0:
        dB_in - dB_out = chi * (1/3 - N)
    sphere          N=1/3 ->  0
    cylinder || B0  N=0   ->  chi/3
    cylinder _|_ B0 N=1/2 -> -chi/6
Outside a sphere of radius a:  dB/B0 = (chi/3)(a/r)^3 (3cos^2 th - 1)
"""
import numpy as np
from b0core import field_shift_ppm

CHI = -10.0

def grid3(n, V):
    nx, ny, nz = n
    ax = (np.arange(nx)-(nx-1)/2)*V
    ay = (np.arange(ny)-(ny-1)/2)*V
    az = (np.arange(nz)-(nz-1)/2)*V
    return np.meshgrid(ax, ay, az, indexing="ij")

def run(name, V, n, mk, ins_f, out_f, target):
    X, Y, Z = grid3(n, V)
    dB = field_shift_ppm(np.where(mk(X,Y,Z), CHI, 0.0), (V,V,V), pad=2)
    ins, out = ins_f(X,Y,Z), out_f(X,Y,Z)
    val = dB[ins].mean() - dB[out].mean()
    err = abs(val-target)/abs(target)*100 if target else abs(val)
    print(f"{name:<20}{target:>11.4f}{val:>11.4f}{dB[ins].std():>12.5f}{err:>10.2f}")

print(f"{'case':<20}{'analytic':>11}{'computed':>11}{'uniformity':>12}{'error %':>10}")
print("-"*64)

a = 4.0
run("sphere", 0.2, (96,96,96), lambda X,Y,Z: X**2+Y**2+Z**2 <= a*a,
    lambda X,Y,Z: X**2+Y**2+Z**2 <= (0.6*a)**2,
    lambda X,Y,Z: (X**2+Y**2+Z**2 > (3.0*a)**2) & (X**2+Y**2+Z**2 < (3.6*a)**2),
    0.0)

a = 2.0
run("cylinder || B0", 0.25, (64,64,600), lambda X,Y,Z: X**2+Y**2 <= a*a,
    lambda X,Y,Z: (X**2+Y**2 <= (0.5*a)**2) & (np.abs(Z) < 10),
    lambda X,Y,Z: (X**2+Y**2 > (3.0*a)**2) & (np.abs(Z) < 10),
    CHI/3)

run("cylinder _|_ B0", 0.25, (600,64,64), lambda X,Y,Z: Y**2+Z**2 <= a*a,
    lambda X,Y,Z: (Y**2+Z**2 <= (0.5*a)**2) & (np.abs(X) < 10),
    lambda X,Y,Z: (Y**2+Z**2 > (3.0*a)**2) & (np.abs(X) < 10),
    -CHI/6)

print()
a, V = 4.0, 0.2
X, Y, Z = grid3((96,96,96), V)
dB = field_shift_ppm(np.where(X**2+Y**2+Z**2 <= a*a, CHI, 0.0), (V,V,V), pad=2)
off = dB[X**2+Y**2+Z**2 > (3.0*a)**2].mean()
print(f"{'sphere exterior (equator)':<28}{'analytic':>11}{'computed':>11}")
print("-"*50)
for r in (5.0, 6.0, 7.0, 8.0):
    sel = (np.abs(Z) < V) & (np.abs(Y) < V) & (np.abs(np.abs(X)-r) < V)
    print(f"  r = {r:4.1f} mm{'':<13}{(CHI/3)*(a/r)**3*(-1.0):>11.4f}"
          f"{dB[sel].mean()-off:>11.4f}")

# ---------------------------------------------------------------------------
# Two more exact cases: an infinite slab, both orientations.
# Demagnetising factor N = 1 across the slab, 0 along it.
#   perpendicular to B0 (thin in z):  chi * (1/3 - 1) = -2 chi / 3
#   parallel to B0      (thin in x):  chi * (1/3 - 0) =  chi / 3
# Anisotropic voxels: fine across the slab, coarse along it, since the
# field is flat in the wide directions.
# ---------------------------------------------------------------------------

def _ax(half, n):
    return (np.arange(n) - (n-1)/2) * (2*half/n), 2*half/n

print()
print(f"{'infinite slab':<28}{'analytic':>11}{'computed':>11}{'error %':>10}")
print("-" * 60)

at, vt = _ax(3.0, 300)       # thin direction
aw, vw = _ax(20.0, 64)       # wide directions

for name, thin_axis, target in (("perpendicular to B0", 2, CHI*(1/3 - 1)),
                                ("parallel to B0",      0, CHI/3)):
    if thin_axis == 2:
        X, Y, Z = np.meshgrid(aw, aw, at, indexing="ij"); T, V3 = Z, (vw, vw, vt)
        W1, W2 = X, Y
    else:
        X, Y, Z = np.meshgrid(at, aw, aw, indexing="ij"); T, V3 = X, (vt, vw, vw)
        W1, W2 = Y, Z
    dB = field_shift_ppm(np.where(np.abs(T) <= 0.25, CHI, 0.0), V3, pad=2)
    wide = (np.abs(W1) <= 4) & (np.abs(W2) <= 4)
    val = dB[(np.abs(T) <= 0.1) & wide].mean() - dB[(np.abs(T) >= 2.0) & wide].mean()
    print(f"{name:<28}{target:>11.4f}{val:>11.4f}"
          f"{abs(val-target)/abs(target)*100:>10.2f}")

# ---------------------------------------------------------------------------
# Finite cylinder, against an EXACT result: Hoffman, J Magn Reson 178 (2006)
# 237, Eq. 14, extended along the axis. On the axis of a vertical cylinder
# of length h and radius R the shape factor (demagnetising factor) is
#     N(z) = 1 - [ a/sqrt(R^2+a^2) + b/sqrt(R^2+b^2) ] / 2,  a,b = h/2 -+ z
# Inside, dB/B0 = chi (1/3 - N) + const, so the field DIFFERENCE between
# two points needs no reference:  dB(z) - dB(0) = -chi [N(z) - N(0)].
#
# This is the important test. Every case above is an infinite limit; a real
# sample is finite, and the variation along it is exactly what broadens the
# line. The absolute shape factor cannot be tested this way, because the
# method discards the uniform offset, as the spectrometer lock does.
#
# The box must be wide (6R here): the FFT is periodic, so copies of the
# cylinder sit side by side and their end fields leak onto the axis. The app
# is not affected, because shimming removes that smooth background.
# ---------------------------------------------------------------------------

def _N_axis(z, h, R):
    a, b = h/2 - z, h/2 + z
    return 1 - 0.5*(a/np.sqrt(R**2 + a**2) + b/np.sqrt(R**2 + b**2))

def _axis_profile(h, R, xyh, vx, vz):
    ax, _ = _ax(xyh, int(round(2*xyh/vx)))
    az, _ = _ax(0.9*h, int(round(1.8*h/vz)))
    X, Y, Z = np.meshgrid(ax, ax, az, indexing="ij")
    m = (np.hypot(X, Y) <= R) & (np.abs(Z) <= h/2)
    dB = field_shift_ppm(m.astype(float), (vx, vx, vz), pad=2)
    i0 = len(ax)//2
    line = dB[i0-1:i0+1, i0-1:i0+1, :].mean(axis=(0, 1))
    return az, line - np.interp(0.0, az, line)

print()
print("finite cylinder, field along the axis relative to the centre,")
print("per unit chi, against Hoffman 2006 Eq. 14 (exact)")
for h, R in ((4.0, 1.0), (20.0, 1.0)):
    az, ln = _axis_profile(h, R, 6.0*R, 0.125, 0.10 if h < 10 else 0.2)
    print(f"\n  h/2R = {h/2/R:.0f}")
    print(f"  {'z/(h/2)':>8}{'exact':>11}{'computed':>11}{'diff':>10}")
    for f in (0.5, 0.8, 0.9):
        z = f*h/2
        ex = -(_N_axis(z, h, R) - _N_axis(0.0, h, R))
        got = float(np.interp(z, az, ln))
        print(f"  {f:>8.2f}{ex:>11.5f}{got:>11.5f}{got-ex:>10.5f}")
