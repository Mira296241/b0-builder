# B0 tube explorer

Susceptibility-induced B0 distortion in NMR / DNP sample tubes.

There are two apps. Both use the same physics (`b0core.py`).

| app | for whom | run it with |
|---|---|---|
| **builder.py** (new, simple) | anyone: start from a tube, add only the parts your probe has | `python -m streamlit run builder.py` |
| **app.py** (full) | us: sweeps, recommendations, microwave models, susceptibility matching | `python -m streamlit run app.py` |

## The builder in 30 seconds

1. **Start from** a preset in the sidebar: *Blank* (tube only), *Standard NMR probe*, or *FRASCAN II* (our probe, from the CAD).
2. **Sample tube**: diameter, wall, liquid height, tube tip height, materials. "more tube options" hides the bottom shape, meniscus and surroundings.
3. **Probe parts**: switch on only what your probe has (RF coil, waveguide, RF shields, modulation coil). Each one opens its own settings.
4. **Add a custom part** for anything else: a cylinder, ring or box of any material (holder, spacer, PTFE lens...).
5. Press **Run**. You get the linewidth, the field map and the NMR line you would measure. Detection region, shimming and frequency update live, without pressing Run again.
6. **Save this setup** writes a small .json file. Send it to a colleague; they load it with **Load a setup**.

All heights are in mm from **z = 0**, the probe reference (the waveguide top, if there is one). Move the sample with "tube tip height".

### New in November 2026 (after Honza's feedback)

- **A / B comparison.** Sidebar toggle "Compare two setups". B starts as a copy of A; change one thing (waveguide off, another tube, another position) and Run. You get both previews on one scale, a table with A, B and the difference, both NMR lines on one plot and both field maps on one colour scale.
- **Shimming up to 4th order** (was 2nd). The shim functions are the real solid harmonics a shim set produces (25 of them up to 4th order). For orders 0 to 2 the result is identical to before. "More results" shows the linewidth for every order, which is the honest error bar: a real shim set goes to 4th order and beyond along z but is rarely set perfectly.
- **Compare with a measured spectrum.** Upload two columns (frequency in Hz or ppm, intensity); it is centred, scaled and drawn over the simulated line, with its FWHM.
- **"Linewidth without the probe"** (was "natural linewidth"). Put here the linewidth measured on the same magnet with a commercial probe: that covers the sample's own width and whatever the magnet leaves after shimming. The tool adds this probe's broadening on top.
- **Check the solver** (sidebar, Accuracy). Solves a uniform sphere, which has an exact answer, at the current voxel size: about 0.5 % error.
- **Half the memory, twice the speed**: the FFT now runs in single precision (differs from double by 3e-6 ppm on the full probe).

### Second round (9 Nov 2026)

The right side now has four tabs: **Result, Sweep, 3D view, About**.

- **More presets**: standard 5 mm tube, 3 mm tube, Shigemi tube, 4 mm quartz tube, FRASCAN II. No MAS rotor on purpose: spinning averages these effects, so a static calculation would mislead.
- **Sweep tab**: vary one parameter (tube height, wall, waveguide position...) and plot the linewidth against it. Table and figure can be downloaded.
- **3D view tab**: the probe in 3D, with a cut-away. Needs `plotly`.
- **About tab**: what the tool does, how, what it cannot do, how far to trust it, sources. Set `AUTHOR` and `CITE` at the top of builder.py to show them there.
- **Downloads**: figure as PNG or PDF, numbers and the NMR line as CSV.
- **Susceptibility plugs** (Shigemi style) below and above the sample, in any material or "matched to the sample". Extra plastics from Wapler 2014 (PMMA, Kapton, polystyrene, polycarbonate...).
- **Sample temperature**: the liquid's susceptibility follows its density. Only valid while the sample is liquid.
- **Observed nucleus**: ¹H, ¹⁹F, ³¹P, ¹³C, ²H, ¹⁵N. Linewidths in Hz scale with the frequency.
- **Smooth coil edges**: weights the sample by a realistic coil sensitivity instead of a sharp box. Off by default so earlier numbers stay comparable.
- **Waveguide top height**: the waveguide can be lowered away from the tube.
- **Share by link**: "Create a link to this setup" packs the setup into the web address.
- **30-second tour** on first open, and a **feedback** link (`FEEDBACK_URL` at the top of builder.py).

### Memory and free servers (10 Nov 2026)

The solver used to allow a fixed 72 million padded FFT cells, which peaks at about 1.3 GB. A free Streamlit Community Cloud container has 1 GB, so the FRASCAN II preset throttled or crashed it.

The limit now follows the machine. The solver's peak is about 18 bytes per padded cell (measured on the full probe at several voxel sizes), it reads the container's cgroup limit or the physical RAM, and spends at most 55 % of it on the FFT. On a 1 GB server the FRASCAN II preset drops to a 0.35 mm voxel (444 MB peak, FWHM 20.0 Hz instead of 18.7); on a 2 GB machine it keeps 0.25 mm. The chosen voxel and the reason are shown in the app. Set `B0_MEMORY_MB` to override.

Sweeps are capped at 7 points on servers under 1.8 GB, because each point is a full solve and free hosts throttle anything that runs flat out for minutes.

### FRASCAN II from the real CAD (9 Nov 2026, evening)

The FRASCAN II preset is now read from the STEP export of the Fusion assembly (NMR_Probe_v617), not from hand measurements.

- **Corrected**: tube 4.98 mm OD with 0.39 mm wall (was 0.5), tube tip 1.0 mm above the waveguide (was 0.5), 34 mm of sample (was 45), coil windows 90° wide facing ±y (was 88°).
- **Added, they were missing**: the borosilicate glass former under the RF coil, the coil ground ring, the PEEK modulation coil spool, the SLA lid, the PEEK electronics support, and the two chip capacitors at the coil slits.
- **Extra parts panel**: parts have names, parts with the same name are switched on and off together ("included in the calculation"), and you pick one to edit.
- **Two parts are listed but OFF by default.** The *lid*, because it sits 15 mm above the detected sample and moves the line by 0.3 Hz out of 19, so it is only there to show that it does not matter. The *chip capacitors*, because their susceptibility is not known: between -30 and 0 ppm the predicted line moves from 61 to 19 Hz. The capacitance (3 pF) is irrelevant; the dielectric and the terminations are what count.
- **Preview**: choose the cut plane, through x or through y (the windows face y).
- **3D view**: if `frascan_cad.npz` is in the folder, the FRASCAN II preset can show the real CAD of the probe head. Delete that file to keep the design out of a public copy; everything else works without it.

Predicted FWHM for the preset (toluene, 500 MHz, 0.5 Hz baseline), lid and capacitors off: 19 Hz at 2nd order shimming, 8.9 Hz at 3rd, 1.2 Hz at 4th.

What the tool still cannot know: the magnet's own inhomogeneity and how well the shims were actually set. It predicts what the probe and sample ADD, assuming ideal shimming up to the chosen order.

Differences from app.py, all small: the builder removes the uniform background before the FFT (a uniform medium only shifts the field, which the lock removes; this removes the fake edge of the computation box), uses a faster real FFT, and sizes the box to hold every part. On the FRASCAN II preset the two agree to about 3 % (6.2 vs 6.0 Hz FWHM).


## Install

    pip install numpy matplotlib streamlit scipy plotly

(scipy is optional: it makes the builder faster and lighter.)

## Run the UI

    python -m streamlit run app.py

Opens in your browser. Five tabs:

| tab | what it does |
|---|---|
| Single design | one geometry, field map, histogram, residual in ppm and Hz |
| Sweep | vary one parameter, plot the residual against it |
| Recommend | given a sample volume, find the best bottom shape and the best place to detect |
| Microwave (1D) | reflected and absorbed power, standing wave, where the bottom focuses the beam |
| Lineshape & matching | the simulated NMR line, and which part of the tube causes it |

## Coordinates

**Every height the app shows or asks for is measured from the TUBE TIP.**
Sliders, the Sweep position, the Recommend window, both microwave plots and
the rod tab all use it, and the axes say "height above tube tip".

Internally `b0core` and `thz.beam_through_bottom` work in launcher
coordinates, where z = 0 is the launcher aperture and the tip sits at
`d_cwg`, because the beam has to propagate that gap and `build_tube` has to
place the tube somewhere. The conversion lives in one place in `app.py`
(`from_tip` / `to_tip`) and nothing below the UI knows about it.

`thz.axial_profile` is the odd one out: it already returns height above the
tube tip. Plotting it against a window drawn in launcher z was a real bug,
wrong by `d_cwg` with nothing on screen to show it. `validate_thz.py` now
checks both functions agree on where the glass-to-liquid interface is, so
the mismatch cannot come back silently.

The only launcher-referenced number left in the sidebar is "tube tip above
the launcher", which is what `d_cwg` means.

## Check it still works

    python validate.py          # the B0 side
    python validate_rod.py      # the external rod
    python validate_thz.py      # the microwave side

`validate.py` compares against three cases with exact analytic answers
(sphere, cylinder parallel to B0, cylinder perpendicular to B0). All three
should agree to four decimals.

`validate_thz.py` checks the transfer matrix against the half-wave and
quarter-wave slab, Fresnel reflection, energy conservation with an
absorbing sample and Beer-Lambert; checks the Gaussian beam against
free-space propagation and the single-surface focal length; then compares
against the HFSS wall sweep and states plainly where the beam model does
not apply.

`validate_rod.py` checks the external rod against two exact limits: the
long rod across B0 (a 2D line dipole, (χ/2)(a/ρ)² cos 2φ) and the
semi-infinite rod along B0 (a charged end disc, (χ/2)[1 − d/√(d²+a²)]).
It also confirms the claim the whole experiment rests on, that the body of
a long vertical rod produces no external field: the spread over an active
volume beside the middle of a 90 mm rod is 0.0003 ppm against 1.07 ppm
straddling its end, a factor of 3500.

Run all three after any change to b0core.py or thz.py.

## The rod experiment

`rod_plan.py` predicts the measurement before you book magnet time, and
writes `rod_plan.png`. For the current tube (4 mm fused quartz, toluene,
active volume 9 to 19 mm above the tip, shimmed to 0.9 Hz rms) a 3 mm
aluminium 7075 rod lowered in beside it gives, at its best height:

| rod axis from the tube axis | residual rms, shims frozen |
|---|---|
| 5 mm | 54 Hz |
| 7 mm | 25 Hz |
| 10 mm | 10 Hz (10.3 Hz in the wider box of holder_tolerance_fast.py) |

against 0.9 Hz with no rod, so the effect is 10 to 60 times the bare
linewidth and is not a subtle measurement. The sensitive position is the
rod tip level with the active volume; by 10 mm above or below it the
effect has mostly gone, which is why rod height was Schönzart's only fit
parameter.

**Box width.** This is the one place in the tool where the box size
matters. The FFT is periodic, so image copies of the rod sit beside the
real one at the padded box pitch and their end fields leak onto the axis.
The tube is immune to this (weak, smooth, and shimming removes what
leaks); a +13 ppm rod sitting off axis is not. `validate_rod.py` measures
it: at a 5 mm box half-width the on-axis field is 10 % low, at 8 mm 3 %,
at 12 mm 1.4 %. Use 12 mm or more for anything you compare with a
measurement.

**Holder tolerance.** `holder_tolerance_fast.py` (about 10 s). The rod field
is solved once and shifted for every position, which is exact because the
method is linear in chi and every shift is a whole voxel. It checks itself
against the end-disc formula (1.5 %), a direct tube+rod solve (0.5 %) and
x padding 3 vs 4 (2 % at 12 mm out). Frozen shims are fitted on the bare
tube in the SAME box as the rod solve; a different box gave a spurious 4 %.
It replaces the old `holder_tolerance.py` (12 mm box clipped the rod at
10 mm out, 1 mm step, central difference reading 0 % at the peak).

Tolerance for a 10 % worst-case error in the predicted rms:

| rod axis from tube axis | peak rms | height, at peak | height, on flank | lateral |
|---|---|---|---|---|
| 5 mm | 55 Hz | ±1.7 mm | ±0.29 mm | ±0.20 mm |
| 7 mm | 25 Hz | ±1.7 mm | ±0.33 mm | ±0.26 mm |
| 10 mm | 10.3 Hz | ±2.0 mm | ±0.47 mm | ±0.37 mm |

Spec for the holder: lateral ±0.25 mm, height ±0.3 mm. Measure at the
peak (rod tip 14 mm above the tube tip, headline agreement) and on the
flanks near 10 and 18 mm (to fit the rod height).

**Frozen shims.** `shim_coef` fits the shim currents on the bare sample;
`shim(..., coef=...)` then subtracts those same currents from the
perturbed field instead of refitting. That is what the bench does, and it
is the only way the comparison means anything, since refitting would shim
most of the rod away. `shim(..., order=None)` is no shimming at all.

## Files

| file | what it is |
|---|---|
| `b0core.py` | physics and geometry. The only file that matters. |
| `validate.py` | analytic checks. Run before trusting anything. |
| `app.py` | Streamlit interface. |
| `compare.py` | batch run of round / flat / punt bottoms. |
| `plotting.py` | all figure styling, in one place. |
| `preview.py` | renders a static figure without the UI. |
| `convcheck.py` | resolution convergence check. |
| `thz.py` | reduced microwave models: transfer matrix and Gaussian beam. |
| `validate_thz.py` | checks for the microwave side, including its limits. |

## Method

k-space (Fourier) convolution with the dipole kernel:

    dBz(r)/B0 = IFFT[ FFT(chi) * (1/3 - kz^2/k^2) ]

Standard in MRI (Salomir 2003; Marques & Bowtell 2005), first order in chi,
which is exact enough for chi ~ 1e-5.

Only field *differences* are physical: a uniform offset is absorbed by the
lock, so the DC term is set to zero and all results are reported as spreads.

`shim()` removes the low-order spatial terms a shim set could remove, so the
reported residual is the part you are actually stuck with.

## Features

- tube geometry: diameter, wall, round / flat / punt bottom, fill height
- a concave meniscus instead of a flat liquid surface
- an insert resting in the bottom: a sphere, or a plug with a domed top
- a coaxial inner tube, with its own diameter, wall and starting height
- the waveguide (CWG) end below the tube: a metal tube whose top sits
  `d_cwg` below the tube tip (sidebar: 'tube tip above the launcher');
  brass is an ESTIMATE, measure or ask before trusting it
- RF coil as a slotted conducting tube: radius, foil thickness, length,
  position, number and width of slots
- 18 materials plus a custom-susceptibility entry
- shim fitting to order 0, 1 or 2, so the number reported is the residual
  you are actually stuck with
- sweep tab: vary one parameter, plot residual against it, export CSV

### Thin-foil handling

Real coil foil is tens of microns, far below any usable voxel. For a thin
sheet the perturbation depends on the product chi x thickness, so the foil
is placed in a single voxel layer with chi scaled to preserve that product.

### Material values

All susceptibilities are LITERATURE ESTIMATES, not measurements. Check any
value before publishing. Use the custom entry for anything you have measured.

## Not yet included

- the coil former and probe body
- shims above second order
- a curved meniscus (the liquid surface is flat)
- any experimental validation


## What this does not do

It solves magnetostatics, which is linear in chi and collapses to one
convolution. It does not solve Maxwell's equations. The microwave tab
holds two reduced models that are exact in their own limits and useless
outside them: a 1D transfer matrix (no curvature, no diffraction) and a
paraxial Gaussian beam (no reflections, no waveguiding). Neither replaces
a full-wave solver. `validate_thz.py` prints the case where the beam model
fails and why.

The susceptibility table is literature estimates, and the microwave
permittivities more so. Check any value before publishing it.

Nothing here has yet been compared against a measured linewidth. Until it
has, it is a calculation, not an instrument.
