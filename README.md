# pinpoint

*Installed as `pinpoint-radio`; imported and run as `pinpoint`.*

Fit bright, problematic sources in wsclean sub-band images to sub-pixel precision, and
DFT-predict them into a Measurement Set column, for example to peel them in QuartiCal with a
direction-dependent gain term.

A CLEAN model, or a wsclean component list, puts every component on a pixel centre. A
bright source that falls between pixels is then modelled as a cluster of pixel-locked
deltas. For a source 0.4 pixels off the grid, on 8 km baselines, that is wrong by tens of
per cent (see `tests/test_end_to_end.py`). pinpoint fits one continuous position, shared by
every sub-band, plus an intrinsic shape and a smooth apparent spectrum. It then predicts
that model analytically, with no gridding and no pixels.

## Install

```bash
pip install -e ".[fast,plots,test]"      # from a clone; later: pip install "pinpoint-radio[fast]"
pip install codex-africanus              # optional: only for one cross-check test
pytest -s                                # run the tests first; see TESTING.md
```

Runtime dependencies are numpy, scipy, astropy, pyyaml and python-casacore. numba is
optional but strongly recommended: it's the fast predictor.

## Workflow (peeling with QuartiCal)

```bash
# 1. Image with sub-bands as usual (restored, residual and model images are all used)
wsclean -name img -channels-out 8 -join-channels ... obs.ms

# 2. Draw a DS9 region round each problem source; tag what to fit and where to put it
#    circle(...) # text={point col=DIR1_DATA name=3C123}
#    -circle(...)            <- optional exclusion: e.g. a neighbour inside the region

# 3. Fit
pinpoint fit --images img --regions bright.reg --out model.yaml --per-band-positions --plots fitplots

# 4. Rest-of-field model with the sources removed, predicted by wsclean into MODEL_DATA
pinpoint mask --images img --regions bright.reg --out-prefix img-masked
wsclean -predict -name img-masked -channels-out 8 <same -size/-scale as step 1> obs.ms

# 5. DFT-predict the parametric source(s) into their own column(s)
pinpoint predict --model model.yaml --ms obs.ms --time-smearing --freq-smearing

# 6. QuartiCal: direction-independent term plus a direction-dependent term on DIR1_DATA
#    e.g. input_model.recipe="MODEL_DATA:DIR1_DATA", with the dE term direction-dependent
#    (check the recipe syntax against your QuartiCal version's documentation)
```

Each step is a separate command, so you can stop and inspect after any of them:
`pinpoint show model.yaml`, the diagnostic PNGs, and `pinpoint compare a.yaml b.yaml`.

## Region text tag

| tag | meaning |
|---|---|
| `point`, `gauss`, `point+gauss`, `2point` | components to fit jointly in this region (default `point`) |
| `col=DIR1_DATA` | MS column for this source (default: `--column` at fit time) |
| `name=3C123` | source name in the model file |

Coordinate systems: `fk5`, `icrs` or `image`. Shapes: circle, ellipse, box and polygon.
Lines starting with `-` are exclusions.

## The model file

`model.yaml` is plain YAML in human units (degrees, arcsec, Hz, Jy), with 1-sigma errors.
You can read and edit it. Spectra use the wsclean conventions:
- `logpol`: S = c0 · 10^(c1·y + c2·y² …), with y = log10(ν/ν0), so c1 is the spectral index.
- `pol`: S = c0 + c1·x + c2·x² …, with x = ν/ν0 − 1.

## Conventions (all in one place)

- **(l, m):** l points East, m points North, using the SIN projection about the MS
  `PHASE_DIR` of the field.
- **Phase:** V = Σ S(ν) · G · exp(−2πi (ν/c)(u l + v m + w(n−1))), the CASA/wsclean
  sign. `--phase-sign` exists only so it can be checked on your data
  (VALIDATION.md step 2).
- **Gaussians:** FWHM major and minor axes, with PA measured North through East (the same
  as BPA in FITS headers). G is the exact Fourier transform of the image-plane Gaussian.
- **Correlations:** XX = YY = RR = LL = I, with zero cross-hands. This follows the MS
  convention of XX = I + Q with no factor ½. Stokes I only in this version.
- **Smearing (optional):** a sinc factor for the channel width, and a sinc factor for the
  integration time using the analytic hour-angle rate of change of the phase.

## What this version does not do (yet)

- **No polarisation.** There are no Q/U/V components and no parallactic-angle rotation.
  For a strongly polarised source on linear feeds, XX ≠ YY, so expect residuals.
- **No GPU kernel.** The CPU numba kernel uses channel recurrences. A CUDA version should
  only follow once benchmarks show compute, not MS I/O, is the bottleneck.
- **Sequential MS I/O.** `predict` reports read, compute and write times separately, so you
  can see where the time goes.
- **Shared phase centre.** All sub-band images must share one pixel grid, and a
  multi-field MS is predicted one field at a time (`--field`).
- **Flat-sky Gaussian orientation.** A Gaussian's orientation is not re-projected for
  sources very far (many degrees) from the phase centre.

See TESTING.md for what each test establishes, and VALIDATION.md for checks on real data.
