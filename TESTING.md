# What the tests establish

Run the whole suite with `pytest -rs -s`. `-rs` lists anything that was skipped and why;
`-s` shows the kernel throughput figure.

Every test file starts with a plain-language docstring listing exactly what it checks.
This page maps the pipeline stages to those files.

| Stage | Module | Test file | Key claims checked |
|---|---|---|---|
| Sky coordinates | `coords.py` | `test_coords.py` | East = +l and North = +m; sky → (l, m) → sky round trip to < 1 µas at any declination |
| Gaussian shapes | `shapes.py` | `test_shapes.py` | PA runs North through East; adding covariances matches a brute-force numerical convolution, including Jy/beam peaks |
| Spectra | `spectral.py` | `test_spectral.py` | Formulas match wsclean conventions; fits recover the truth; error bars are honest (pull test); band integration removes curvature bias; AIC picks the right order |
| Model file | `skymodel.py` | `test_skymodel.py` | Lossless YAML round trip in human units; source → column mapping |
| Regions | `regions.py`, `images.region_cutout` | `test_regions.py` | DS9 syntax variants; text tags; exclusions; masks for every shape; DS9 angle convention |
| Image fitting | `fitting.py` | `test_fitting.py` | Off-grid source recovered to ~1/1000 pixel; position errors honest under beam-correlated noise; Gaussian size and PA recovered at 0°, 60° and 135°; point + Gaussian blend; background; per-band positions |
| DFT predict | `predict.py` | `test_predict.py` | Analytic point and Gaussian visibilities; Gaussian equals a numerical Fourier transform of the image-plane Gaussian; fast recurrence kernel equals direct evaluation to 1e-11 over 1500 channels (ascending, descending, irregular); smearing equals brute-force averaging; correlation mapping |
| MS writing | `ms_io.py` | `test_ms_io.py` | Correct values in every spectral window and correlation setup; add and subtract modes; per-source columns; DATA protected and untouched; zero-fill of other fields; results independent of chunk size |
| Masking, compare, CLI | `masking.py`, `compare.py`, `cli.py` | `test_masking_compare_cli.py` | Only in-region pixels are zeroed; known offsets reported correctly; every sub-command parses |
| End to end | everything | `test_end_to_end.py` | Images → fit → file → MS reproduces the true visibilities to 1e-6 of the flux; a pixel-snapped model is off by ~38% |

## How the tests avoid grading themselves

- **Synthetic images.** `pinpoint/testing/synth_images.py` renders sources in pixel
  coordinates with its own rotation formula, and convolves extended sources by FFT. It
  shares no code with the fitter's sky-frame maths, so a PA sign or East/West flip in the
  fitter would make the fitting tests fail.
- **Gaussian visibilities.** These are checked against a brute-force numerical Fourier
  transform of an explicitly written image-plane Gaussian.
- **Smearing factors.** These are checked by averaging hundreds of unsmeared visibilities
  across the channel or integration. For time smearing, the uvw tracks are regenerated
  from the textbook hour-angle formulas at each sub-step.
- **The fast kernel.** This is checked against a deliberately naive vectorised numpy
  implementation.

## Optional tests, which skip when a package is missing

| File | Needs | Checks |
|---|---|---|
| `test_optional_astropy.py` | astropy | pinpoint's SIN projection equals astropy.wcs to < 1 µas over a 2° image; reading wsclean-style 4-D FITS (frequency, beam, residual-image noise, MFS ignored); fitting through FITS; masking FITS model images |
| `test_optional_casacore.py` | python-casacore | Real-MS metadata; predicting into a real MS gives bit-identical results to the in-memory MS, so `test_ms_io.py` carries over; column creation; DATA untouched |
| `test_optional_numba.py` | numba | The compiled kernel equals the numpy reference; prints throughput |
| `test_optional_africanus.py` | codex-africanus | Independent check of phase sign, l/m orientation, w-term, spectral polynomials and Gaussian PA against another widely used DFT |

**Status.** The first CI run (GitHub Actions, Python 3.12) passed every astropy,
python-casacore and numba test. The numba kernel ran at about 313 M
component-visibilities/s. The africanus cross-check failed, and that failure was the
test doing its job: pinpoint's original phase sign was the opposite of the convention
used for MS data (africanus' 'casa' convention, also used by MeqTrees). The default is
now +1. The run also showed that africanus uses natural logs for log-polynomial
curvature, while wsclean documents base 10; the test now converts between the two.
