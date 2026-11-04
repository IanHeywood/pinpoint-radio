# Checks on real data

The tests prove internal correctness. These checks show the tool does the right thing on
your data. They're ordered so each one builds on the last.

## 1. Run the full test suite, with nothing skipped

```bash
pip install -e ".[fast,plots,test]" codex-africanus
pytest -rs -s
```

All four `test_optional_*` files should pass. In particular, `test_optional_africanus.py`
pins the phase sign and Gaussian conventions against an independent implementation.

## 2. Round trip through wsclean (sign, orientation, units)

This is the decisive convention check. It doesn't trust any of our maths.

```bash
pinpoint fit --images img --regions bright.reg --out fit1.yaml
pinpoint predict --model fit1.yaml --ms copy.ms --column PINPOINT_TEST
wsclean -name rt -data-column PINPOINT_TEST -channels-out 8 <same -size/-scale/-weight> -niter 0 copy.ms
```

Then make a restored image of that column. Either CLEAN it lightly
(`-niter 1000 -auto-threshold 1`), or fit the dirty image with `--no-background` and a
matching PSF. Then compare:

```bash
pinpoint fit --images rt --regions bright.reg --out fit2.yaml
pinpoint compare fit1.yaml fit2.yaml
```

- **Expected:** offsets well below 1/100 of a pixel, a flux ratio within 1e-3, and the same
  PA.
- **Position mirrored about the phase centre:** the phase sign is flipped. Re-run predict
  with `--phase-sign -1`, and please report it, because that would contradict the CASA
  convention confirmed against africanus.
- **Only one axis flipped:** l/m orientation is wrong.

## 3. Is the fit as good as its error bars?

- Run `pinpoint fit … --per-band-positions --plots fitplots`. The per-band positions should
  scatter around zero, mostly within ±2σ. A systematic trend with frequency points to
  chromatic astrometry or calibration problems; the fit itself is still fine.
- Look at the fit-residual panels in the PNGs. Structure left at the source position means
  the component model needs another component, e.g. `point+gauss`.

## 4. The actual goal: dynamic range after peeling

1. Run QuartiCal twice. The first run is a direction-independent solve against the full
   wsclean model. The second solves against `MODEL_DATA` (masked field) plus `DIR1_DATA`
   (pinpoint), with a direction-dependent term on `DIR1_DATA`.
2. Image both results with identical settings.
3. Measure the rms in an annulus round the source and far from it.

The gridding and pixellisation hypothesis predicts that most of the improvement appears
close to the source, on the scales of the long-baseline sidelobes.

Control experiment: put the pixel-snapped model into `DIR1_DATA` instead, using the same
YAML with RA/Dec rounded to the nearest pixel centre. This separates the effect of the
direction-dependent solve from the effect of the precise position.

## 5. Speed

`pinpoint predict` prints read, compute and write times. If compute dominates with numba,
a GPU kernel is worth writing. If write dominates (likely for large MeerKAT data sets), the
next step is overlapping I/O with compute, or a faster table backend.
