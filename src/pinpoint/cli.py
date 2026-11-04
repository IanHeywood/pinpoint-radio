"""Command-line interface. Each stage is its own sub-command so it can be run and checked alone.

  pinpoint fit      sub-band images + region file   -> sky-model YAML (+ report, plots)
  pinpoint show     sky-model YAML                  -> readable summary
  pinpoint predict  sky-model YAML + MS             -> model column(s)
  pinpoint mask     model images + region file      -> model images with the sources zeroed
  pinpoint compare  two sky-model YAMLs             -> position / flux / shape differences
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_fit(a):
    from .fitting import FitOptions, fit_region
    from .images import load_subband_images
    from .regions import load_regions
    from .report import plot_result, summarise
    from .skymodel import SkyModel

    images = load_subband_images(a.images)
    print(f"loaded {len(images)} sub-band images: "
          f"{images[0].freq / 1e6:.2f}-{images[-1].freq / 1e6:.2f} MHz, noise "
          + ", ".join(f"{im.noise * 1e6:.1f}" for im in images) + " uJy/beam")
    sources, exclusions = load_regions(a.regions)
    if not sources:
        sys.exit("no (non-exclusion) regions found")
    nterms = "auto" if a.nterms == "auto" else int(a.nterms)
    opts = FitOptions(background=not a.no_background, spectral_kind=a.spectral, nterms=nterms,
                      max_terms=a.max_terms, band_integrate=a.band_integrate,
                      ref_freq=a.ref_freq, per_band_positions=a.per_band_positions)
    out = []
    for reg in sources:
        res = fit_region(images, reg, exclusions, opts)
        print(summarise(res))
        if a.plots:
            p = plot_result(res, a.plots)
            if p:
                print(f"       plot: {p}")
        src = res.to_source()
        if src.column is None:
            src.column = a.column
        out.append(src)
    model = SkyModel(out, {"images": a.images, "regions": a.regions})
    model.save(a.out)
    print(f"wrote {a.out}")


def _cmd_show(a):
    from .constants import ARCSEC, DEG
    from .report import sexagesimal
    from .skymodel import SkyModel

    m = SkyModel.load(a.model)
    for s in m.sources:
        print(f"{s.name} -> {s.column}")
        for c in s.components:
            shape = f" {c.maj / ARCSEC:.3f}\"x{c.min / ARCSEC:.3f}\" pa {c.pa / DEG:.1f}" if c.type == "gauss" else ""
            print(f"  {c.name:14s} {c.type:5s} {sexagesimal(c.ra, c.dec)}{shape}  "
                  f"{c.spectrum.kind} {c.spectrum.coeffs} @ {c.spectrum.ref_freq / 1e6:.2f} MHz")


def _cmd_predict(a):
    from .ms_io import CasacoreMS, predict_into_ms
    from .skymodel import SkyModel

    m = SkyModel.load(a.model)
    fields = None if a.field is None else [a.field]
    with CasacoreMS(a.ms, readonly=False) as ms:
        rep = predict_into_ms(ms, m, default_column=a.default_column, override_column=a.column, mode=a.mode,
                              field_ids=fields, chunk_rows=a.chunk_rows, backend=a.backend, sign=a.phase_sign,
                              freq_smearing=a.freq_smearing, time_smearing=a.time_smearing,
                              allow_data_column=a.allow_data_column)
    print(rep.summary())


def _cmd_mask(a):
    from .masking import mask_model_images
    from .regions import load_regions

    sources, _ = load_regions(a.regions)
    mask_model_images(a.images, sources, a.out_prefix)


def _cmd_compare(a):
    from .compare import compare_models, format_comparison
    from .skymodel import SkyModel

    print(format_comparison(compare_models(SkyModel.load(a.a), SkyModel.load(a.b))))


def build_parser():
    p = argparse.ArgumentParser(prog="pinpoint", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit sources in sub-band images")
    f.add_argument("--images", required=True, help="wsclean -name prefix (reads PREFIX-NNNN-image.fits)")
    f.add_argument("--regions", required=True, help="DS9 region file")
    f.add_argument("--out", required=True, help="output sky-model YAML")
    f.add_argument("--column", default="DIR1_DATA", help="column for regions without col= (default DIR1_DATA)")
    f.add_argument("--spectral", choices=["logpol", "pol"], default="logpol")
    f.add_argument("--nterms", default="auto", help="number of spectral terms, or 'auto' (AIC)")
    f.add_argument("--max-terms", type=int, default=4)
    f.add_argument("--ref-freq", type=float, default=None, help="Hz (default: mean band frequency)")
    f.add_argument("--band-integrate", action="store_true", help="model each band as its band-average")
    f.add_argument("--no-background", action="store_true")
    f.add_argument("--per-band-positions", action="store_true", help="also fit positions per band (diagnostic)")
    f.add_argument("--plots", default=None, help="directory for diagnostic PNGs")
    f.set_defaults(func=_cmd_fit)

    s = sub.add_parser("show", help="print a sky-model file")
    s.add_argument("model")
    s.set_defaults(func=_cmd_show)

    pr = sub.add_parser("predict", help="DFT-predict a sky model into MS column(s)")
    pr.add_argument("--model", required=True)
    pr.add_argument("--ms", required=True)
    pr.add_argument("--column", default=None, help="write ALL sources to this column (overrides the model file)")
    pr.add_argument("--default-column", default="MODEL_DATA", help="for sources with no column in the model file")
    pr.add_argument("--mode", choices=["overwrite", "add", "subtract"], default="overwrite")
    pr.add_argument("--field", type=int, default=None, help="FIELD_ID (required if the MS has several)")
    pr.add_argument("--chunk-rows", type=int, default=50_000)
    pr.add_argument("--backend", choices=["auto", "numba", "numpy"], default="auto")
    pr.add_argument("--freq-smearing", action="store_true", help="apply channel-width decorrelation")
    pr.add_argument("--time-smearing", action="store_true", help="apply integration-time decorrelation")
    pr.add_argument("--phase-sign", type=float, choices=[-1.0, 1.0], default=-1.0,
                    help="DFT sign convention; leave at -1 unless VALIDATION.md step 2 says otherwise")
    pr.add_argument("--allow-data-column", action="store_true", help="permit writing into DATA / CORRECTED_DATA")
    pr.set_defaults(func=_cmd_predict)

    mk = sub.add_parser("mask", help="zero region pixels in wsclean model images")
    mk.add_argument("--images", required=True, help="wsclean -name prefix (reads PREFIX-NNNN-model.fits)")
    mk.add_argument("--regions", required=True)
    mk.add_argument("--out-prefix", required=True)
    mk.set_defaults(func=_cmd_mask)

    c = sub.add_parser("compare", help="compare two sky-model files")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=_cmd_compare)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
