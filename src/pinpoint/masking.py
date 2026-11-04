"""Zero the fitted sources' regions in wsclean model images.

The result is a "rest of the field" model: predict it with wsclean into MODEL_DATA,
and predict the parametric source(s) with pinpoint into their own column(s). This
avoids modelling the bright source twice in the direction-dependent solve.

Writes <out_prefix>-NNNN-model.fits (and <out_prefix>-MFS-model.fits if present), so
`wsclean -predict -name <out_prefix> -channels-out N ...` picks them up.
"""

from __future__ import annotations

import os

import numpy as np

from .images import find_subbands, region_cutout


def zero_regions(data, geometry, regions):
    """Return (copy of data with region pixels set to 0, number of pixels zeroed, flux removed)."""
    out = np.array(data, dtype=np.float64, copy=True)
    nz = 0
    removed = 0.0
    for r in regions:
        cut = region_cutout(r, geometry, out.shape, pad=1)
        view = out[cut.ys, cut.xs]
        removed += float(view[cut.mask].sum())
        nz += int(np.count_nonzero(view[cut.mask]))
        view[cut.mask] = 0.0
    return out, nz, removed


def mask_model_images(prefix, regions, out_prefix, log=print):
    from astropy.io import fits

    from .images import AstropyGeometry

    files = find_subbands(prefix, "model")
    mfs = f"{prefix}-MFS-model.fits"
    if os.path.exists(mfs):
        files.append(mfs)
    if not files:
        raise FileNotFoundError(f"no model images matching {prefix}-NNNN-model.fits")
    written = []
    for f in files:
        with fits.open(f) as hdul:
            hdr = hdul[0].header
            raw = hdul[0].data
        plane = np.squeeze(raw)
        geom = AstropyGeometry(hdr)
        masked, nz, removed = zero_regions(plane, geom, regions)
        out = out_prefix + f[len(prefix):]  # "<prefix>-0003-model.fits" -> "<out_prefix>-0003-model.fits"
        fits.writeto(out, masked.reshape(raw.shape).astype(raw.dtype), hdr, overwrite=True)
        log(f"{os.path.basename(f)} -> {os.path.basename(out)}: zeroed {nz} clean components ({removed:.4g} Jy)")
        written.append(out)
    return written
