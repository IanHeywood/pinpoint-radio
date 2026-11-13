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


def zero_regions_inplace(data, geometry, regions):
    """Set region pixels to 0 IN PLACE in `data`, whose last two axes are (y, x); any leading
    axes (e.g. wsclean's 1 x 1 FREQ/STOKES) are zeroed too. Only each region's small cutout is
    touched, so `data` can be a memory-mapped 35k x 35k FITS image.
    Returns (number of non-zero pixels zeroed, flux removed)."""
    shape = data.shape[-2:]
    nz = 0
    removed = 0.0
    for r in regions:
        cut = region_cutout(r, geometry, shape, pad=1)
        view = data[..., cut.ys, cut.xs]  # basic slicing: a view, not a copy
        vals = np.asarray(view[..., cut.mask], dtype=np.float64)
        removed += float(vals.sum())
        nz += int(np.count_nonzero(vals))
        view[..., cut.mask] = 0
    return nz, removed


def zero_regions(data, geometry, regions):
    """Return (copy of data with region pixels set to 0, number of pixels zeroed, flux removed)."""
    out = np.array(data, dtype=np.float64, copy=True)
    nz, removed = zero_regions_inplace(out, geometry, regions)
    return out, nz, removed


def mask_model_images(prefix, regions, out_prefix=None, log=print, in_place=False):
    """Zero source regions in every <prefix>-NNNN-model.fits (and -MFS-model.fits).

    Each output is made by a plain file copy, then only the region cutouts are modified
    through a memory map, so memory use and work are tiny even for 35k x 35k images.
    in_place=True skips the copy and modifies the input files themselves.
    """
    import shutil

    from astropy.io import fits

    from .images import AstropyGeometry

    if not in_place and not out_prefix:
        raise ValueError("give out_prefix, or in_place=True")
    files = find_subbands(prefix, "model")
    mfs = f"{prefix}-MFS-model.fits"
    if os.path.exists(mfs):
        files.append(mfs)
    if not files:
        raise FileNotFoundError(f"no model images matching {prefix}-NNNN-model.fits")
    written = []
    for f in files:
        out = f if in_place else out_prefix + f[len(prefix):]  # "<prefix>-0003-model.fits" -> "<out_prefix>-0003-model.fits"
        if not in_place:
            if os.path.abspath(out) == os.path.abspath(f):
                raise ValueError("out_prefix equals the input prefix; use in_place=True to modify inputs")
            shutil.copyfile(f, out)  # kernel-level copy; no data passes through Python
        with fits.open(out, mode="update", memmap=True, do_not_scale_image_data=True) as hdul:
            hdr = hdul[0].header
            if float(hdr.get("BSCALE", 1.0)) != 1.0 or float(hdr.get("BZERO", 0.0)) != 0.0:
                raise ValueError(f"{out}: scaled FITS data (BSCALE/BZERO) are not supported")
            geom = AstropyGeometry(hdr)
            nz, removed = zero_regions_inplace(hdul[0].data, geom, regions)
            hdul.flush()
        log(f"{os.path.basename(f)} -> {os.path.basename(out)}: zeroed {nz} clean components ({removed:.4g} Jy)")
        written.append(out)
    return written
