"""Reading wsclean sub-band images, and cutting out regions from them.

wsclean naming: <prefix>-0000-image.fits, <prefix>-0001-image.fits, ...
(plus -model, -residual, -psf; and <prefix>-MFS-*.fits which we ignore).

Only this module and masking.py touch FITS files; everything downstream works on
plain arrays via the SubbandImage container, so fitting can be tested with
synthetic images without astropy.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from .constants import DEG
from .coords import radec_to_lmn

_CHAN_RE = re.compile(r"-(\d{4})-(image|model|residual|psf|dirty)\.fits$")


@dataclass
class SubbandImage:
    data: np.ndarray  # (ny, nx), Jy/beam
    freq: float  # Hz, band centre
    bandwidth: float  # Hz
    beam: tuple[float, float, float]  # (maj FWHM rad, min FWHM rad, pa rad N->E)
    geometry: object  # has pix2world(x, y), world2pix(ra, dec), pixel_scale (rad)
    noise: float | None = None  # Jy/beam, robust rms
    path: str = ""


def robust_rms(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


# ---------------- FITS (astropy) ----------------

class AstropyGeometry:
    """pix2world / world2pix via astropy.wcs, 0-based pixel indices."""

    def __init__(self, header):
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales

        self.wcs = WCS(header).celestial
        # plain ndarray in the WCS units (degrees for RA/DEC)
        self.pixel_scale = float(np.sqrt(np.prod(np.abs(proj_plane_pixel_scales(self.wcs))))) * DEG

    def pix2world(self, x, y):
        ra, dec = self.wcs.all_pix2world(np.asarray(x, float), np.asarray(y, float), 0)
        return np.asarray(ra) * DEG, np.asarray(dec) * DEG

    def world2pix(self, ra, dec):
        x, y = self.wcs.all_world2pix(np.asarray(ra, float) / DEG, np.asarray(dec, float) / DEG, 0)
        return np.asarray(x), np.asarray(y)


def _freq_from_header(h):
    for i in range(1, int(h.get("NAXIS", 0)) + 1):
        if str(h.get(f"CTYPE{i}", "")).upper().startswith("FREQ"):
            return float(h[f"CRVAL{i}"]), abs(float(h.get(f"CDELT{i}", 0.0)))
    if "RESTFRQ" in h:
        return float(h["RESTFRQ"]), 0.0
    raise ValueError("no FREQ axis in FITS header")


def read_fits_image(path, need_beam=True):
    from astropy.io import fits

    with fits.open(path) as hdul:
        h = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=np.float64)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"{path}: expected a single 2-D plane after squeezing, got shape {data.shape}")
    freq, bw = _freq_from_header(h)
    if need_beam:
        if not all(k in h for k in ("BMAJ", "BMIN", "BPA")):
            raise ValueError(f"{path}: missing BMAJ/BMIN/BPA restoring-beam keywords")
        beam = (float(h["BMAJ"]) * DEG, float(h["BMIN"]) * DEG, float(h["BPA"]) * DEG)
    else:
        beam = (0.0, 0.0, 0.0)
    return data, h, freq, bw, beam


def find_subbands(prefix, kind="image"):
    files = sorted(glob.glob(f"{glob.escape(prefix)}-[0-9][0-9][0-9][0-9]-{kind}.fits"))
    return [f for f in files if _CHAN_RE.search(f)]


def load_subband_images(prefix, use_residual_noise=True) -> list[SubbandImage]:
    files = find_subbands(prefix, "image")
    if not files:
        raise FileNotFoundError(f"no wsclean sub-band images matching {prefix}-NNNN-image.fits")
    out = []
    ref = None
    for f in files:
        data, h, freq, bw, beam = read_fits_image(f)
        key = tuple(h.get(k) for k in ("NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CRPIX1", "CRPIX2"))
        if ref is None:
            ref = key
            geom = AstropyGeometry(h)
        elif key != ref:
            raise ValueError(f"{f}: pixel grid differs from first sub-band; all sub-bands must share one grid")
        noise = None
        resid = f.replace("-image.fits", "-residual.fits")
        if use_residual_noise and os.path.exists(resid):
            rdata, *_ = read_fits_image(resid, need_beam=False)
            noise = robust_rms(rdata)
        if noise is None:
            noise = robust_rms(data)
        out.append(SubbandImage(data, freq, bw, beam, geom, noise, f))
    return out


# ---------------- cutouts ----------------

@dataclass
class Cutout:
    ys: slice
    xs: slice
    mask: np.ndarray  # (ny, nx) bool: pixels used
    de: np.ndarray  # East offsets (rad) of each pixel from the region centre
    dn: np.ndarray  # North offsets (rad)
    ra: np.ndarray
    dec: np.ndarray
    centre: tuple[float, float]  # (ra, dec) of region centre


def region_centre_radec(region, geometry):
    if region.frame == "sky":
        return region.centre
    ra, dec = geometry.pix2world(region.centre[0], region.centre[1])
    return float(ra), float(dec)


def region_cutout(region, geometry, shape, exclusions=(), pad=2) -> Cutout:
    ny, nx = shape
    cra, cdec = region_centre_radec(region, geometry)
    cx, cy = geometry.world2pix(cra, cdec)
    r = region.bounding_radius()
    rpix = r / geometry.pixel_scale if region.frame == "sky" else r
    x0 = max(int(np.floor(cx - rpix)) - pad, 0)
    x1 = min(int(np.ceil(cx + rpix)) + pad + 1, nx)
    y0 = max(int(np.floor(cy - rpix)) - pad, 0)
    y1 = min(int(np.ceil(cy + rpix)) + pad + 1, ny)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"region {region.name!r} lies outside the image")
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    ra, dec = geometry.pix2world(xx, yy)
    mask = region.contains(ra, dec, xx, yy)
    for ex in exclusions:
        mask &= ~ex.contains(ra, dec, xx, yy)
    de, dn, _ = radec_to_lmn(ra, dec, cra, cdec)
    return Cutout(slice(y0, y1), slice(x0, x1), mask, de, dn, ra, dec, (cra, cdec))
