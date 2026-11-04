"""Synthetic wsclean-like restored sub-band images.

IMPORTANT: this renderer deliberately does NOT reuse pinpoint.shapes. It works in
pixel coordinates with its own rotation formula (angle counter-clockwise from +x,
derived from the header PA), and convolves extended sources numerically with an
FFT. So if the fitter's sky-frame conventions were wrong (e.g. PA sign, East/West
flip) the fitting tests would fail rather than agreeing with themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from ..constants import ARCSEC, FWHM_TO_SIGMA
from ..coords import SinGeometry
from ..images import SubbandImage


@dataclass
class TrueComponent:
    ra: float
    dec: float
    flux_fn: object  # callable nu -> Jy
    maj: float = 0.0  # FWHM rad (0 => point)
    min: float = 0.0
    pa: float = 0.0  # rad N->E


def _pix_gauss(dx, dy, smaj_pix, smin_pix, pa):
    """Unit-peak Gaussian in pixel coords. Assumes North = +y and East = -x
    (CDELT1 < 0, CDELT2 > 0), so a PA (N through E) maps to a pixel-frame angle
    t = pa + 90 deg counter-clockwise from +x."""
    t = pa + np.pi / 2
    xp = dx * np.cos(t) + dy * np.sin(t)
    yp = -dx * np.sin(t) + dy * np.cos(t)
    return np.exp(-0.5 * ((xp / smaj_pix) ** 2 + (yp / smin_pix) ** 2))


def render_restored(geom: SinGeometry, comps, nu, beam):
    """Restored image (Jy/beam) of the components at frequency nu."""
    pix = geom.pixel_scale
    yy, xx = np.mgrid[0:geom.ny, 0:geom.nx].astype(np.float64)
    bmaj, bmin, bpa = beam
    sb_maj, sb_min = bmaj * FWHM_TO_SIGMA / pix, bmin * FWHM_TO_SIGMA / pix
    img = np.zeros((geom.ny, geom.nx))
    for c in comps:
        x0, y0 = geom.world2pix(c.ra, c.dec)
        S = c.flux_fn(nu)
        if c.maj == 0.0:
            img += S * _pix_gauss(xx - x0, yy - y0, sb_maj, sb_min, bpa)
        else:
            src = _pix_gauss(xx - x0, yy - y0, c.maj * FWHM_TO_SIGMA / pix, c.min * FWHM_TO_SIGMA / pix, c.pa)
            src *= S / src.sum()  # Jy per pixel
            # odd-sized kernel so that mode="same" introduces no half/whole-pixel shift
            hy, hx = geom.ny // 2, geom.nx // 2
            ky, kx = np.mgrid[-hy:hy + 1, -hx:hx + 1].astype(np.float64)
            kern = _pix_gauss(kx, ky, sb_maj, sb_min, bpa)
            img += fftconvolve(src, kern, mode="same")
    return img


def correlated_noise(shape, beam, pix, rms, rng):
    """White noise smoothed by the beam and rescaled to the requested rms."""
    ny, nx = shape
    ky, kx = np.mgrid[-(ny // 2):ny // 2 + 1, -(nx // 2):nx // 2 + 1].astype(np.float64)
    kern = _pix_gauss(kx, ky, beam[0] * FWHM_TO_SIGMA / pix, beam[1] * FWHM_TO_SIGMA / pix, beam[2])
    n = fftconvolve(rng.normal(size=shape), kern, mode="same")
    return n * (rms / n.std())


def make_subbands(comps, freqs, *, ra0, dec0, npix=128, cell_arcsec=1.0, beam_at_1ghz=(8.0, 5.0, 30.0),
                  rms=1e-4, bandwidth=None, rng=None, correlated=True):
    """List of SubbandImage with beams scaling as 1/nu. beam_at_1ghz in (arcsec, arcsec, deg)."""
    rng = rng or np.random.default_rng(0)
    geom = SinGeometry(npix, npix, ra0, dec0, -cell_arcsec * ARCSEC, cell_arcsec * ARCSEC,
                       npix // 2 + 1.0, npix // 2 + 1.0)
    if bandwidth is None:
        bandwidth = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1e6
    out = []
    for nu in freqs:
        s = 1e9 / nu
        beam = (beam_at_1ghz[0] * s * ARCSEC, beam_at_1ghz[1] * s * ARCSEC, np.deg2rad(beam_at_1ghz[2]))
        img = render_restored(geom, comps, nu, beam)
        if rms > 0:
            if correlated:
                img = img + correlated_noise(img.shape, beam, geom.pixel_scale, rms, rng)
            else:
                img = img + rng.normal(0, rms, img.shape)
        out.append(SubbandImage(img, float(nu), bandwidth, beam, geom, rms if rms > 0 else 1e-6))
    return out
