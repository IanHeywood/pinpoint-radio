"""Sky coordinate maths.

Conventions (used everywhere in pinpoint):
  * All angles are radians internally.
  * (l, m, n) are direction cosines relative to a reference direction (ra0, dec0):
      l points East (direction of increasing RA), m points North.
      l = cos(dec) sin(ra - ra0)
      m = sin(dec) cos(dec0) - cos(dec) sin(dec0) cos(ra - ra0)
      n = sin(dec) sin(dec0) + cos(dec) cos(dec0) cos(ra - ra0)
    This is the standard orthographic (SIN) projection used by radio imagers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def radec_to_lmn(ra, dec, ra0, dec0):
    """Direction cosines (l, m, n) of (ra, dec) relative to (ra0, dec0)."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    dra = ra - ra0
    cd, sd = np.cos(dec), np.sin(dec)
    l = cd * np.sin(dra)
    m = sd * np.cos(dec0) - cd * np.sin(dec0) * np.cos(dra)
    n = sd * np.sin(dec0) + cd * np.cos(dec0) * np.cos(dra)
    return l, m, n


def lmn_to_radec(l, m, ra0, dec0):
    """Inverse of radec_to_lmn (front hemisphere, n >= 0)."""
    l = np.asarray(l, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    n = np.sqrt(np.clip(1.0 - l * l - m * m, 0.0, None))
    dec = np.arcsin(np.clip(m * np.cos(dec0) + n * np.sin(dec0), -1.0, 1.0))
    ra = ra0 + np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    return np.mod(ra, 2 * np.pi), dec


def angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle separation (Vincenty formula, stable at all separations)."""
    dra = ra2 - ra1
    s1, c1 = np.sin(dec1), np.cos(dec1)
    s2, c2 = np.sin(dec2), np.cos(dec2)
    num = np.hypot(c2 * np.sin(dra), c1 * s2 - s1 * c2 * np.cos(dra))
    den = s1 * s2 + c1 * c2 * np.cos(dra)
    return np.arctan2(num, den)


@dataclass
class SinGeometry:
    """Minimal FITS SIN-projection pixel grid (no rotation, no PV terms).

    This mirrors what wsclean writes (RA---SIN / DEC--SIN). It is used to build
    synthetic test images without astropy, and is cross-checked against
    astropy.wcs in tests/test_optional_astropy.py.

    Pixel indices here are 0-based numpy indices (x = column, y = row);
    FITS CRPIX is 1-based, hence the +1.
    """

    nx: int
    ny: int
    crval_ra: float  # rad
    crval_dec: float  # rad
    cdelt_x: float  # rad (negative for RA increasing to the left)
    cdelt_y: float  # rad
    crpix_x: float  # 1-based FITS
    crpix_y: float  # 1-based FITS

    def pix2world(self, x, y):
        l = self.cdelt_x * (np.asarray(x, dtype=np.float64) + 1.0 - self.crpix_x)
        m = self.cdelt_y * (np.asarray(y, dtype=np.float64) + 1.0 - self.crpix_y)
        return lmn_to_radec(l, m, self.crval_ra, self.crval_dec)

    def world2pix(self, ra, dec):
        l, m, _ = radec_to_lmn(ra, dec, self.crval_ra, self.crval_dec)
        x = l / self.cdelt_x + self.crpix_x - 1.0
        y = m / self.cdelt_y + self.crpix_y - 1.0
        return x, y

    @property
    def pixel_scale(self) -> float:
        return float(np.sqrt(abs(self.cdelt_x * self.cdelt_y)))
