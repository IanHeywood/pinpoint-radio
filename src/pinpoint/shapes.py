"""Elliptical Gaussian shape maths.

A shape is described either as (major FWHM, minor FWHM, position angle) or as a
2x2 covariance matrix in the local (East, North) sky frame.

Position angle convention: radians, measured from North through East
(the astronomical convention, same as BPA in FITS headers and wsclean).
So pa = 0 puts the major axis North-South; pa = pi/2 puts it East-West.

Everything that needs a Gaussian (image-plane fitting, beam convolution and the
visibility-plane taper) goes through `shape_to_cov`, so there is exactly one
place where the convention lives.
"""

from __future__ import annotations

import numpy as np

from .constants import FWHM_TO_SIGMA


def shape_to_cov(maj, mn, pa):
    """Covariance matrix [[C_ee, C_en], [C_en, C_nn]] (rad^2) from FWHMs and PA."""
    smaj = maj * FWHM_TO_SIGMA
    smin = mn * FWHM_TO_SIGMA
    s, c = np.sin(pa), np.cos(pa)
    # major-axis unit vector in (E, N) is (sin pa, cos pa); minor is (cos pa, -sin pa)
    cee = smaj**2 * s * s + smin**2 * c * c
    cnn = smaj**2 * c * c + smin**2 * s * s
    cen = (smaj**2 - smin**2) * s * c
    return np.array([[cee, cen], [cen, cnn]], dtype=np.float64)


def cov_to_shape(cov):
    """Inverse of shape_to_cov. Returns (maj FWHM, min FWHM, pa in [0, pi))."""
    cov = np.asarray(cov, dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    evals = np.clip(evals, 0.0, None)
    smin, smaj = np.sqrt(evals[0]), np.sqrt(evals[1])
    ve, vn = evecs[0, 1], evecs[1, 1]  # major eigenvector (E, N)
    pa = np.mod(np.arctan2(ve, vn), np.pi)
    if np.isclose(smaj, smin, rtol=1e-9, atol=0.0):
        pa = 0.0
    return smaj / FWHM_TO_SIGMA, smin / FWHM_TO_SIGMA, float(pa)


def gaussian_profile(de, dn, cov):
    """Unit-peak Gaussian exp(-0.5 r^T C^-1 r) at offsets (de, dn) in radians."""
    inv = np.linalg.inv(cov)
    q = inv[0, 0] * de * de + 2.0 * inv[0, 1] * de * dn + inv[1, 1] * dn * dn
    return np.exp(-0.5 * q)


def restored_profile(de, dn, src_cov, beam_cov):
    """Image (Jy/beam) of a 1 Jy source with covariance src_cov restored with a
    unit-peak beam of covariance beam_cov. A point source has src_cov = 0.

    Peak = sqrt(det(beam) / det(beam + src)), which is 1 for a point source.
    """
    tot = beam_cov + src_cov
    peak = np.sqrt(np.linalg.det(beam_cov) / np.linalg.det(tot))
    return peak * gaussian_profile(de, dn, tot)
