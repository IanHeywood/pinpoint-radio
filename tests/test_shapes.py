"""Stage: Gaussian shape conventions (pinpoint/shapes.py).

What these tests establish:
  * position angle is measured North through East: PA=0 elongates N-S, PA=90 deg E-W;
  * (maj, min, PA) <-> covariance converts losslessly;
  * convolving two Gaussians (beam + source) by adding covariances agrees with a
    brute-force numerical convolution, including the peak brightness in Jy/beam.
"""

import numpy as np
from scipy.signal import fftconvolve

from pinpoint.constants import ARCSEC, DEG
from pinpoint.shapes import cov_to_shape, gaussian_profile, restored_profile, shape_to_cov


def test_pa_zero_is_north_south():
    cov = shape_to_cov(10 * ARCSEC, 2 * ARCSEC, 0.0)
    assert cov[1, 1] > cov[0, 0]  # more variance along North
    assert abs(cov[0, 1]) < 1e-30


def test_pa_ninety_is_east_west():
    cov = shape_to_cov(10 * ARCSEC, 2 * ARCSEC, 90 * DEG)
    assert cov[0, 0] > cov[1, 1]


def test_pa_45_points_north_east():
    # Major axis toward NE: profile is higher at (+E,+N) than (+E,-N)
    cov = shape_to_cov(10 * ARCSEC, 2 * ARCSEC, 45 * DEG)
    d = 3 * ARCSEC
    assert gaussian_profile(d, d, cov) > gaussian_profile(d, -d, cov)


def test_round_trip_shape_cov():
    rng = np.random.default_rng(3)
    for _ in range(200):
        maj = rng.uniform(1, 20) * ARCSEC
        mn = rng.uniform(0.1, 1.0) * maj * 0.999
        pa = rng.uniform(0, np.pi)
        maj2, mn2, pa2 = cov_to_shape(shape_to_cov(maj, mn, pa))
        assert np.isclose(maj, maj2, rtol=1e-9)
        assert np.isclose(mn, mn2, rtol=1e-9)
        dpa = (pa - pa2 + np.pi / 2) % np.pi - np.pi / 2
        assert abs(dpa) < 1e-7


def test_fwhm_definition():
    cov = shape_to_cov(4 * ARCSEC, 4 * ARCSEC, 0.0)
    assert np.isclose(gaussian_profile(2 * ARCSEC, 0.0, cov), 0.5, rtol=1e-12)


def test_convolution_matches_numerical():
    pix = 0.25 * ARCSEC
    n = 257
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    de = (xx - c) * pix  # East to the right here (just a local frame for this test)
    dn = (yy - c) * pix
    beam = shape_to_cov(6 * ARCSEC, 3 * ARCSEC, 30 * DEG)
    src = shape_to_cov(4 * ARCSEC, 2 * ARCSEC, 110 * DEG)
    S = 2.5  # Jy
    src_img = gaussian_profile(de, dn, src)
    src_img *= S / src_img.sum()  # Jy per pixel
    beam_img = gaussian_profile(de, dn, beam)  # unit peak
    restored_num = fftconvolve(src_img, beam_img, mode="same")
    restored_ana = S * restored_profile(de, dn, src, beam)
    assert np.max(np.abs(restored_num - restored_ana)) < 1e-6 * S
