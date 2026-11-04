"""Stage: coordinate maths (pinpoint/coords.py).

What these tests establish:
  * the phase centre maps to (l, m, n) = (0, 0, 1);
  * East is +l and North is +m;
  * converting sky -> (l, m) -> sky returns the input to < 1 micro-arcsec;
  * the synthetic SIN image grid used elsewhere in the tests is self-consistent.
"""

import numpy as np
import pytest

from pinpoint.constants import ARCSEC, DEG
from pinpoint.coords import SinGeometry, angular_separation, lmn_to_radec, radec_to_lmn

RA0, DEC0 = 123.4 * DEG, -35.0 * DEG


def test_phase_centre_is_origin():
    l, m, n = radec_to_lmn(RA0, DEC0, RA0, DEC0)
    assert abs(l) < 1e-15 and abs(m) < 1e-15 and abs(n - 1) < 1e-15


def test_east_is_positive_l_north_is_positive_m():
    l, m, _ = radec_to_lmn(RA0 + 10 * ARCSEC / np.cos(DEC0), DEC0, RA0, DEC0)
    assert l > 0 and abs(m) < 1e-3 * abs(l)
    l, m, _ = radec_to_lmn(RA0, DEC0 + 10 * ARCSEC, RA0, DEC0)
    assert m > 0 and abs(l) < 1e-12


def test_small_offset_magnitude():
    # 1 arcmin North: m = sin(1')
    _, m, n = radec_to_lmn(RA0, DEC0 + 60 * ARCSEC, RA0, DEC0)
    assert np.isclose(m, np.sin(60 * ARCSEC), rtol=1e-12)
    assert np.isclose(n, np.cos(60 * ARCSEC), rtol=1e-12)


@pytest.mark.parametrize("dec0", [-80 * DEG, -30 * DEG, 0.0, 45 * DEG, 85 * DEG])
def test_round_trip(dec0):
    rng = np.random.default_rng(1)
    ra = RA0 + rng.uniform(-2, 2, 500) * DEG
    dec = dec0 + rng.uniform(-2, 2, 500) * DEG
    dec = np.clip(dec, -89.9 * DEG, 89.9 * DEG)
    l, m, _ = radec_to_lmn(ra, dec, RA0, dec0)
    ra2, dec2 = lmn_to_radec(l, m, RA0, dec0)
    sep = angular_separation(ra, dec, ra2, dec2)
    assert np.max(sep) < 1e-6 * ARCSEC


def test_sin_geometry_round_trip_and_orientation():
    g = SinGeometry(256, 256, RA0, DEC0, -1.5 * ARCSEC, 1.5 * ARCSEC, 129.0, 129.0)
    # CRPIX (1-based 129) is 0-based index 128
    ra, dec = g.pix2world(128.0, 128.0)
    assert angular_separation(ra, dec, RA0, DEC0) < 1e-9 * ARCSEC
    # moving +x (to the right) decreases RA (West) with CDELT1 < 0
    ra1, _ = g.pix2world(129.0, 128.0)
    assert ((ra1 - RA0 + np.pi) % (2 * np.pi) - np.pi) < 0
    x = np.array([3.3, 100.7, 250.1])
    y = np.array([7.9, 128.2, 10.0])
    ra, dec = g.pix2world(x, y)
    x2, y2 = g.world2pix(ra, dec)
    assert np.allclose(x, x2, atol=1e-9) and np.allclose(y, y2, atol=1e-9)
