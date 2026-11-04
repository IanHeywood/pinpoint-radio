"""Stage: sky-model file (pinpoint/skymodel.py).

What these tests establish:
  * saving and re-loading the YAML file loses nothing (positions to < 1 micro-arcsec);
  * the file is in human units (degrees, arcsec) so you can read/edit it;
  * column mapping: each source goes to its own column, a CLI override wins.
"""

import numpy as np
import yaml

from pinpoint.constants import ARCSEC, DEG
from pinpoint.skymodel import Component, SkyModel, Source
from pinpoint.spectral import Spectrum


def _model():
    c1 = Component("a0", "point", 10.123456789 * DEG, -30.987654321 * DEG,
                   Spectrum("logpol", 1.3e9, [12.0, -0.7, -0.1]))
    c2 = Component("a1", "gauss", 10.124 * DEG, -30.988 * DEG,
                   Spectrum("pol", 1.3e9, [1.0, 0.2]), maj=3 * ARCSEC, min=1 * ARCSEC, pa=40 * DEG)
    c3 = Component("b0", "point", 11.0 * DEG, -31.0 * DEG, Spectrum("logpol", 1.3e9, [2.0]))
    return SkyModel([Source("A", [c1, c2], "DIR1_DATA"), Source("B", [c3], None)])


def test_round_trip(tmp_path):
    m = _model()
    p = tmp_path / "m.yaml"
    m.save(p)
    m2 = SkyModel.load(p)
    for a, b in zip(m.components(), m2.components()):
        assert a.type == b.type and a.name == b.name
        assert abs(a.ra - b.ra) < 1e-6 * ARCSEC and abs(a.dec - b.dec) < 1e-6 * ARCSEC
        assert np.isclose(a.maj, b.maj) and np.isclose(a.min, b.min) and np.isclose(a.pa, b.pa)
        assert a.spectrum.kind == b.spectrum.kind
        assert np.allclose(a.spectrum.coeffs, b.spectrum.coeffs)


def test_human_units(tmp_path):
    p = tmp_path / "m.yaml"
    _model().save(p)
    d = yaml.safe_load(open(p))
    c = d["sources"][0]["components"][1]
    assert np.isclose(c["maj_arcsec"], 3.0) and np.isclose(c["pa_deg"], 40.0)
    assert np.isclose(c["ra_deg"], 10.124)


def test_column_mapping():
    m = _model()
    cols = m.by_column(default_column="MODEL_DATA")
    assert sorted(cols) == ["DIR1_DATA", "MODEL_DATA"]
    assert [c.name for c in cols["DIR1_DATA"]] == ["a0", "a1"]
    cols = m.by_column(override="X")
    assert list(cols) == ["X"] and len(cols["X"]) == 3


def test_minor_larger_than_major_is_swapped():
    c = Component("x", "gauss", 0.0, 0.0, Spectrum("pol", 1e9, [1]), maj=1 * ARCSEC, min=2 * ARCSEC, pa=0.0)
    assert c.maj > c.min and np.isclose(c.pa, np.pi / 2)
