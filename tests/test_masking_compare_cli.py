"""Stages: model-image masking, model comparison, and the CLI wiring.

What these tests establish:
  * masking zeroes exactly the clean components inside source regions and nothing else;
  * `compare` reports a known injected offset (in mas) and flux ratio correctly;
  * the CLI parses every sub-command and `show` / `compare` run on real files.
"""

import numpy as np

from pinpoint.cli import build_parser, main
from pinpoint.compare import compare_models
from pinpoint.constants import ARCSEC, DEG
from pinpoint.coords import SinGeometry
from pinpoint.masking import zero_regions
from pinpoint.regions import parse_region_string
from pinpoint.skymodel import Component, SkyModel, Source
from pinpoint.spectral import Spectrum

RA0, DEC0 = 150 * DEG, -30 * DEG


def test_zero_regions_only_inside():
    g = SinGeometry(100, 100, RA0, DEC0, -1 * ARCSEC, 1 * ARCSEC, 51.0, 51.0)
    data = np.zeros((100, 100))
    data[50, 50] = 5.0  # inside
    data[50, 53] = 1.0  # inside (3" away)
    data[50, 70] = 2.0  # outside
    regs = parse_region_string('fk5\ncircle(150,-30,5")')
    out, nz, removed = zero_regions(data, g, regs)
    assert nz == 2 and np.isclose(removed, 6.0)
    assert out[50, 70] == 2.0 and out[50, 50] == 0 and out[50, 53] == 0
    assert data[50, 50] == 5.0  # input untouched


def _m(dra_mas=0.0, ddec_mas=0.0, flux=1.0):
    c = Component("c", "point", RA0 + dra_mas * 1e-3 * ARCSEC / np.cos(DEC0), DEC0 + ddec_mas * 1e-3 * ARCSEC,
                  Spectrum("logpol", 1e9, [flux, -0.7]), errors={"ra_arcsec": 0.001, "dec_arcsec": 0.001})
    return SkyModel([Source("s", [c], "X")])


def test_compare_reports_offsets():
    rows = compare_models(_m(), _m(3.0, -2.0, 1.01))
    r = rows[0]
    assert np.isclose(r["dRA_mas"], 3.0, atol=1e-4) and np.isclose(r["dDec_mas"], -2.0, atol=1e-4)
    assert np.isclose(r["flux_ratio"], 1.01)
    assert np.isclose(r["dRA_sigma"], 3.0, atol=1e-3)


def test_cli_parses_all_commands():
    p = build_parser()
    p.parse_args(["fit", "--images", "x", "--regions", "r", "--out", "o"])
    p.parse_args(["predict", "--model", "m", "--ms", "x.ms", "--mode", "add", "--time-smearing"])
    p.parse_args(["mask", "--images", "x", "--regions", "r", "--out-prefix", "y"])
    p.parse_args(["compare", "a", "b"])
    p.parse_args(["show", "a"])


def test_cli_show_and_compare(tmp_path, capsys):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    _m().save(a)
    _m(1.0, 0.0).save(b)
    main(["show", str(a)])
    main(["compare", str(a), str(b)])
    out = capsys.readouterr().out
    assert "s -> X" in out and "dRA +1.000 mas" in out
