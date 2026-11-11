"""Optional (needs python-casacore): the real-MS adapter.

What these tests establish:
  * metadata (phase centre, spectral windows, correlation types) is read correctly;
  * predicting into a real casacore MS gives bit-for-bit the same column contents as
    the in-memory MS used in test_ms_io.py (so all those tests transfer);
  * a new column is created with the right shapes and DATA is untouched;
  * add mode works on the real table.
"""

import numpy as np
import pytest

pytest.importorskip("casacore.tables")

from casacore.tables import table  # noqa: E402

from pinpoint.constants import ARCSEC, DEG  # noqa: E402
from pinpoint.ms_io import CasacoreMS, predict_into_ms  # noqa: E402
from pinpoint.skymodel import Component, SkyModel, Source  # noqa: E402
from pinpoint.spectral import Spectrum  # noqa: E402
from pinpoint.testing.casacore_ms import write_casacore_ms  # noqa: E402
from pinpoint.testing.synth_ms import make_inmemory_ms  # noqa: E402

RA0, DEC0 = 1.0, -30 * DEG


def _setup(tmp_path):
    spws = [(np.linspace(0.9e9, 1.0e9, 16), np.full(16, 6.25e6)),
            (np.linspace(1.4e9, 1.3e9, 7), np.full(7, -16.0e6))]
    mem = make_inmemory_ms(spws=spws, pols=[[9, 10, 11, 12], [5, 8]], ddids=[(0, 0), (1, 1)], ra0=RA0, dec0=DEC0)
    path = write_casacore_ms(tmp_path / "t.ms", mem)
    a = Component("a", "point", RA0 + 120 * ARCSEC, DEC0 - 60 * ARCSEC, Spectrum("logpol", 1e9, [3.0, -0.8]))
    b = Component("b", "gauss", RA0 - 300 * ARCSEC, DEC0 + 30 * ARCSEC, Spectrum("pol", 1e9, [1.0, 0.1]),
                  maj=5 * ARCSEC, min=2 * ARCSEC, pa=10 * DEG)
    return mem, path, SkyModel([Source("A", [a, b], "DIR1_DATA")])


def test_metadata(tmp_path):
    mem, path, _ = _setup(tmp_path)
    with CasacoreMS(path, readonly=True) as ms:
        meta = ms.meta()
    exp = mem.meta()
    assert np.allclose(meta.field_phase_dir, exp.field_phase_dir)
    assert meta.ddid_spw == exp.ddid_spw and meta.ddid_pol == exp.ddid_pol
    for a, b in zip(meta.spw_freqs, exp.spw_freqs):
        assert np.array_equal(a, b)
    assert meta.pol_corr_types == exp.pol_corr_types


def test_matches_inmemory(tmp_path):
    mem, path, model = _setup(tmp_path)
    with table(path, ack=False) as t:
        data_before = [t.getcell("DATA", r) for r in range(t.nrows())]
    with CasacoreMS(path) as ms:
        predict_into_ms(ms, model, chunk_rows=17, log=lambda *a: None)
    predict_into_ms(mem, model, chunk_rows=17, log=lambda *a: None)
    with table(path, ack=False) as t:
        assert "DIR1_DATA" in t.colnames()
        for r in range(t.nrows()):
            got = t.getcell("DIR1_DATA", r)
            assert got.shape == mem.data_cols["DIR1_DATA"][r].shape
            assert np.array_equal(got, mem.data_cols["DIR1_DATA"][r])
            assert np.array_equal(t.getcell("DATA", r), data_before[r])


def test_add_mode_on_real_table(tmp_path):
    mem, path, model = _setup(tmp_path)
    with CasacoreMS(path) as ms:
        predict_into_ms(ms, model, log=lambda *a: None)
        predict_into_ms(ms, model, mode="add", log=lambda *a: None)
    predict_into_ms(mem, model, log=lambda *a: None)
    with table(path, ack=False) as t:
        for r in range(0, t.nrows(), 50):
            assert np.allclose(t.getcell("DIR1_DATA", r), 2 * mem.data_cols["DIR1_DATA"][r], atol=1e-5)


def test_fit_uv_matches_inmemory(tmp_path):
    """fit-uv reads a real MS (FLAG, WEIGHT, EXPOSURE via casacore) exactly as the in-memory one."""
    from pinpoint.coords import lmn_to_radec
    from pinpoint.fit_uv import UVFitOptions, fit_uv
    from pinpoint.testing.synth_ms import fill_column, make_emerlin_like_ms

    mem = make_emerlin_like_ms(ntime=12)
    ra0, dec0 = mem.fields[0]
    ra, dec = lmn_to_radec(150 * ARCSEC, -100 * ARCSEC, ra0, dec0)
    truth = Component("b", "point", float(ra), float(dec), Spectrum("logpol", 1.5e9, [0.2, -0.3]))
    fill_column(mem, [truth], "DATA", sigma=0.05, rng=np.random.default_rng(1))
    mem.data_cols["FLAG"][3][:] = True
    path = write_casacore_ms(tmp_path / "e.ms", mem)
    sra, sdec = lmn_to_radec(30e-3 * ARCSEC, 0.0, truth.ra, truth.dec)
    start = SkyModel([Source("b", [Component("b_c0", "point", float(sra), float(sdec),
                                             Spectrum("logpol", 1.5e9, [0.12, 0.0]))], "DIR1_DATA")])
    o = UVFitOptions(data_column="DATA", subtract_column=None)
    a = fit_uv(mem, start, o, log=lambda *x: None).model.sources[0].components[0]
    with CasacoreMS(path, readonly=True) as ms:
        b = fit_uv(ms, start, o, log=lambda *x: None).model.sources[0].components[0]
    assert abs(a.ra - b.ra) < 1e-5 * ARCSEC and abs(a.dec - b.dec) < 1e-5 * ARCSEC
    assert np.isclose(a.spectrum.coeffs[0], b.spectrum.coeffs[0], rtol=1e-5)
