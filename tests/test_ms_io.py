"""Stage: writing into Measurement Set columns (pinpoint/ms_io.py), using an in-memory MS.

What these tests establish:
  * overwrite mode writes exactly the predicted visibilities (checked independently
    against the reference predictor) into the requested column, in every spectral
    window (different channel counts) and correlation setup (linear and circular);
  * add / subtract modes add to / subtract from what is already there;
  * a new column is created when missing, and rows of non-target fields in a fresh
    column are zero-filled (never left uninitialised);
  * the DATA column itself is protected unless explicitly allowed;
  * sources mapped to different columns end up in different columns;
  * chunking doesn't change the answer;
  * DATA is never modified when writing a model column.
"""

import numpy as np
import pytest

from pinpoint.constants import ARCSEC, DEG
from pinpoint.ms_io import predict_into_ms
from pinpoint.predict import predict_reference, prepare_components
from pinpoint.skymodel import Component, SkyModel, Source
from pinpoint.spectral import Spectrum
from pinpoint.testing.synth_ms import make_inmemory_ms

RA0, DEC0 = 1.0, -30 * DEG


def model(col_a="DIR1_DATA", col_b=None):
    a = Component("a", "point", RA0 + 120 * ARCSEC, DEC0 - 60 * ARCSEC, Spectrum("logpol", 1e9, [3.0, -0.8]))
    b = Component("b", "gauss", RA0 - 300 * ARCSEC, DEC0 + 30 * ARCSEC, Spectrum("pol", 1e9, [1.0, 0.1]),
                  maj=5 * ARCSEC, min=2 * ARCSEC, pa=10 * DEG)
    return SkyModel([Source("A", [a], col_a), Source("B", [b], col_b)])


def two_spw_ms(**kw):
    spws = [(np.linspace(0.9e9, 1.0e9, 16), np.full(16, 6.25e6)),
            (np.linspace(1.4e9, 1.3e9, 7), np.full(7, -16.0e6))]  # different nchan, descending
    pols = [[9, 10, 11, 12], [5, 8]]
    ddids = [(0, 0), (1, 1)]
    return make_inmemory_ms(spws=spws, pols=pols, ddids=ddids, **kw)


def expected(ms, comps, rows, weights, freqs, fieldc=(RA0, DEC0)):
    ca = prepare_components(comps, fieldc[0], fieldc[1], freqs)
    v = predict_reference(ms.cols["UVW"][rows], freqs, ca)
    return v[:, :, None] * np.asarray(weights)[None, None, :]


def test_overwrite_all_spws_and_correlations():
    ms = two_spw_ms()
    data_before = [d.copy() for d in ms.data_cols["DATA"]]
    m = model(col_a="DIR1_DATA", col_b="DIR1_DATA")
    rep = predict_into_ms(ms, m, log=lambda *a: None, chunk_rows=37)
    assert rep.columns_created == ["DIR1_DATA"]
    comps = list(m.components())
    for d, (freqs, wts) in enumerate([(ms.spws[0][0], [1, 0, 0, 1]), (ms.spws[1][0], [1, 1])]):
        rows = np.flatnonzero(ms.cols["DATA_DESC_ID"] == d)
        got = ms.read("DIR1_DATA", rows)
        exp = expected(ms, comps, rows, wts, freqs)
        assert np.max(np.abs(got - exp)) < 1e-5 * np.max(np.abs(exp))
    for a, b in zip(data_before, ms.data_cols["DATA"]):
        assert np.array_equal(a, b)


def test_add_and_subtract_modes():
    ms = two_spw_ms()
    m = model(col_a="MODEL_DATA", col_b="MODEL_DATA")
    ms.add_data_column("MODEL_DATA", fill=0.0)
    base = [np.full(x.shape, 0.25 - 0.5j, np.complex64) for x in ms.data_cols["MODEL_DATA"]]
    ms.data_cols["MODEL_DATA"] = [b.copy() for b in base]
    predict_into_ms(ms, m, mode="add", log=lambda *a: None)
    added = [x.copy() for x in ms.data_cols["MODEL_DATA"]]
    ms.data_cols["MODEL_DATA"] = [b.copy() for b in base]
    predict_into_ms(ms, m, mode="subtract", log=lambda *a: None)
    sub = ms.data_cols["MODEL_DATA"]
    for b, a, s in zip(base, added, sub):
        # add - base == base - subtract == the model
        assert np.allclose(a - b, b - s, atol=1e-5)
        assert np.max(np.abs(a - b)) > 0.1


def test_separate_columns_per_source():
    ms = two_spw_ms()
    m = model(col_a="DIR1_DATA", col_b="DIR2_DATA")
    predict_into_ms(ms, m, log=lambda *a: None)
    rows = np.flatnonzero(ms.cols["DATA_DESC_ID"] == 0)
    freqs = ms.spws[0][0]
    a, b = list(m.components())
    assert np.allclose(ms.read("DIR1_DATA", rows), expected(ms, [a], rows, [1, 0, 0, 1], freqs), atol=1e-5)
    assert np.allclose(ms.read("DIR2_DATA", rows), expected(ms, [b], rows, [1, 0, 0, 1], freqs), atol=1e-5)


def test_default_and_override_columns():
    ms = two_spw_ms()
    m = model(col_a=None, col_b=None)
    predict_into_ms(ms, m, default_column="MODEL_DATA", log=lambda *a: None)
    assert ms.has_column("MODEL_DATA")
    predict_into_ms(ms, model(), override_column="OTHER", log=lambda *a: None)
    assert ms.has_column("OTHER") and not ms.has_column("DIR1_DATA")


def test_data_column_is_protected():
    ms = two_spw_ms()
    with pytest.raises(ValueError, match="refusing"):
        predict_into_ms(ms, model(), override_column="DATA", log=lambda *a: None)


def test_multi_field_requires_choice_and_zero_fills():
    ms = make_inmemory_ms(fields=[(RA0, DEC0), (RA0 + 0.01, DEC0)])
    with pytest.raises(ValueError, match="several fields"):
        predict_into_ms(ms, model(), log=lambda *a: None)
    rep = predict_into_ms(ms, model("DIR1_DATA", "DIR1_DATA"), field_ids=[0], log=lambda *a: None)
    f1 = np.flatnonzero(ms.cols["FIELD_ID"] == 1)
    assert rep.rows_zeroed == len(f1)
    assert np.all(ms.read("DIR1_DATA", f1) == 0)  # not NaN garbage
    f0 = np.flatnonzero(ms.cols["FIELD_ID"] == 0)
    assert np.all(np.isfinite(ms.read("DIR1_DATA", f0)))


def test_chunk_size_invariance():
    m = model("X", "X")
    out = []
    for n in (1, 13, 10_000):
        ms = two_spw_ms()
        predict_into_ms(ms, m, chunk_rows=n, freq_smearing=True, time_smearing=True, log=lambda *a: None)
        out.append(np.concatenate([x.ravel() for x in ms.data_cols["X"]]))
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[1], out[2])
