"""Stage: visibility-domain refinement with smearing (pinpoint/fit_uv.py).

Synthetic data mimic the e-MERLIN case: 7 antennas over ~217 km, a long track at
Dec +65.8, 8 s integrations, a bright source a few arcmin from the phase centre (so time
and bandwidth smearing reduce long-baseline amplitudes substantially), plus a
rest-of-field source present in DATA and in MODEL_DATA.

What these tests establish:
  * starting from a model 30 mas and 40% off (as a smeared image fit might give), the fit
    recovers the true position, flux and spectral index within its quoted errors;
  * the quoted errors are honest (pull distributions of unit width over many noise draws);
  * leaving smearing out of the model biases the flux low by many sigma - i.e. modelling
    smearing is what makes the answer right;
  * a Gaussian source's size and PA are recovered from smeared data;
  * the "elongation aligned with smearing" warning fires when smearing is left unmodelled,
    and stays quiet when it is modelled;
  * flagged data really are ignored (corrupting flagged samples changes nothing);
  * sub-sampling rows / channels gives consistent answers;
  * the rest-of-field column is subtracted; fixed parameter groups stay fixed;
  * the data-vs-model diagnostic sits at ~1 for a correct model.
"""

import numpy as np
import pytest

from pinpoint.constants import ARCSEC, DEG
from pinpoint.coords import lmn_to_radec, radec_to_lmn
from pinpoint.fit_uv import UVFitOptions, fit_uv, format_uv_report
from pinpoint.skymodel import Component, SkyModel, Source
from pinpoint.spectral import Spectrum
from pinpoint.testing.synth_ms import fill_column, make_emerlin_like_ms

QUIET = dict(log=lambda *a: None)


def scene(sigma=0.05, seed=3, truth_kind="point", shape=None, time_smearing=True, freq_smearing=True,
          chan_width=1e6, ntime=48):
    ms = make_emerlin_like_ms(chan_width=chan_width, ntime=ntime)
    ra0, dec0 = ms.fields[0]
    ra, dec = lmn_to_radec(150 * ARCSEC, -100 * ARCSEC, ra0, dec0)  # ~3' off-axis
    truth = Component("b", truth_kind, float(ra), float(dec), Spectrum("logpol", 1.5e9, [0.2, -0.3]),
                      **(shape or {}))
    rest = Component("r", "point", ra0, dec0 + 5 * ARCSEC, Spectrum("logpol", 1.5e9, [0.5]))
    fill_column(ms, [truth, rest], "DATA", sigma=sigma, rng=np.random.default_rng(seed),
                time_smearing=time_smearing, freq_smearing=freq_smearing)
    fill_column(ms, [rest], "MODEL_DATA", time_smearing=time_smearing, freq_smearing=freq_smearing)
    return ms, truth


def start_model(truth, kind=None, dE_mas=30.0, dN_mas=20.0, flux=0.12, shape=None):
    ra, dec = lmn_to_radec(dE_mas * 1e-3 * ARCSEC, dN_mas * 1e-3 * ARCSEC, truth.ra, truth.dec)
    c = Component("b_c0", kind or truth.type, float(ra), float(dec), Spectrum("logpol", 1.5e9, [flux, 0.0]),
                  **(shape or {}))
    return SkyModel([Source("b", [c], "DIR1_DATA")])


def offsets_mas(c, truth):
    l, m, _ = radec_to_lmn(c.ra, c.dec, truth.ra, truth.dec)
    return l / ARCSEC * 1e3, m / ARCSEC * 1e3


def opts(**kw):
    return UVFitOptions(data_column="DATA", **kw)


def test_recovers_truth_from_biased_start():
    ms, truth = scene()
    r = fit_uv(ms, start_model(truth), opts(), **QUIET)
    c = r.model.sources[0].components[0]
    dE, dN = offsets_mas(c, truth)
    eE, eN = c.errors["ra_arcsec"] * 1e3, c.errors["dec_arcsec"] * 1e3
    assert abs(dE) < 4 * eE and abs(dN) < 4 * eN
    assert abs(c.spectrum.coeffs[0] - 0.2) < 4 * c.spectrum.coeff_errors[0]
    assert abs(c.spectrum.coeffs[1] + 0.3) < 4 * c.spectrum.coeff_errors[1]
    assert 0.9 < r.chi2_red < 1.1 and r.converged
    assert r.model.sources[0].column == "DIR1_DATA"
    assert "moved" in format_uv_report(r)


def test_errors_are_honest():
    pulls = []
    for seed in range(25):
        ms, truth = scene(seed=100 + seed)
        c = fit_uv(ms, start_model(truth), opts(), **QUIET).model.sources[0].components[0]
        dE, dN = offsets_mas(c, truth)
        pulls.append([dE / (c.errors["ra_arcsec"] * 1e3), dN / (c.errors["dec_arcsec"] * 1e3),
                      (c.spectrum.coeffs[0] - 0.2) / c.spectrum.coeff_errors[0]])
    s = np.std(pulls, axis=0)
    assert np.all(s > 0.7) and np.all(s < 1.35), s
    assert np.all(np.abs(np.mean(pulls, axis=0)) < 0.6)


def test_ignoring_smearing_biases_flux():
    ms, truth = scene()
    good = fit_uv(ms, start_model(truth), opts(), **QUIET).model.sources[0].components[0]
    bad = fit_uv(ms, start_model(truth), opts(time_smearing=False, freq_smearing=False), **QUIET)
    cb = bad.model.sources[0].components[0]
    assert abs(good.spectrum.coeffs[0] - 0.2) < 4 * good.spectrum.coeff_errors[0]
    assert cb.spectrum.coeffs[0] < 0.2 - 10 * cb.spectrum.coeff_errors[0]  # biased low, many sigma
    assert bad.chi2_red > good.spectrum.coeffs[0] * 0 + 1.02  # and visibly worse fit


def test_gaussian_with_smearing():
    # PA 75 deg: well away from both smearing directions at this position (radial 123.7, tangential 33.7)
    shape = dict(maj=60e-3 * ARCSEC, min=25e-3 * ARCSEC, pa=75 * DEG)
    ms, truth = scene(sigma=0.01, truth_kind="gauss", shape=shape)
    start = start_model(truth, shape=dict(maj=40e-3 * ARCSEC, min=40e-3 * ARCSEC, pa=0.0))
    r = fit_uv(ms, start, opts(), **QUIET)
    c = r.model.sources[0].components[0]
    assert abs(c.maj / ARCSEC * 1e3 - 60) < 4 * c.errors["maj_arcsec"] * 1e3 + 1.0
    assert abs(c.min / ARCSEC * 1e3 - 25) < 4 * c.errors["min_arcsec"] * 1e3 + 1.0
    dpa = (np.rad2deg(c.pa) - 75 + 90) % 180 - 90
    assert abs(dpa) < 4 * c.errors["pa_deg"] + 1.0
    assert not r.warnings


def test_alignment_warning_fires_for_coincident_real_structure_too():
    # The warning is a heuristic: a real source elongated along a smearing direction triggers it.
    shape = dict(maj=60e-3 * ARCSEC, min=25e-3 * ARCSEC, pa=33.7 * DEG)  # tangential here
    ms, truth = scene(sigma=0.01, truth_kind="gauss", shape=shape)
    start = start_model(truth, shape=dict(maj=40e-3 * ARCSEC, min=40e-3 * ARCSEC, pa=0.0))
    r = fit_uv(ms, start, opts(), **QUIET)
    assert any("tangential" in w and "coincidence" in w for w in r.warnings)


def test_alignment_warning_when_smearing_unmodelled():
    # strong bandwidth smearing only (wide channels) of a POINT source, fitted as a Gaussian
    ms, truth = scene(sigma=0.002, time_smearing=False, freq_smearing=True, chan_width=8e6)
    start = start_model(truth, kind="gauss", shape=dict(maj=40e-3 * ARCSEC, min=40e-3 * ARCSEC, pa=0.0))
    bad = fit_uv(ms, start, opts(time_smearing=False, freq_smearing=False), **QUIET)
    assert any("radial" in w for w in bad.warnings), bad.warnings
    good = fit_uv(ms, start, opts(time_smearing=False, freq_smearing=True), **QUIET)
    assert not good.warnings, good.warnings


def test_flagged_data_ignored():
    ms, truth = scene(seed=11)
    rng = np.random.default_rng(0)
    rows = rng.choice(ms.nrows(), size=ms.nrows() // 10, replace=False)
    for r in rows:
        ms.data_cols["FLAG"][r][:] = True
    a = fit_uv(ms, start_model(truth), opts(), **QUIET).model.sources[0].components[0]
    for r in rows:  # corrupt only flagged samples
        ms.data_cols["DATA"][r][:] = 1e6 + 1e6j
    ms.cols["FLAG_ROW"][rows[:5]] = True
    b = fit_uv(ms, start_model(truth), opts(), **QUIET).model.sources[0].components[0]
    assert abs(a.ra - b.ra) < 1e-6 * ARCSEC and abs(a.dec - b.dec) < 1e-6 * ARCSEC
    assert np.isclose(a.spectrum.coeffs[0], b.spectrum.coeffs[0], rtol=1e-6)


def test_subsampling_consistent():
    ms, truth = scene(seed=21, ntime=96)
    full = fit_uv(ms, start_model(truth), opts(), **QUIET).model.sources[0].components[0]
    sub = fit_uv(ms, start_model(truth), opts(chan_step=2, row_step=2), **QUIET).model.sources[0].components[0]
    dE, dN = offsets_mas(sub, full)
    assert abs(dE) < 3 * sub.errors["ra_arcsec"] * 1e3 and abs(dN) < 3 * sub.errors["dec_arcsec"] * 1e3
    assert sub.errors["ra_arcsec"] > full.errors["ra_arcsec"]  # less data -> larger errors


def test_rest_of_field_subtraction_matters():
    ms, truth = scene(seed=5)
    with_sub = fit_uv(ms, start_model(truth), opts(), **QUIET)
    without = fit_uv(ms, start_model(truth), opts(subtract_column=None), **QUIET)
    assert without.chi2_red > 2 * with_sub.chi2_red  # unsubtracted 0.5 Jy source left in the data


def test_fixed_groups_stay_fixed():
    ms, truth = scene()
    start = start_model(truth)
    c0 = start.sources[0].components[0]
    r = fit_uv(ms, start, opts(fit_position=False), **QUIET)
    c = r.model.sources[0].components[0]
    assert c.ra == c0.ra and c.dec == c0.dec
    r = fit_uv(ms, start, opts(fit_spectrum=False), **QUIET)
    assert r.model.sources[0].components[0].spectrum.coeffs == c0.spectrum.coeffs
    with pytest.raises(ValueError):
        fit_uv(ms, start, opts(fit_position=False, fit_spectrum=False), **QUIET)


def test_diagnostic_near_unity():
    ms, truth = scene(sigma=0.005)
    r = fit_uv(ms, start_model(truth), opts(), **QUIET)
    ratios = np.array([q for *_, q in r.diagnostics])
    assert len(ratios) >= 5 and np.all(np.abs(ratios - 1) < 0.02), ratios


def test_missing_columns_and_fields():
    ms, truth = scene()
    with pytest.raises(ValueError, match="data column"):
        fit_uv(ms, start_model(truth), UVFitOptions(data_column="NOPE"), **QUIET)
    with pytest.raises(ValueError, match="subtract column"):
        fit_uv(ms, start_model(truth), opts(subtract_column="NOPE"), **QUIET)
