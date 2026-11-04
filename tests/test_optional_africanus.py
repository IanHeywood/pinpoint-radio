"""Optional (needs codex-africanus): independent cross-check of conventions.

codex-africanus is NOT a dependency of pinpoint; it's used here only as an independent,
widely used DFT implementation of the wsclean source-list model. The checks are split so
that a failure points at one convention:

  * test_phase_convention   flat-spectrum point sources: phase sign, l/m orientation, w-term
  * test_spectral_models    sources at the phase centre (no phase): log- and ordinary polynomials
  * test_gaussian_agrees    Gaussian FWHM and position angle

Known, documented difference: africanus' log-polynomial uses natural logs, wsclean documents
base-10 (pinpoint follows wsclean). Curvature coefficients are converted with
c_i(ln) = c_i(log10) / ln(10)^(i-1) before calling africanus.

History: the first CI run of this file found that pinpoint's original phase sign was the
opposite of africanus' (and CASA's) convention; the default was changed to +1.
"""

import numpy as np
import pytest

pytest.importorskip("africanus")

try:
    from africanus.rime import wsclean_predict
except ImportError:  # pragma: no cover
    try:
        from africanus.rime.wsclean_predict import wsclean_predict
    except ImportError:
        pytest.skip("this codex-africanus version has no wsclean_predict", allow_module_level=True)

from pinpoint.constants import ARCSEC, DEG  # noqa: E402
from pinpoint.coords import radec_to_lmn  # noqa: E402
from pinpoint.predict import predict_reference, prepare_components  # noqa: E402
from pinpoint.skymodel import Component  # noqa: E402
from pinpoint.spectral import Spectrum  # noqa: E402

RA0, DEC0 = 1.0, -30 * DEG
FREQ = np.linspace(0.9e9, 1.7e9, 32)


def _africanus(comps, uvw):
    lm = np.array([radec_to_lmn(c.ra, c.dec, RA0, DEC0)[:2] for c in comps])
    stype = np.array(["GAUSSIAN" if c.type == "gauss" else "POINT" for c in comps])
    flux = np.array([c.spectrum.coeffs[0] for c in comps])
    nco = max(max(len(c.spectrum.coeffs) - 1 for c in comps), 1)
    coeffs = np.zeros((len(comps), nco))
    for i, c in enumerate(comps):
        cs = np.asarray(c.spectrum.coeffs[1:], dtype=float)
        if c.spectrum.kind == "logpol":  # base-10 -> natural-log coefficients
            cs = cs / np.log(10.0) ** np.arange(len(cs))
        coeffs[i, :len(cs)] = cs
    log_poly = np.array([c.spectrum.kind == "logpol" for c in comps])
    ref = np.array([c.spectrum.ref_freq for c in comps])
    shape = np.array([[c.maj, c.min, c.pa] for c in comps])
    out = wsclean_predict(uvw, lm, stype, flux, coeffs, log_poly, ref, shape, FREQ)
    return np.asarray(out).reshape(uvw.shape[0], FREQ.size, -1)[..., 0]


def _uvw(with_w):
    rng = np.random.default_rng(0)
    uvw = rng.uniform(-5000, 5000, (200, 3))
    if not with_w:
        uvw[:, 2] = 0.0
    return uvw


def _rel(ours, theirs):
    return np.max(np.abs(ours - theirs)) / np.max(np.abs(ours))


@pytest.mark.parametrize("with_w", [False, True])
def test_phase_convention(with_w):
    comps = [Component("a", "point", RA0 + 900 * ARCSEC, DEC0 - 400 * ARCSEC, Spectrum("logpol", 1.3e9, [2.0])),
             Component("b", "point", RA0 - 200 * ARCSEC, DEC0 + 50 * ARCSEC, Spectrum("logpol", 1.3e9, [1.0]))]
    uvw = _uvw(with_w)
    ours = predict_reference(uvw, FREQ, prepare_components(comps, RA0, DEC0, FREQ))
    theirs = _africanus(comps, uvw)
    hint = ""
    if _rel(np.conj(ours), theirs) < 1e-6:
        hint = " (theirs == conj(ours): the phase SIGN differs)"
    assert _rel(ours, theirs) < 1e-6, "phase mismatch: sign / l,m orientation" + (" / w-term" if with_w else "") + hint


@pytest.mark.parametrize("kind,coeffs", [("logpol", [2.0, -0.7]), ("logpol", [2.0, -0.7, -0.2, 0.05]),
                                         ("pol", [1.0, -0.3, 0.1])])
def test_spectral_models(kind, coeffs):
    comps = [Component("a", "point", RA0, DEC0, Spectrum(kind, 1.3e9, coeffs))]
    uvw = _uvw(True)
    ours = predict_reference(uvw, FREQ, prepare_components(comps, RA0, DEC0, FREQ))
    theirs = _africanus(comps, uvw)
    assert _rel(ours, theirs) < 1e-9, f"spectral model mismatch for {kind} {coeffs}"


def test_gaussian_agrees():
    comps = [Component("g", "gauss", RA0 + 60 * ARCSEC, DEC0, Spectrum("logpol", 1.3e9, [1.0, -0.7]),
                       maj=20 * ARCSEC, min=8 * ARCSEC, pa=50 * DEG)]
    uvw = _uvw(False)
    ours = predict_reference(uvw, FREQ, prepare_components(comps, RA0, DEC0, FREQ))
    theirs = _africanus(comps, uvw)
    assert _rel(ours, theirs) < 1e-6, "Gaussian mismatch: check PA (North through East) / FWHM convention"
