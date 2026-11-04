"""Optional (needs codex-africanus): independent cross-check of conventions.

codex-africanus is NOT a dependency of pinpoint; it's used here only as an independent,
widely used DFT implementation of the wsclean source-list model. Agreement means our
phase sign, l/m orientation, spectral polynomials and Gaussian PA convention match
what the rest of the ecosystem uses.

If the point-source test fails, suspect the phase sign / l,m orientation.
If only the Gaussian test fails, suspect the Gaussian PA or FWHM convention.
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
    nco = max(len(c.spectrum.coeffs) - 1 for c in comps) or 1
    coeffs = np.zeros((len(comps), nco))
    for i, c in enumerate(comps):
        coeffs[i, :len(c.spectrum.coeffs) - 1] = c.spectrum.coeffs[1:]
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


@pytest.mark.parametrize("with_w", [False, True])
def test_point_sources_agree(with_w):
    comps = [Component("a", "point", RA0 + 900 * ARCSEC, DEC0 - 400 * ARCSEC, Spectrum("logpol", 1.3e9, [2.0, -0.7, -0.2])),
             Component("b", "point", RA0 - 200 * ARCSEC, DEC0 + 50 * ARCSEC, Spectrum("pol", 1.3e9, [1.0, -0.3, 0.1]))]
    uvw = _uvw(with_w)
    ours = predict_reference(uvw, FREQ, prepare_components(comps, RA0, DEC0, FREQ))
    theirs = _africanus(comps, uvw)
    assert np.max(np.abs(ours - theirs)) / np.max(np.abs(ours)) < 1e-6, \
        "point-source mismatch: check phase sign / l,m orientation" + (" / w-term" if with_w else "")


def test_gaussian_agrees():
    comps = [Component("g", "gauss", RA0 + 60 * ARCSEC, DEC0, Spectrum("logpol", 1.3e9, [1.0, -0.7]),
                       maj=20 * ARCSEC, min=8 * ARCSEC, pa=50 * DEG)]
    uvw = _uvw(False)
    ours = predict_reference(uvw, FREQ, prepare_components(comps, RA0, DEC0, FREQ))
    theirs = _africanus(comps, uvw)
    assert np.max(np.abs(ours - theirs)) / np.max(np.abs(ours)) < 1e-6, \
        "Gaussian mismatch: check PA (North through East) / FWHM convention"
