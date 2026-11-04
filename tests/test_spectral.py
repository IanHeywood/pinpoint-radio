"""Stage: spectral models (pinpoint/spectral.py).

What these tests establish:
  * both spectral forms evaluate to the documented formulas (wsclean conventions);
  * fitting noiseless fluxes recovers the true coefficients;
  * fitting noisy fluxes gives errors that are statistically honest (pull test);
  * band integration matters for curved spectra and the band-integrated fit removes
    that bias;
  * AIC model selection picks the right number of terms.
"""

import numpy as np
import pytest

from pinpoint.spectral import band_average, evaluate, fit_spectrum, fit_spectrum_auto

NU0 = 1.284e9
NU = np.linspace(0.9e9, 1.67e9, 12)


def test_pol_formula():
    x = NU / NU0 - 1
    assert np.allclose(evaluate("pol", [2.0, -1.5, 0.3], NU0, NU), 2.0 - 1.5 * x + 0.3 * x**2)


def test_logpol_formula_is_power_law():
    # 10^(alpha*log10(nu/nu0)) == (nu/nu0)^alpha
    assert np.allclose(evaluate("logpol", [3.0, -0.7], NU0, NU), 3.0 * (NU / NU0) ** -0.7)


def test_logpol_value_at_ref_freq():
    assert np.isclose(evaluate("logpol", [5.0, -0.7, -0.2], NU0, NU0), 5.0)


@pytest.mark.parametrize("kind,coeffs", [("logpol", [10.0, -0.8, -0.3]), ("pol", [4.0, -2.0, 0.5])])
def test_noiseless_fit_recovers_truth(kind, coeffs):
    flux = evaluate(kind, coeffs, NU0, NU)
    f = fit_spectrum(NU, flux, np.full_like(NU, 1e-3), kind=kind, nterms=3, ref_freq=NU0)
    assert np.allclose(f.spectrum.coeffs, coeffs, rtol=1e-6, atol=1e-8)


def test_error_bars_are_honest():
    rng = np.random.default_rng(42)
    truth = [1.0, -0.7]
    pulls = []
    for _ in range(300):
        sig = np.full_like(NU, 0.02)
        flux = evaluate("logpol", truth, NU0, NU) + rng.normal(0, 0.02, NU.size)
        f = fit_spectrum(NU, flux, sig, kind="logpol", nterms=2, ref_freq=NU0)
        pulls.append((f.spectrum.coeffs[1] - truth[1]) / f.spectrum.coeff_errors[1])
    pulls = np.array(pulls)
    assert abs(np.mean(pulls)) < 0.2
    assert 0.85 < np.std(pulls) < 1.15


def test_band_integration_removes_bias():
    truth = [10.0, -0.7, -1.5]  # strongly curved
    bw = np.full(NU.size, np.diff(NU)[0])
    flux_avg = band_average("logpol", truth, NU0, NU, bw)  # what an image of a band measures
    sig = np.full_like(NU, 1e-4)
    naive = fit_spectrum(NU, flux_avg, sig, kind="logpol", nterms=3, ref_freq=NU0)
    integ = fit_spectrum(NU, flux_avg, sig, kind="logpol", nterms=3, ref_freq=NU0,
                         bandwidth=bw, band_integrate=True)
    err_naive = np.max(np.abs(naive.spectrum(NU) - evaluate("logpol", truth, NU0, NU)))
    err_integ = np.max(np.abs(integ.spectrum(NU) - evaluate("logpol", truth, NU0, NU)))
    assert err_naive > 1e-4  # bias exists at this curvature
    assert err_integ < 1e-3 * err_naive


def test_aic_selects_power_law():
    rng = np.random.default_rng(7)
    flux = evaluate("logpol", [2.0, -0.7], NU0, NU) + rng.normal(0, 1e-3, NU.size)
    f = fit_spectrum_auto(NU, flux, np.full_like(NU, 1e-3), kind="logpol", max_terms=4, ref_freq=NU0)
    assert len(f.spectrum.coeffs) == 2
