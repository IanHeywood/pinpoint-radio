"""Spectral models.

Two forms, matching the conventions documented for wsclean source lists:

  kind = "pol"     ordinary polynomial
      S(nu) = c0 + c1 x + c2 x^2 + ...,      x = nu/nu0 - 1

  kind = "logpol"  logarithmic polynomial (base-10 logs, as wsclean documents LogarithmicSI)
      S(nu) = c0 * 10^(c1 y + c2 y^2 + ...),   y = log10(nu/nu0)
      i.e. c1 is the usual spectral index, c2 the curvature.

coeffs[0] is always the flux density (Jy) at the reference frequency nu0.

Note: not every package uses base 10. codex-africanus uses natural logs, so its
curvature terms relate to ours by c_i(ln) = c_i(log10) / ln(10)^(i-1). The spectral
index c1 is the same in both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

KINDS = ("pol", "logpol")


@dataclass
class Spectrum:
    kind: str
    ref_freq: float  # Hz
    coeffs: list[float]
    coeff_errors: list[float] | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown spectral kind {self.kind!r}; expected one of {KINDS}")
        self.coeffs = [float(c) for c in self.coeffs]

    def __call__(self, nu):
        return evaluate(self.kind, self.coeffs, self.ref_freq, nu)


def evaluate(kind, coeffs, nu0, nu):
    nu = np.asarray(nu, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if kind == "pol":
        x = nu / nu0 - 1.0
        out = np.zeros_like(nu) + coeffs[0]
        for i, c in enumerate(coeffs[1:], start=1):
            out = out + c * x**i
        return out
    if kind == "logpol":
        y = np.log10(nu / nu0)
        expo = np.zeros_like(nu)
        for i, c in enumerate(coeffs[1:], start=1):
            expo = expo + c * y**i
        return coeffs[0] * 10.0**expo
    raise ValueError(kind)


def band_average(kind, coeffs, nu0, nu_centre, bandwidth, nsub=16):
    """Mean of S(nu) over each band [centre - bw/2, centre + bw/2] (uniform weighting)."""
    nu_centre = np.atleast_1d(np.asarray(nu_centre, dtype=np.float64))
    bandwidth = np.broadcast_to(np.abs(np.asarray(bandwidth, dtype=np.float64)), nu_centre.shape)
    t = (np.arange(nsub) + 0.5) / nsub - 0.5
    nus = nu_centre[:, None] + bandwidth[:, None] * t[None, :]
    return evaluate(kind, coeffs, nu0, nus).mean(axis=1)


@dataclass
class SpectralFit:
    spectrum: Spectrum
    chi2: float
    dof: int
    aic: float
    covariance: np.ndarray = field(repr=False)


def fit_spectrum(nu, flux, sigma, kind="logpol", nterms=2, ref_freq=None,
                 bandwidth=None, band_integrate=False):
    """Weighted least-squares fit of a spectral model to per-band fluxes.

    nu, flux, sigma : per-band centre frequency (Hz), flux (Jy), 1-sigma error (Jy)
    nterms          : number of coefficients (1 = flat, 2 = power law / linear, ...)
    band_integrate  : model each band as the average of S(nu) across its bandwidth
                      instead of S at the band centre.
    """
    nu = np.asarray(nu, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if nterms < 1:
        raise ValueError("nterms must be >= 1")
    if nterms > len(nu):
        raise ValueError(f"nterms={nterms} exceeds number of bands ({len(nu)})")
    if ref_freq is None:
        ref_freq = float(np.mean(nu))
    if band_integrate and bandwidth is None:
        raise ValueError("band_integrate requires bandwidth")

    def model(p):
        if band_integrate:
            return band_average(kind, p, ref_freq, nu, bandwidth)
        return evaluate(kind, p, ref_freq, nu)

    # Linear initial guess
    x0 = np.zeros(nterms)
    w = 1.0 / sigma**2
    if kind == "pol":
        x = nu / ref_freq - 1.0
        A = np.vander(x, nterms, increasing=True)
        x0 = np.linalg.lstsq(A * np.sqrt(w)[:, None], flux * np.sqrt(w), rcond=None)[0]
    else:
        good = flux > 0
        if good.sum() >= nterms:
            y = np.log10(nu[good] / ref_freq)
            A = np.vander(y, nterms, increasing=True)
            sol = np.linalg.lstsq(A, np.log10(flux[good]), rcond=None)[0]
            x0 = sol.copy()
            x0[0] = 10.0 ** sol[0]
        else:
            x0[0] = max(np.mean(flux), 1e-6)

    res = least_squares(lambda p: (model(p) - flux) / sigma, x0, method="lm" if len(nu) > nterms else "trf",
                        x_scale="jac")
    p = res.x
    J = res.jac
    chi2 = float(np.sum(res.fun**2))
    dof = max(len(nu) - nterms, 0)
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((nterms, nterms), np.nan)
    aic = chi2 + 2 * nterms
    errs = list(np.sqrt(np.clip(np.diag(cov), 0, None)))
    spec = Spectrum(kind, float(ref_freq), list(p), errs)
    return SpectralFit(spec, chi2, dof, aic, cov)


def fit_spectrum_auto(nu, flux, sigma, kind="logpol", max_terms=4, **kw):
    """Choose nterms (1..max_terms) by minimum AIC."""
    best = None
    for n in range(1, min(max_terms, len(nu)) + 1):
        f = fit_spectrum(nu, flux, sigma, kind=kind, nterms=n, **kw)
        if best is None or f.aic < best.aic - 1e-9:
            best = f
    return best
