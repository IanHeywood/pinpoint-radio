"""Joint multi-band image-plane fitting.

For one region and all sub-bands at once, the model in band b is

    I_b(pixel) = sum_k F_bk * R(pixel - x_k; source shape_k, restoring beam_b) + bg_b

  * x_k   : component position, SHARED by all bands (this is what buys sub-pixel accuracy:
            every band constrains the same two numbers)
  * shape : intrinsic Gaussian shape (point = zero size), SHARED by all bands
  * F_bk  : flux of component k in band b, FREE per band (the apparent spectrum)
  * bg_b  : optional constant background per band
  * R     : beam-convolved profile in Jy/beam (pinpoint.shapes.restored_profile)

Positions are offsets (arcsec, East/North) from the region centre in the orthographic
projection about that centre; they are turned back into RA/Dec exactly at the end.
Gaussian shapes are parametrised by a Cholesky factor of their covariance so the fit
never produces an invalid shape and can shrink smoothly to a point.

After the joint fit, a spectral model is fitted to each component's per-band fluxes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .constants import ARCSEC
from .coords import lmn_to_radec
from .images import Cutout, SubbandImage, region_cutout
from .shapes import cov_to_shape, restored_profile, shape_to_cov
from .skymodel import Component, Source
from .spectral import SpectralFit, fit_spectrum, fit_spectrum_auto

# Empirical variance inflation for beam-correlated image noise, in units of the
# restoring-beam area in pixels. Calibrated by tests/test_fitting.py::test_position_errors_are_honest
# (pull-distribution test on images with beam-correlated noise).
DEFAULT_NOISE_CORR_FACTOR = 0.6


@dataclass
class FitOptions:
    background: bool = True
    spectral_kind: str = "logpol"
    nterms: int | str = "auto"  # int or "auto" (AIC)
    max_terms: int = 4
    band_integrate: bool = False
    ref_freq: float | None = None
    correlated_noise: bool = True
    noise_corr_factor: float = DEFAULT_NOISE_CORR_FACTOR
    per_band_positions: bool = False


@dataclass
class ComponentResult:
    type: str
    ra: float
    dec: float
    de_err_arcsec: float  # 1-sigma East offset error
    dn_err_arcsec: float  # 1-sigma North offset error
    maj: float = 0.0
    min: float = 0.0
    pa: float = 0.0
    shape_err: tuple = (0.0, 0.0, 0.0)  # (maj arcsec, min arcsec, pa deg) 1-sigma
    band_flux: np.ndarray = field(default=None, repr=False)
    band_flux_err: np.ndarray = field(default=None, repr=False)
    spectral: SpectralFit | None = None
    band_offsets_arcsec: np.ndarray | None = field(default=None, repr=False)  # (nb, 2) per-band (dE, dN) - joint
    band_offsets_err: np.ndarray | None = field(default=None, repr=False)


@dataclass
class SourceResult:
    name: str
    column: str | None
    components: list[ComponentResult]
    freqs: np.ndarray
    chi2_red: float
    npix: int
    cutout: Cutout = field(repr=False)
    data: np.ndarray = field(repr=False)  # (nb, ny, nx)
    model: np.ndarray = field(repr=False)
    error_scale: float = 1.0

    def to_source(self) -> Source:
        comps = []
        for i, c in enumerate(self.components):
            errs = {"ra_arcsec": c.de_err_arcsec, "dec_arcsec": c.dn_err_arcsec}
            if c.type == "gauss":
                errs.update(maj_arcsec=c.shape_err[0], min_arcsec=c.shape_err[1], pa_deg=c.shape_err[2])
            comps.append(Component(f"{self.name}_c{i}", c.type, c.ra, c.dec, c.spectral.spectrum,
                                   maj=c.maj, min=c.min, pa=c.pa, errors=errs))
        return Source(self.name, comps, self.column)


class _Layout:
    """Bookkeeping for the parameter vector."""

    def __init__(self, types, nb, background, fix_shapes=None):
        self.types = types
        self.nc = len(types)
        self.nb = nb
        self.background = background
        self.fix_shapes = fix_shapes  # list of cov (arcsec^2) or None
        idx = 0
        self.pos = []
        for _ in types:
            self.pos.append(slice(idx, idx + 2))
            idx += 2
        self.shape = []
        for t in types:
            if t == "gauss" and fix_shapes is None:
                self.shape.append(slice(idx, idx + 3))
                idx += 3
            else:
                self.shape.append(None)
        self.flux = slice(idx, idx + nb * self.nc)
        idx += nb * self.nc
        self.bg = slice(idx, idx + nb) if background else None
        if background:
            idx += nb
        self.n = idx

    def unpack(self, p):
        pos = [p[s] for s in self.pos]
        covs = []
        for k, s in enumerate(self.shape):
            if self.fix_shapes is not None:
                covs.append(self.fix_shapes[k])
            elif s is None:
                covs.append(np.zeros((2, 2)))
            else:
                a, b, c = p[s]
                covs.append(np.array([[a * a, a * b], [a * b, b * b + c * c]]))
        flux = p[self.flux].reshape(self.nb, self.nc)
        bg = p[self.bg] if self.background else np.zeros(self.nb)
        return pos, covs, flux, bg


def _model_cube(p, lay, de_as, dn_as, beams_cov_as):
    pos, covs, flux, bg = lay.unpack(p)
    out = np.empty((lay.nb,) + de_as.shape)
    for b in range(lay.nb):
        img = np.full(de_as.shape, bg[b])
        for k in range(lay.nc):
            img += flux[b, k] * restored_profile(de_as - pos[k][0], dn_as - pos[k][1], covs[k], beams_cov_as[b])
        out[b] = img
    return out


def _initial_guess(data, sig, mask, de_as, dn_as, beams_cov_as, types, background):
    nb = data.shape[0]
    w = 1.0 / sig**2
    resid = data.copy()
    mean_beam = np.mean(beams_cov_as, axis=0)
    pos, fluxes, shapes = [], [], []
    for t in types:
        comb = np.tensordot(w, resid, axes=1) / w.sum()
        comb = np.where(mask, comb, -np.inf)
        iy, ix = np.unravel_index(np.argmax(comb), comb.shape)
        e0, n0 = de_as[iy, ix], dn_as[iy, ix]
        peaks = resid[:, iy, ix].copy()
        if t == "gauss":
            L = 0.5 * np.linalg.cholesky(mean_beam)
            cov = L @ L.T
            shapes.append([L[0, 0], L[1, 0], L[1, 1]])
        else:
            cov = np.zeros((2, 2))
        f = np.empty(nb)
        for b in range(nb):
            prof = restored_profile(de_as - e0, dn_as - n0, cov, beams_cov_as[b])
            f[b] = peaks[b] / prof[iy, ix]
            resid[b] -= f[b] * prof
        pos.append([e0, n0])
        fluxes.append(f)
    p = []
    for k in range(len(types)):
        p += list(pos[k])
    # _Layout order: all positions, then Gaussian shape params, then fluxes, then background
    p +=[x for s in shapes for x in s]
    p += list(np.stack(fluxes, axis=1).ravel())
    if background:
        p += [0.0] * nb
    return np.array(p, dtype=np.float64)


def _run_fit(data, sig, mask, de_as, dn_as, beams_cov_as, types, background, x0=None, fix_shapes=None):
    lay = _Layout(types, data.shape[0], background, fix_shapes)
    if x0 is None:
        x0 = _initial_guess(data, sig, mask, de_as, dn_as, beams_cov_as, types, background)
    m3 = np.broadcast_to(mask, data.shape)

    def resid(p):
        mod = _model_cube(p, lay, de_as, dn_as, beams_cov_as)
        return ((data - mod) / sig[:, None, None])[m3]

    res = least_squares(resid, x0, method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=200 * len(x0))
    if not res.success:
        warnings.warn(f"least-squares did not converge: {res.message}")
    J = res.jac
    try:
        cov = np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((lay.n, lay.n), np.nan)
    ndof = max(res.fun.size - lay.n, 1)
    chi2_red = float(np.sum(res.fun**2) / ndof)
    return res.x, cov, chi2_red, lay


def _shape_errors(Lp, Lcov, nsamp=400, rng=None):
    """Propagate Cholesky-parameter covariance to (maj, min, pa) by sampling."""
    rng = rng or np.random.default_rng(12345)
    try:
        draws = rng.multivariate_normal(Lp, Lcov, size=nsamp)
    except (np.linalg.LinAlgError, ValueError):
        return (np.nan, np.nan, np.nan)
    shp = []
    for a, b, c in draws:
        C = np.array([[a * a, a * b], [a * b, b * b + c * c]])
        shp.append(cov_to_shape(C))
    shp = np.array(shp)
    ref_pa = cov_to_shape(np.array([[Lp[0] ** 2, Lp[0] * Lp[1]], [Lp[0] * Lp[1], Lp[1] ** 2 + Lp[2] ** 2]]))[2]
    dpa = (shp[:, 2] - ref_pa + np.pi / 2) % np.pi - np.pi / 2
    return (float(np.std(shp[:, 0])), float(np.std(shp[:, 1])), float(np.rad2deg(np.std(dpa))))


def fit_region(images: list[SubbandImage], region, exclusions=(), opts: FitOptions | None = None) -> SourceResult:
    opts = opts or FitOptions()
    if not images:
        raise ValueError("no images")
    shape = images[0].data.shape
    geom = images[0].geometry
    cut = region_cutout(region, geom, shape, exclusions)
    if cut.mask.sum() < 10:
        raise ValueError(f"region {region.name!r}: only {cut.mask.sum()} usable pixels")
    data = np.stack([im.data[cut.ys, cut.xs] for im in images])
    finite = np.all(np.isfinite(data), axis=0)
    mask = cut.mask & finite
    sig = np.array([im.noise for im in images], dtype=np.float64)
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
        raise ValueError("invalid image noise estimates")
    freqs = np.array([im.freq for im in images])
    de_as, dn_as = cut.de / ARCSEC, cut.dn / ARCSEC
    beams_cov_as = np.stack([shape_to_cov(im.beam[0] / ARCSEC, im.beam[1] / ARCSEC, im.beam[2]) for im in images])
    types = list(region.components)

    p, cov, chi2_red, lay = _run_fit(data, sig, mask, de_as, dn_as, beams_cov_as, types, opts.background)

    # error scaling: correlated noise + any excess scatter (artefacts) in the residuals
    pix_as = geom.pixel_scale / ARCSEC
    beam_area_pix = np.mean([np.pi * im.beam[0] * im.beam[1] / (4 * np.log(2)) for im in images]) / geom.pixel_scale**2
    scale = 1.0
    if opts.correlated_noise:
        scale *= max(opts.noise_corr_factor * beam_area_pix, 1.0)
    scale *= max(chi2_red, 1.0)
    cov = cov * scale

    pos, covs, flux, bg = lay.unpack(p)
    model = _model_cube(p, lay, de_as, dn_as, beams_cov_as)
    ref_freq = opts.ref_freq or float(np.mean(freqs))
    bws = np.array([im.bandwidth for im in images])

    comps = []
    for k, t in enumerate(types):
        e, n = pos[k]
        ra, dec = lmn_to_radec(e * ARCSEC, n * ARCSEC, *cut.centre)
        perr = np.sqrt(np.clip(np.diag(cov)[lay.pos[k]], 0, None))
        fidx = np.arange(lay.flux.start, lay.flux.stop).reshape(lay.nb, lay.nc)[:, k]
        ferr = np.sqrt(np.clip(np.diag(cov)[fidx], 0, None))
        f = flux[:, k]
        if opts.nterms == "auto":
            sfit = fit_spectrum_auto(freqs, f, ferr, kind=opts.spectral_kind, max_terms=opts.max_terms,
                                     ref_freq=ref_freq, bandwidth=bws, band_integrate=opts.band_integrate)
        else:
            sfit = fit_spectrum(freqs, f, ferr, kind=opts.spectral_kind, nterms=int(opts.nterms),
                                ref_freq=ref_freq, bandwidth=bws, band_integrate=opts.band_integrate)
        cr = ComponentResult(t, float(ra), float(dec), float(perr[0]), float(perr[1]),
                             band_flux=f.copy(), band_flux_err=ferr, spectral=sfit)
        if t == "gauss":
            maj, mn, pa = cov_to_shape(covs[k])
            cr.maj, cr.min, cr.pa = maj * ARCSEC, mn * ARCSEC, pa
            s = lay.shape[k]
            cr.shape_err = _shape_errors(p[s], cov[s, s])
        comps.append(cr)

    if opts.per_band_positions:
        fixed = covs
        x_joint = p
        for b in range(len(images)):
            lay_b = _Layout(types, 1, opts.background, fixed)
            x0 = []
            for k in range(len(types)):
                x0 += list(pos[k])
            x0 += list(flux[b])
            if opts.background:
                x0 += [bg[b]]
            pb, covb, chi2b, _ = _run_fit(data[b:b + 1], sig[b:b + 1], mask, de_as, dn_as, beams_cov_as[b:b + 1],
                                          types, opts.background, x0=np.array(x0), fix_shapes=fixed)
            covb = covb * scale / max(chi2_red, 1.0) * max(chi2b, 1.0)
            for k in range(len(types)):
                if comps[k].band_offsets_arcsec is None:
                    comps[k].band_offsets_arcsec = np.zeros((len(images), 2))
                    comps[k].band_offsets_err = np.zeros((len(images), 2))
                comps[k].band_offsets_arcsec[b] = pb[2 * k:2 * k + 2] - x_joint[2 * k:2 * k + 2]
                comps[k].band_offsets_err[b] = np.sqrt(np.clip(np.diag(covb)[2 * k:2 * k + 2], 0, None))

    return SourceResult(region.name, region.column, comps, freqs, chi2_red, int(mask.sum()), cut, data, model, scale)
