"""Direct Fourier transform (DFT) prediction of point and Gaussian components.

Measurement equation used (Stokes I, per row and channel):

  V(u,v,w,nu) = sum_k  S_k(nu) * G_k(u,v,nu) * exp( sign * 2 pi i (nu/c) tau_k )

      tau_k = u l_k + v m_k + w (n_k - 1)          [metres; u,v,w are MS UVW in metres]
      G_k   = exp( -2 pi^2 (nu/c)^2 (C_ee u^2 + 2 C_en u v + C_nn v^2) )
              where C is the component's (East, North) covariance (pinpoint.shapes),
              i.e. the exact Fourier transform of the image-plane Gaussian.
      sign  = -1 by default (the CASA / wsclean convention). Exposed as an option only
              so it can be checked on real data (see VALIDATION.md); do not change it
              lightly.

Optional decorrelation ("smearing") factors, both real and multiplicative:
  frequency: sinc( tau * |dnu| / c )                         dnu = channel width
  time     : sinc( nu * dtau/dt * dt / c )                   dt  = row EXPOSURE
             dtau/dt = Omega_E [ l (w cos d0 - v sin d0) + m u sin d0 - (n-1) u cos d0 ]
             (d0 = phase-centre declination; derived from the standard (u,v,w)
              hour-angle relations, Thompson, Moran & Swenson eq. 4.1)
  with sinc(x) = sin(pi x) / (pi x).

Two implementations are provided and tested against each other:
  * `predict_reference`  vectorised numpy, direct cos/sin/exp for every sample. Slow but obvious.
  * `_kernel`            row-parallel loop using recurrences across channels: for uniformly
                         spaced channels, exp(i a nu_{j+1}) = exp(i a nu_j) * exp(i a dnu),
                         so each sample costs one complex multiply instead of a sin+cos
                         (and similarly for the Gaussian taper and time-smearing sinc).
                         Re-seeded exactly every `reseed` channels to bound round-off.
                         Compiled with numba when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._jit import HAVE_NUMBA, maybe_njit, prange
from .constants import C, OMEGA_EARTH
from .coords import radec_to_lmn
from .shapes import shape_to_cov

TWO_PI_OVER_C = 2.0 * np.pi / C
TWO_PI2_OVER_C2 = 2.0 * np.pi**2 / C**2
PI_OVER_C = np.pi / C


@dataclass
class ComponentArrays:
    l: np.ndarray
    m: np.ndarray
    nm1: np.ndarray  # n - 1, computed without cancellation
    cee: np.ndarray  # rad^2
    cen: np.ndarray
    cnn: np.ndarray
    spec: np.ndarray  # (ncomp, nchan) Jy

    @property
    def ncomp(self):
        return len(self.l)


def prepare_components(components, ra0, dec0, freqs) -> ComponentArrays:
    """Turn sky-model components into the flat arrays the kernels use, for one phase
    centre (ra0, dec0) and one set of channel frequencies."""
    freqs = np.asarray(freqs, dtype=np.float64)
    n = len(components)
    l = np.empty(n)
    m = np.empty(n)
    nm1 = np.empty(n)
    cov = np.zeros((n, 3))
    spec = np.empty((n, freqs.size))
    for k, c in enumerate(components):
        lk, mk, nk = radec_to_lmn(c.ra, c.dec, ra0, dec0)
        l[k], m[k] = lk, mk
        r2 = lk * lk + mk * mk
        nm1[k] = -r2 / (1.0 + nk)
        if c.type == "gauss":
            cv = shape_to_cov(c.maj, c.min, c.pa)
            cov[k] = (cv[0, 0], cv[0, 1], cv[1, 1])
        spec[k] = c.spectrum(freqs)
    return ComponentArrays(l, m, nm1, cov[:, 0].copy(), cov[:, 1].copy(), cov[:, 2].copy(), spec)


def _np_sinc(x):
    return np.sinc(x)


def predict_reference(uvw, freqs, comps: ComponentArrays, *, chan_width=None, interval=None, dec0=0.0,
                      sign=-1.0, freq_smearing=False, time_smearing=False):
    """Straightforward vectorised implementation. Returns complex128 (nrow, nchan)."""
    uvw = np.asarray(uvw, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    out = np.zeros((uvw.shape[0], freqs.size), dtype=np.complex128)
    for k in range(comps.ncomp):
        tau = u * comps.l[k] + v * comps.m[k] + w * comps.nm1[k]
        val = comps.spec[k][None, :] * np.exp(1j * sign * TWO_PI_OVER_C * tau[:, None] * freqs[None, :])
        if comps.cee[k] or comps.cen[k] or comps.cnn[k]:
            q = comps.cee[k] * u * u + 2 * comps.cen[k] * u * v + comps.cnn[k] * v * v
            val *= np.exp(-TWO_PI2_OVER_C2 * q[:, None] * freqs[None, :] ** 2)
        if freq_smearing:
            val *= _np_sinc(tau[:, None] * np.abs(np.asarray(chan_width))[None, :] / C)
        if time_smearing:
            sd, cd = np.sin(dec0), np.cos(dec0)
            dtau = OMEGA_EARTH * (comps.l[k] * (w * cd - v * sd) + comps.m[k] * u * sd - comps.nm1[k] * u * cd)
            val *= _np_sinc(dtau[:, None] * np.asarray(interval)[:, None] * freqs[None, :] / C)
        out += val
    return out


@maybe_njit(parallel=True)
def _kernel(u, v, w, freqs, chan_width, interval, l, m, nm1, cee, cen, cnn, spec,
            sign, uniform, fsmear, tsmear, sind0, cosd0, reseed, out):
    nrow = u.shape[0]
    nchan = freqs.shape[0]
    ncomp = l.shape[0]
    two_pi_c = 2.0 * np.pi / 299792458.0
    two_pi2_c2 = 2.0 * np.pi * np.pi / (299792458.0 * 299792458.0)
    pi_c = np.pi / 299792458.0
    omega = 7.292115e-5
    # ascending-frequency iteration order keeps every recurrence ratio <= 1 (no overflow)
    descending = uniform and nchan > 1 and freqs[nchan - 1] < freqs[0]
    dnu = 0.0
    if uniform and nchan > 1:
        dnu = abs(freqs[1] - freqs[0])
    width_uniform = True
    for j in range(1, nchan):
        if abs(abs(chan_width[j]) - abs(chan_width[0])) > 1e-9 * abs(chan_width[0]):
            width_uniform = False
    for r in prange(nrow):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        for k in range(ncomp):
            tau = ur * l[k] + vr * m[k] + wr * nm1[k]
            a = sign * two_pi_c * tau  # phase per Hz
            isg = (cee[k] != 0.0) or (cen[k] != 0.0) or (cnn[k] != 0.0)
            alpha = 0.0
            if isg:
                alpha = two_pi2_c2 * (cee[k] * ur * ur + 2.0 * cen[k] * ur * vr + cnn[k] * vr * vr)
            fs_const = 1.0
            if fsmear and width_uniform:
                x = pi_c * tau * abs(chan_width[0])
                if abs(x) > 1e-8:
                    fs_const = np.sin(x) / x
            b = 0.0
            if tsmear:
                dtau = omega * (l[k] * (wr * cosd0 - vr * sind0) + m[k] * ur * sind0 - nm1[k] * ur * cosd0)
                b = pi_c * dtau * interval[r]  # time-smearing argument per Hz
            if uniform:
                sr = np.cos(a * dnu)
                si = np.sin(a * dnu)
                tsr = np.cos(b * dnu)
                tsi = np.sin(b * dnu)
                rr = np.exp(-2.0 * alpha * dnu * dnu)
                zr = 0.0
                zi = 0.0
                g = 1.0
                rat = 1.0
                tr = 1.0
                ti = 0.0
                for jj in range(nchan):
                    j = nchan - 1 - jj if descending else jj
                    nu = freqs[j]
                    if jj % reseed == 0:
                        ph = a * nu
                        zr = np.cos(ph)
                        zi = np.sin(ph)
                        if isg:
                            g = np.exp(-alpha * nu * nu)
                            rat = np.exp(-alpha * (2.0 * nu * dnu + dnu * dnu))
                        if tsmear:
                            tr = np.cos(b * nu)
                            ti = np.sin(b * nu)
                    amp = spec[k, j] * g
                    if fsmear:
                        if width_uniform:
                            amp *= fs_const
                        else:
                            x = pi_c * tau * abs(chan_width[j])
                            if abs(x) > 1e-8:
                                amp *= np.sin(x) / x
                    if tsmear:
                        x = b * nu
                        if abs(x) > 1e-8:
                            amp *= ti / x
                    out[r, j] += amp * (zr + 1j * zi)
                    # advance recurrences to the next (higher) frequency
                    t0 = zr * sr - zi * si
                    zi = zr * si + zi * sr
                    zr = t0
                    if isg:
                        g *= rat
                        rat *= rr
                    if tsmear:
                        t0 = tr * tsr - ti * tsi
                        ti = tr * tsi + ti * tsr
                        tr = t0
            else:
                for j in range(nchan):
                    nu = freqs[j]
                    ph = a * nu
                    amp = spec[k, j]
                    if isg:
                        amp *= np.exp(-alpha * nu * nu)
                    if fsmear:
                        x = pi_c * tau * abs(chan_width[j])
                        if abs(x) > 1e-8:
                            amp *= np.sin(x) / x
                    if tsmear:
                        x = b * nu
                        if abs(x) > 1e-8:
                            amp *= np.sin(x) / x
                    out[r, j] += amp * (np.cos(ph) + 1j * np.sin(ph))


def is_uniform(freqs, rtol=1e-9):
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.size < 2:
        return False
    d = np.diff(freqs)
    return bool(np.all(np.abs(d - d[0]) <= rtol * np.abs(freqs).max()) and d[0] != 0)


def predict_loop(uvw, freqs, comps: ComponentArrays, *, chan_width=None, interval=None, dec0=0.0, sign=-1.0,
                 freq_smearing=False, time_smearing=False, reseed=64, force_direct=False):
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    nrow = uvw.shape[0]
    cw = np.ascontiguousarray(np.zeros_like(freqs) if chan_width is None else np.broadcast_to(chan_width, freqs.shape),
                              dtype=np.float64)
    it = np.ascontiguousarray(np.zeros(nrow) if interval is None else np.broadcast_to(interval, (nrow,)),
                              dtype=np.float64)
    out = np.zeros((nrow, freqs.size), dtype=np.complex128)
    uniform = (not force_direct) and is_uniform(freqs)
    _kernel(np.ascontiguousarray(uvw[:, 0]), np.ascontiguousarray(uvw[:, 1]), np.ascontiguousarray(uvw[:, 2]),
            freqs, cw, it, comps.l, comps.m, comps.nm1, comps.cee, comps.cen, comps.cnn,
            np.ascontiguousarray(comps.spec), float(sign), uniform, bool(freq_smearing), bool(time_smearing),
            float(np.sin(dec0)), float(np.cos(dec0)), int(reseed), out)
    return out


BACKENDS = ("auto", "numpy", "numba", "loop")


def predict_stokes_i(uvw, freqs, comps: ComponentArrays, *, backend="auto", row_block=4096, **kw):
    """Stokes I visibilities (nrow, nchan) complex128.

    backend: "numba" compiled recurrence kernel (needs numba), "numpy" vectorised reference,
             "loop" the recurrence kernel without forcing numba (pure Python if numba is absent:
             only for tests), "auto" = numba if installed else numpy.
    """
    if kw.get("freq_smearing") and kw.get("chan_width") is None:
        raise ValueError("freq_smearing requires chan_width")
    if kw.get("time_smearing") and kw.get("interval") is None:
        raise ValueError("time_smearing requires interval")
    if backend == "auto":
        backend = "numba" if HAVE_NUMBA else "numpy"
    if backend == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("backend 'numba' requested but numba is not installed (pip install numba)")
        return predict_loop(uvw, freqs, comps, **kw)
    if backend == "loop":
        return predict_loop(uvw, freqs, comps, **kw)
    if backend == "numpy":
        kw.pop("reseed", None)
        kw.pop("force_direct", None)
        uvw = np.asarray(uvw)
        interval = kw.pop("interval", None)
        out = np.empty((uvw.shape[0], np.size(freqs)), dtype=np.complex128)
        for s in range(0, uvw.shape[0], row_block):
            sl = slice(s, s + row_block)
            iv = None if interval is None else np.broadcast_to(interval, (uvw.shape[0],))[sl]
            out[sl] = predict_reference(uvw[sl], freqs, comps, interval=iv, **kw)
        return out
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


# ---------------- correlations ----------------

# casacore Stokes enum codes -> weight of Stokes I in that correlation
# (MS / CASA convention: XX = I + Q, YY = I - Q, RR = I + V, LL = I - V; no factor 1/2)
STOKES_CODES = {1: "I", 5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
_I_WEIGHT = {1: 1.0, 5: 1.0, 8: 1.0, 9: 1.0, 12: 1.0, 6: 0.0, 7: 0.0, 10: 0.0, 11: 0.0}


def stokes_i_weights(corr_types):
    try:
        return np.array([_I_WEIGHT[int(c)] for c in corr_types], dtype=np.float64)
    except KeyError as e:
        raise ValueError(f"unsupported correlation type code {e.args[0]} (supported: "
                         f"{sorted(STOKES_CODES.items())})") from None


def stokes_i_to_corr(vis_i, corr_types, dtype=np.complex64):
    wts = stokes_i_weights(corr_types)
    return (vis_i[:, :, None] * wts[None, None, :]).astype(dtype)
