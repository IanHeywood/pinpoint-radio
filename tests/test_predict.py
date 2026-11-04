"""Stage: DFT prediction (pinpoint/predict.py).

What these tests establish:
  * a source at the phase centre gives V = S(nu) on every baseline;
  * an offset point source gives exactly S exp(-2 pi i nu/c (ul + vm + w(n-1)));
  * the Gaussian taper equals a brute-force numerical Fourier transform of the
    image-plane Gaussian (so image-plane PA and visibility-plane PA agree);
  * the fast recurrence kernel equals the direct reference to ~1e-12 across
    thousands of channels, for ascending, descending and irregular channel layouts,
    with and without smearing;
  * the analytic frequency- and time-smearing factors match brute-force averaging
    of unsmeared visibilities over the channel width / integration time;
  * Stokes I goes to XX, YY (or RR, LL) with cross-hands zero.
"""

import numpy as np
import pytest

from pinpoint.constants import ARCSEC, C, DEG, OMEGA_EARTH
from pinpoint.coords import radec_to_lmn
from pinpoint.predict import (prepare_components, predict_loop, predict_reference, predict_stokes_i,
                              stokes_i_to_corr)
from pinpoint.skymodel import Component
from pinpoint.spectral import Spectrum
from pinpoint.testing.synth_ms import baselines, random_array, uvw_from_hour_angle

RA0, DEC0 = 1.2, -30 * DEG


def comp(dra_arcsec=0.0, ddec_arcsec=0.0, kind="point", flux=(2.0, -0.7), **shape):
    return Component("c", kind, RA0 + dra_arcsec * ARCSEC / np.cos(DEC0), DEC0 + ddec_arcsec * ARCSEC,
                     Spectrum("logpol", 1.3e9, list(flux)), **shape)


def uvw_tracks(nant=12, ntime=10, extent=8000.0, dec0=DEC0):
    _, _, bl = baselines(random_array(nant, extent))
    return uvw_from_hour_angle(bl, np.linspace(-1, 1, ntime), dec0).reshape(-1, 3)


FREQS = np.linspace(0.9e9, 1.7e9, 64)


def test_phase_centre_is_flux():
    uvw = uvw_tracks()
    ca = prepare_components([comp()], RA0, DEC0, FREQS)
    V = predict_reference(uvw, FREQS, ca)
    S = 2.0 * (FREQS / 1.3e9) ** -0.7
    assert np.allclose(V, S[None, :], rtol=1e-14, atol=0)


def test_offset_point_analytic():
    uvw = uvw_tracks()
    c = comp(300.0, -200.0)
    ca = prepare_components([c], RA0, DEC0, FREQS)
    V = predict_reference(uvw, FREQS, ca)
    l, m, n = radec_to_lmn(c.ra, c.dec, RA0, DEC0)
    tau = uvw[:, 0] * l + uvw[:, 1] * m + uvw[:, 2] * (n - 1)
    expect = c.spectrum(FREQS)[None, :] * np.exp(-2j * np.pi * tau[:, None] * FREQS[None, :] / C)
    assert np.max(np.abs(V - expect)) < 1e-9


def test_gaussian_matches_numerical_fourier_transform():
    maj, mn, pa = 20 * ARCSEC, 8 * ARCSEC, 50 * DEG
    c = comp(kind="gauss", flux=(1.0,), maj=maj, min=mn, pa=pa)
    # image-plane Gaussian written out explicitly: major axis along (E,N) = (sin pa, cos pa)
    h = 0.5 * ARCSEC
    g = np.arange(-120, 121) * h
    E, N = np.meshgrid(g, g)
    along = E * np.sin(pa) + N * np.cos(pa)
    perp = E * np.cos(pa) - N * np.sin(pa)
    s = 1 / (2 * np.sqrt(2 * np.log(2)))
    I = np.exp(-0.5 * ((along / (maj * s)) ** 2 + (perp / (mn * s)) ** 2))
    I /= I.sum()
    nu = np.array([1.3e9])
    lam = C / nu[0]
    rng = np.random.default_rng(0)
    uv = rng.uniform(-6000, 6000, (40, 2)) * lam  # metres, i.e. baselines up to 6000 wavelengths
    uvw = np.column_stack([uv, np.zeros(40)])
    ca = prepare_components([c], RA0, DEC0, nu)
    V = predict_reference(uvw, nu, ca)[:, 0]
    k = uv / lam
    Vnum = np.array([np.sum(I * np.exp(-2j * np.pi * (ku * E + kv * N))) for ku, kv in k])
    assert np.max(np.abs(V - Vnum)) < 1e-6
    # and it is genuinely non-trivial (resolved on these baselines)
    assert np.min(np.abs(V)) < 0.5


def _mixed_components():
    return [comp(400.0, -150.0),
            comp(-2000.0, 900.0, kind="gauss", flux=(0.5, -1.1, 0.2), maj=6 * ARCSEC, min=2 * ARCSEC, pa=30 * DEG),
            comp(3600.0, 3600.0, flux=(0.1,))]


@pytest.mark.parametrize("layout", ["ascending", "descending", "irregular"])
@pytest.mark.parametrize("smear", [False, True])
def test_recurrence_kernel_matches_reference(layout, smear):
    nchan = 1500
    f = np.linspace(0.856e9, 1.712e9, nchan)
    if layout == "descending":
        f = f[::-1].copy()
    if layout == "irregular":
        f = np.sort(f + np.random.default_rng(2).uniform(-1e5, 1e5, nchan))
    width = np.full(nchan, 0.856e9 / nchan)
    uvw = uvw_tracks(nant=6, ntime=4)
    interval = np.full(len(uvw), 8.0)
    ca = prepare_components(_mixed_components(), RA0, DEC0, f)
    kw = dict(chan_width=width, interval=interval, dec0=DEC0, freq_smearing=smear, time_smearing=smear)
    ref = predict_reference(uvw, f, ca, **kw)
    fast = predict_loop(uvw, f, ca, **kw)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(fast - ref)) / scale < 1e-11


def test_frequency_smearing_matches_brute_force():
    c = comp(1800.0, 900.0)  # ~0.56 deg off axis
    uvw = uvw_tracks(extent=16000.0)
    nu = np.array([1.0e9, 1.4e9])
    width = np.array([2.0e6, 2.0e6])
    nsub = 256
    t = (np.arange(nsub) + 0.5) / nsub - 0.5
    brute = np.zeros((len(uvw), 2), complex)
    for j in range(2):
        fsub = nu[j] + width[j] * t
        ca_sub = prepare_components([c], RA0, DEC0, fsub)
        # hold the spectrum fixed at the channel centre (we test decorrelation, not spectra)
        ca_sub.spec[:] = c.spectrum(nu[j])
        brute[:, j] = predict_reference(uvw, fsub, ca_sub).mean(axis=1)
    ca = prepare_components([c], RA0, DEC0, nu)
    smeared = predict_reference(uvw, nu, ca, chan_width=width, freq_smearing=True)
    unsmeared = predict_reference(uvw, nu, ca)
    S = c.spectrum(nu)
    assert np.max(np.abs(np.abs(unsmeared) - np.abs(brute)) / S) > 0.01  # the effect is real here
    assert np.max(np.abs(smeared - brute) / S) < 2e-4


def test_time_smearing_matches_brute_force():
    c = comp(-1500.0, 2500.0)
    _, _, bl = baselines(random_array(10, 16000.0))
    H0 = np.linspace(-1.0, 1.0, 5)
    dt = 32.0
    nsub = 256
    t = ((np.arange(nsub) + 0.5) / nsub - 0.5) * dt
    nu = np.array([1.4e9])
    ca = prepare_components([c], RA0, DEC0, nu)
    uvw_c = uvw_from_hour_angle(bl, H0, DEC0).reshape(-1, 3)
    brute = np.zeros((len(uvw_c), 1), complex)
    for ts in t:
        uvw_s = uvw_from_hour_angle(bl, H0 + OMEGA_EARTH * ts, DEC0).reshape(-1, 3)
        brute += predict_reference(uvw_s, nu, ca)
    brute /= nsub
    interval = np.full(len(uvw_c), dt)
    smeared = predict_reference(uvw_c, nu, ca, interval=interval, dec0=DEC0, time_smearing=True)
    unsmeared = predict_reference(uvw_c, nu, ca)
    S = c.spectrum(nu)[0]
    assert np.max(np.abs(np.abs(unsmeared) - np.abs(brute))) / S > 0.01
    # residual is second order (curvature of the phase track over the integration)
    assert np.max(np.abs(np.abs(smeared) - np.abs(brute))) / S < 2e-3


def test_backends_agree():
    uvw = uvw_tracks(nant=5, ntime=3)
    ca = prepare_components(_mixed_components(), RA0, DEC0, FREQS)
    a = predict_stokes_i(uvw, FREQS, ca, backend="numpy", row_block=7)
    b = predict_stokes_i(uvw, FREQS, ca, backend="loop")
    assert np.max(np.abs(a - b)) < 1e-11


def test_smearing_requires_metadata():
    ca = prepare_components([comp()], RA0, DEC0, FREQS)
    with pytest.raises(ValueError):
        predict_stokes_i(np.zeros((2, 3)), FREQS, ca, freq_smearing=True)


def test_correlation_mapping():
    vis = np.full((3, 2), 1.5 + 0.5j)
    lin = stokes_i_to_corr(vis, [9, 10, 11, 12])
    assert np.allclose(lin[..., 0], vis) and np.allclose(lin[..., 3], vis)
    assert np.all(lin[..., 1] == 0) and np.all(lin[..., 2] == 0)
    circ = stokes_i_to_corr(vis, [5, 8])
    assert np.allclose(circ, vis[..., None])
    with pytest.raises(ValueError):
        stokes_i_to_corr(vis, [2])  # Stokes Q column: not supported
