"""Stage: joint image-plane fitting (pinpoint/fitting.py).

The synthetic images come from pinpoint.testing.synth_images, which renders in pixel
coordinates with its own formulas and FFT convolution (independent of the fitter's
sky-frame maths). So these tests check conventions, not just self-consistency.

What these tests establish:
  * a point source placed off the pixel grid (+0.37, -0.21 pix) is recovered to
    ~1/1000 of a pixel at high SNR (the whole point of the tool);
  * the per-band fluxes follow the true spectrum, and the spectral fit recovers it;
  * quoted position errors are honest on images with beam-correlated noise
    (pull distribution has unit width);
  * an elliptical Gaussian's size and position angle (North through East) are
    recovered, so the PA convention matches the FITS BPA convention;
  * a point + Gaussian blend is separated;
  * a constant background offset doesn't bias the fit;
  * per-band position fits scatter around the joint position (chromatic-astrometry
    diagnostic).
"""

import numpy as np
import pytest

from pinpoint.constants import ARCSEC, DEG
from pinpoint.coords import radec_to_lmn
from pinpoint.fitting import FitOptions, fit_region
from pinpoint.regions import parse_region_string
from pinpoint.testing.synth_images import TrueComponent, make_subbands

RA0, DEC0 = 150 * DEG, -30 * DEG
FREQS = np.linspace(0.9e9, 1.6e9, 8)
CELL = 1.0  # arcsec


def offset(dx_pix, dy_pix):
    """Sky position offset from the image centre by dx pixels to the right (West) and dy up (North)."""
    return RA0 - dx_pix * CELL * ARCSEC / np.cos(DEC0), DEC0 + dy_pix * CELL * ARCSEC


def sep_arcsec(c, ra, dec):
    l, m, _ = radec_to_lmn(c.ra, c.dec, ra, dec)
    return l / ARCSEC, m / ARCSEC


def region(text="", r=15):
    return parse_region_string(f'fk5\ncircle(150,-30,{r}") # text={{{text or "point"}}}')[0]


def spec(nu):
    return 2.0 * (nu / 1.25e9) ** -0.7


def test_subpixel_point_source_high_snr():
    ra, dec = offset(0.37, -0.21)
    ims = make_subbands([TrueComponent(ra, dec, spec)], FREQS, ra0=RA0, dec0=DEC0, npix=96, rms=1e-6)
    r = fit_region(ims, region(), opts=FitOptions(nterms=2, ref_freq=1.25e9))
    c = r.components[0]
    dl, dm = sep_arcsec(c, ra, dec)
    assert abs(dl) < 1e-3 * CELL and abs(dm) < 1e-3 * CELL
    assert np.allclose(c.band_flux, spec(FREQS), rtol=1e-4)
    s0, alpha = c.spectral.spectrum.coeffs
    assert np.isclose(s0, 2.0, rtol=1e-4) and np.isclose(alpha, -0.7, atol=1e-3)


def test_position_errors_are_honest():
    ra, dec = offset(0.37, -0.21)
    pulls = []
    for i in range(60):
        ims = make_subbands([TrueComponent(ra, dec, lambda nu: 0.01 * (nu / 1.2e9) ** -0.7)], FREQS,
                            ra0=RA0, dec0=DEC0, npix=96, rms=1e-4, rng=np.random.default_rng(100 + i))
        c = fit_region(ims, region()).components[0]
        dl, dm = sep_arcsec(c, ra, dec)
        pulls.append([dl / c.de_err_arcsec, dm / c.dn_err_arcsec])
    s = np.std(pulls, axis=0)
    assert np.all(s > 0.75) and np.all(s < 1.25), s


@pytest.mark.parametrize("pa_deg", [0.0, 60.0, 135.0])
def test_gaussian_shape_and_pa(pa_deg):
    ra, dec = offset(-0.3, 0.45)
    true = TrueComponent(ra, dec, spec, maj=9 * ARCSEC, min=4 * ARCSEC, pa=pa_deg * DEG)
    ims = make_subbands([true], FREQS, ra0=RA0, dec0=DEC0, npix=128, rms=1e-6)
    r = fit_region(ims, region("gauss", r=30), opts=FitOptions(nterms=2, ref_freq=1.25e9))
    c = r.components[0]
    dl, dm = sep_arcsec(c, ra, dec)
    assert abs(dl) < 0.01 and abs(dm) < 0.01
    assert np.isclose(c.maj / ARCSEC, 9.0, rtol=0.01)
    assert np.isclose(c.min / ARCSEC, 4.0, rtol=0.02)
    dpa = (np.rad2deg(c.pa) - pa_deg + 90) % 180 - 90
    assert abs(dpa) < 0.5
    assert np.isclose(c.spectral.spectrum.coeffs[0], 2.0, rtol=0.005)


def test_point_plus_gaussian_blend():
    ra1, dec1 = offset(0.2, 0.1)
    ra2, dec2 = offset(-8.3, 5.6)
    comps = [TrueComponent(ra1, dec1, spec),
             TrueComponent(ra2, dec2, lambda nu: 0.5 * (nu / 1.25e9) ** -1.0, maj=6 * ARCSEC, min=3 * ARCSEC, pa=20 * DEG)]
    ims = make_subbands(comps, FREQS, ra0=RA0, dec0=DEC0, npix=128, rms=1e-6)
    r = fit_region(ims, region("point+gauss", r=30), opts=FitOptions(nterms=2, ref_freq=1.25e9))
    p, g = r.components
    assert p.type == "point" and g.type == "gauss"
    assert max(map(abs, sep_arcsec(p, ra1, dec1))) < 0.01
    assert max(map(abs, sep_arcsec(g, ra2, dec2))) < 0.05
    assert np.isclose(g.spectral.spectrum.coeffs[1], -1.0, atol=0.02)


def test_background_offset_handled():
    ra, dec = offset(0.37, -0.21)
    ims = make_subbands([TrueComponent(ra, dec, spec)], FREQS, ra0=RA0, dec0=DEC0, npix=96, rms=1e-6)
    for im in ims:
        im.data = im.data + 0.003
    c = fit_region(ims, region()).components[0]
    assert max(map(abs, sep_arcsec(c, ra, dec))) < 1e-3
    assert np.allclose(c.band_flux, spec(FREQS), rtol=1e-4)


def test_per_band_positions_diagnostic():
    ra, dec = offset(0.37, -0.21)
    ims = make_subbands([TrueComponent(ra, dec, lambda nu: 0.01 * (nu / 1.2e9) ** -0.7)], FREQS,
                        ra0=RA0, dec0=DEC0, npix=96, rms=1e-4, rng=np.random.default_rng(5))
    c = fit_region(ims, region(), opts=FitOptions(per_band_positions=True)).components[0]
    off, err = c.band_offsets_arcsec, c.band_offsets_err
    assert off.shape == (len(FREQS), 2)
    chi = off / err
    assert np.max(np.abs(chi)) < 4.5  # consistent with no chromatic shift


def test_to_source_round_trip():
    ra, dec = offset(0.37, -0.21)
    ims = make_subbands([TrueComponent(ra, dec, spec)], FREQS, ra0=RA0, dec0=DEC0, npix=96, rms=1e-6)
    reg = parse_region_string('fk5\ncircle(150,-30,15") # text={point col=DIR1_DATA name=bright}')[0]
    src = fit_region(ims, reg).to_source()
    assert src.name == "bright" and src.column == "DIR1_DATA"
    assert src.components[0].name == "bright_c0"


def test_fully_flagged_subbands_are_skipped():
    """Sub-bands that are all NaN, all zero, or have no valid noise/beam (fully flagged
    channels) are left out with a reason, and the fit from the rest is unaffected."""
    ra, dec = offset(0.37, -0.21)
    ims = make_subbands([TrueComponent(ra, dec, spec)], FREQS, ra0=RA0, dec0=DEC0, npix=96, rms=1e-6)
    ims[1].data = np.full_like(ims[1].data, np.nan)           # all NaN
    ims[3].data = np.zeros_like(ims[3].data); ims[3].noise = 0.0  # all zero, noise 0 (as seen on real data)
    ims[5].beam = (0.0, 0.0, 0.0)                              # no restoring beam
    r = fit_region(ims, region(), opts=FitOptions(nterms=2, ref_freq=1.25e9))
    assert len(r.skipped_bands) == 3 and len(r.freqs) == len(FREQS) - 3
    reasons = " ".join(w for _, w in r.skipped_bands)
    assert "NaN" in reasons and "zero" in reasons and "beam" in reasons
    c = r.components[0]
    assert max(map(abs, sep_arcsec(c, ra, dec))) < 1e-3
    assert np.isclose(c.spectral.spectrum.coeffs[1], -0.7, atol=1e-3)


def test_all_subbands_flagged_gives_clear_error():
    ra, dec = offset(0.0, 0.0)
    ims = make_subbands([TrueComponent(ra, dec, spec)], FREQS[:2], ra0=RA0, dec0=DEC0, npix=64, rms=1e-6)
    for im in ims:
        im.data = np.zeros_like(im.data)
    with pytest.raises(ValueError, match="no usable sub-bands"):
        fit_region(ims, region())


def test_subband_problem_checks():
    from pinpoint.images import subband_problem
    good = np.ones((4, 4))
    beam = (1e-5, 1e-5, 0.0)
    assert subband_problem(good, 1e-3, beam) is None
    assert "NaN" in subband_problem(np.full((4, 4), np.nan), 1e-3, beam)
    assert "zero" in subband_problem(np.zeros((4, 4)), 1e-3, beam)
    assert "noise" in subband_problem(good, 0.0, beam)
    assert "noise" in subband_problem(good, float("nan"), beam)
    assert "beam" in subband_problem(good, 1e-3, (0.0, 0.0, 0.0))
    partly = good.copy(); partly[0] = np.nan
    assert subband_problem(partly, 1e-3, beam) is None  # partial NaNs are fine (masked per pixel)
