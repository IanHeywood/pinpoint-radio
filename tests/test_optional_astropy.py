"""Optional (needs astropy): FITS reading and the WCS cross-check.

What these tests establish:
  * pinpoint's own SIN projection (used by the synthetic tests) agrees with astropy.wcs
    to < 1 micro-arcsec over a wide image, so the synthetic results transfer;
  * wsclean-style 4-D FITS sub-band images (RA, DEC, FREQ, STOKES) are found and read:
    frequency, bandwidth, restoring beam, and noise from the -residual image;
  * fitting through the real FITS path recovers a sub-pixel source position;
  * masking model FITS images writes correctly named outputs with the region zeroed.
"""

import numpy as np
import pytest

pytest.importorskip("astropy")

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from pinpoint.constants import ARCSEC, DEG  # noqa: E402
from pinpoint.coords import SinGeometry, angular_separation, radec_to_lmn  # noqa: E402
from pinpoint.fitting import fit_region  # noqa: E402
from pinpoint.images import AstropyGeometry, load_subband_images  # noqa: E402
from pinpoint.masking import mask_model_images  # noqa: E402
from pinpoint.regions import parse_region_string  # noqa: E402
from pinpoint.testing.synth_images import TrueComponent, make_subbands  # noqa: E402

RA0, DEC0 = 150 * DEG, -30 * DEG


def header(npix, cell_arcsec, freq, bw, beam=None):
    h = fits.Header()
    h["SIMPLE"] = True
    h["BITPIX"] = -32
    h["NAXIS"] = 4
    for i, n in enumerate([npix, npix, 1, 1], start=1):
        h[f"NAXIS{i}"] = n
    h["CTYPE1"], h["CRVAL1"], h["CDELT1"], h["CRPIX1"], h["CUNIT1"] = "RA---SIN", RA0 / DEG, -cell_arcsec / 3600, npix // 2 + 1, "deg"
    h["CTYPE2"], h["CRVAL2"], h["CDELT2"], h["CRPIX2"], h["CUNIT2"] = "DEC--SIN", DEC0 / DEG, cell_arcsec / 3600, npix // 2 + 1, "deg"
    h["CTYPE3"], h["CRVAL3"], h["CDELT3"], h["CRPIX3"], h["CUNIT3"] = "FREQ", freq, bw, 1, "Hz"
    h["CTYPE4"], h["CRVAL4"], h["CDELT4"], h["CRPIX4"] = "STOKES", 1, 1, 1
    h["RADESYS"] = "FK5"
    h["EQUINOX"] = 2000.0
    if beam:
        h["BMAJ"], h["BMIN"], h["BPA"] = beam[0] / DEG, beam[1] / DEG, beam[2] / DEG
    h["BUNIT"] = "JY/BEAM"
    return h


def test_sin_matches_astropy():
    npix, cell = 4000, 2.0  # a 2.2 deg wide image
    h = header(npix, cell, 1.4e9, 1e6)
    a = AstropyGeometry(h)
    s = SinGeometry(npix, npix, RA0, DEC0, -cell * ARCSEC, cell * ARCSEC, npix // 2 + 1, npix // 2 + 1)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, npix - 1, 500), rng.uniform(0, npix - 1, 500)
    ra1, dec1 = a.pix2world(x, y)
    ra2, dec2 = s.pix2world(x, y)
    assert np.max(angular_separation(ra1, dec1, ra2, dec2)) < 1e-6 * ARCSEC
    assert np.isclose(a.pixel_scale, cell * ARCSEC, rtol=1e-9)


def _write_wsclean_set(tmp_path, ims, prefix="img"):
    for i, im in enumerate(ims):
        h = header(im.data.shape[0], im.geometry.pixel_scale / ARCSEC, im.freq, im.bandwidth, im.beam)
        fits.writeto(tmp_path / f"{prefix}-{i:04d}-image.fits", im.data[None, None].astype(np.float32), h)
        rng = np.random.default_rng(i)
        fits.writeto(tmp_path / f"{prefix}-{i:04d}-residual.fits",
                     rng.normal(0, 2e-5, im.data.shape)[None, None].astype(np.float32), h)
        mdl = np.zeros_like(im.data)
        mdl[48, 48] = 1.0
        mdl[10, 10] = 0.5
        fits.writeto(tmp_path / f"{prefix}-{i:04d}-model.fits", mdl[None, None].astype(np.float32), h)
    # an MFS image that must be ignored by the sub-band reader
    fits.writeto(tmp_path / f"{prefix}-MFS-image.fits", ims[0].data[None, None].astype(np.float32),
                 header(ims[0].data.shape[0], 1.0, 1.25e9, 7e8, ims[0].beam))
    return str(tmp_path / prefix)


def test_fits_round_trip_and_fit(tmp_path):
    freqs = np.linspace(0.9e9, 1.6e9, 4)
    ra, dec = RA0 - 0.37 * ARCSEC / np.cos(DEC0), DEC0 - 0.21 * ARCSEC
    ims = make_subbands([TrueComponent(ra, dec, lambda nu: 2.0 * (nu / 1.25e9) ** -0.7)], freqs,
                        ra0=RA0, dec0=DEC0, npix=96, rms=1e-6)
    prefix = _write_wsclean_set(tmp_path, ims)
    loaded = load_subband_images(prefix)
    assert len(loaded) == 4  # MFS ignored
    for a, b in zip(loaded, ims):
        assert np.isclose(a.freq, b.freq) and np.isclose(a.bandwidth, b.bandwidth)
        assert np.allclose(a.beam, b.beam, rtol=1e-6)
        assert np.isclose(a.noise, 2e-5, rtol=0.1)  # from the residual image
    reg = parse_region_string('fk5\ncircle(150,-30,12")')[0]
    c = fit_region(loaded, reg).components[0]
    l, m, _ = radec_to_lmn(c.ra, c.dec, ra, dec)
    assert abs(l) < 2e-3 * ARCSEC and abs(m) < 2e-3 * ARCSEC  # float32 FITS limits this


def test_mask_model_fits(tmp_path):
    freqs = np.linspace(0.9e9, 1.6e9, 2)
    ims = make_subbands([], freqs, ra0=RA0, dec0=DEC0, npix=96, rms=1e-6)
    prefix = _write_wsclean_set(tmp_path, ims)
    regs = parse_region_string('fk5\ncircle(150,-30,5")')
    out = mask_model_images(prefix, regs, str(tmp_path / "masked"), log=lambda *a: None)
    assert [p.split("/")[-1] for p in out] == ["masked-0000-model.fits", "masked-0001-model.fits"]
    d = np.squeeze(fits.getdata(out[0]))
    assert d[48, 48] == 0.0 and d[10, 10] == 0.5
    orig = np.squeeze(fits.getdata(prefix + "-0000-model.fits"))
    assert orig[48, 48] == 1.0  # input left untouched
    assert fits.getheader(out[0])["NAXIS"] == 4 and fits.getheader(out[0])["BMAJ"] > 0  # header intact


def test_mask_model_fits_in_place(tmp_path):
    freqs = np.linspace(0.9e9, 1.6e9, 2)
    ims = make_subbands([], freqs, ra0=RA0, dec0=DEC0, npix=96, rms=1e-6)
    prefix = _write_wsclean_set(tmp_path, ims)
    regs = parse_region_string('fk5\ncircle(150,-30,5")')
    out = mask_model_images(prefix, regs, in_place=True, log=lambda *a: None)
    assert out[0] == prefix + "-0000-model.fits"
    d = np.squeeze(fits.getdata(out[0]))
    assert d[48, 48] == 0.0 and d[10, 10] == 0.5
