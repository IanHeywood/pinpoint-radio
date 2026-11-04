"""End to end: images -> fit -> sky-model file -> predict into an (in-memory) MS.

What this test establishes:
  * for a bright source sitting 0.37 / -0.21 pixels off the grid, the visibilities
    predicted from the fitted model match the TRUE source's visibilities to < 1e-3
    of its flux on 8 km baselines, across 0.9-1.6 GHz;
  * whereas a model snapped to the nearest pixel centre (what a CLEAN component
    list effectively gives you) is wrong by tens of per cent on the long baselines.
  That gap is the dynamic-range problem this tool exists to remove.
"""

import numpy as np

from pinpoint.constants import ARCSEC, DEG
from pinpoint.fitting import FitOptions, fit_region
from pinpoint.ms_io import predict_into_ms
from pinpoint.predict import predict_reference, prepare_components
from pinpoint.regions import parse_region_string
from pinpoint.skymodel import Component, SkyModel
from pinpoint.spectral import Spectrum
from pinpoint.testing.synth_images import TrueComponent, make_subbands
from pinpoint.testing.synth_ms import make_inmemory_ms

RA0, DEC0 = 150 * DEG, -30 * DEG
FREQS = np.linspace(0.9e9, 1.6e9, 8)


def test_images_to_visibilities(tmp_path):
    ra = RA0 - 0.37 * ARCSEC / np.cos(DEC0)
    dec = DEC0 - 0.21 * ARCSEC
    truth = Component("t", "point", ra, dec, Spectrum("logpol", 1.25e9, [2.0, -0.7, -0.2]))
    ims = make_subbands([TrueComponent(ra, dec, truth.spectrum)], FREQS, ra0=RA0, dec0=DEC0, npix=96, rms=1e-6,
                        cell_arcsec=1.0, beam_at_1ghz=(6.0, 4.0, 30.0))
    reg = parse_region_string('fk5\ncircle(150,-30,12") # text={point col=DIR1_DATA name=bright}')[0]
    res = fit_region(ims, reg, opts=FitOptions(nterms=3, ref_freq=1.25e9))
    model = SkyModel([res.to_source()])
    path = tmp_path / "model.yaml"
    model.save(path)
    model = SkyModel.load(path)  # go through the file, as the CLI does

    spw_freqs = np.linspace(0.9e9, 1.6e9, 32)
    ms = make_inmemory_ms(nant=12, ntime=8, ra0=RA0, dec0=DEC0, extent_m=8000.0,
                          spws=[(spw_freqs, np.full(32, 22e6))])
    predict_into_ms(ms, model, log=lambda *a: None)
    rows = np.arange(ms.nrows())
    got = ms.read("DIR1_DATA", rows)[:, :, 0]

    true_v = predict_reference(ms.cols["UVW"], spw_freqs, prepare_components([truth], RA0, DEC0, spw_freqs))
    S = truth.spectrum(spw_freqs)[None, :]
    err_fit = np.max(np.abs(got - true_v) / S)

    # pixel-snapped model: same flux, position rounded to the nearest pixel centre
    g = ims[0].geometry
    x, y = g.world2pix(ra, dec)
    sra, sdec = g.pix2world(np.round(x), np.round(y))
    snapped = Component("s", "point", float(sra), float(sdec), truth.spectrum)
    snap_v = predict_reference(ms.cols["UVW"], spw_freqs, prepare_components([snapped], RA0, DEC0, spw_freqs))
    err_snap = np.max(np.abs(snap_v - true_v) / S)

    assert err_fit < 1e-3, err_fit
    assert err_snap > 0.1, err_snap
