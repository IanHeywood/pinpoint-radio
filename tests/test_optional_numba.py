"""Optional (needs numba): the compiled kernel.

What these tests establish:
  * the numba-compiled recurrence kernel gives the same answer as the numpy reference
    (so the pure-Python checks of the kernel logic in test_predict.py carry over);
  * a rough throughput number is printed (run pytest with -s to see it).
"""

import time

import numpy as np
import pytest

pytest.importorskip("numba")

from pinpoint.constants import ARCSEC, DEG  # noqa: E402
from pinpoint.predict import predict_stokes_i, prepare_components  # noqa: E402
from pinpoint.skymodel import Component  # noqa: E402
from pinpoint.spectral import Spectrum  # noqa: E402

RA0, DEC0 = 1.0, -30 * DEG


def _comps():
    return [Component("a", "point", RA0 + 600 * ARCSEC, DEC0, Spectrum("logpol", 1.3e9, [5.0, -0.7, -0.1])),
            Component("b", "gauss", RA0, DEC0 + 300 * ARCSEC, Spectrum("pol", 1.3e9, [1.0, -0.5]),
                      maj=4 * ARCSEC, min=2 * ARCSEC, pa=0.3)]


@pytest.mark.parametrize("smear", [False, True])
def test_numba_equals_reference(smear):
    rng = np.random.default_rng(0)
    nrow, nchan = 3000, 512
    uvw = rng.uniform(-8000, 8000, (nrow, 3))
    f = np.linspace(0.856e9, 1.712e9, nchan)
    ca = prepare_components(_comps(), RA0, DEC0, f)
    kw = dict(chan_width=np.full(nchan, 1.67e6), interval=np.full(nrow, 8.0), dec0=DEC0,
              freq_smearing=smear, time_smearing=smear)
    a = predict_stokes_i(uvw, f, ca, backend="numpy", **kw)
    b = predict_stokes_i(uvw, f, ca, backend="numba", **kw)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-11


def test_throughput_report():
    rng = np.random.default_rng(1)
    nrow, nchan = 20000, 4096
    uvw = rng.uniform(-8000, 8000, (nrow, 3))
    f = np.linspace(0.856e9, 1.712e9, nchan)
    ca = prepare_components(_comps(), RA0, DEC0, f)
    predict_stokes_i(uvw[:10], f, ca, backend="numba")  # compile
    t = time.perf_counter()
    predict_stokes_i(uvw, f, ca, backend="numba")
    dt = time.perf_counter() - t
    rate = nrow * nchan * ca.ncomp / dt
    print(f"\nnumba kernel: {rate / 1e6:.1f} M component-visibilities/s ({nrow}x{nchan}x{ca.ncomp} in {dt:.2f}s)")
    assert rate > 0
