"""Measure time+bandwidth decorrelation at a source position in a real MS.

Run AFTER predicting a 1 Jy flat-spectrum point source into two columns:
  pinpoint predict --model unit.yaml --ms obs.ms --field F --column SMEAR_OFF --no-time-smearing --no-freq-smearing
  pinpoint predict --model unit.yaml --ms obs.ms --field F --column SMEAR_ON

Usage:  python smearing_check.py obs.ms F

For each spectral window, prints per baseline-length bin the median and worst
decorrelation factor ON/OFF (1.0 = none; negative = beyond the first sinc null)
at the lowest, middle and highest channel.
"""

import sys

import numpy as np
from casacore.tables import table

ms, field = sys.argv[1], int(sys.argv[2])
t = table(ms, ack=False)
ddids = np.unique(t.query(f"FIELD_ID=={field}").getcol("DATA_DESC_ID"))

for d in ddids:
    sel = t.query(f"FIELD_ID=={field} && DATA_DESC_ID=={d} && ANTENNA1!=ANTENNA2")
    uvw = sel.getcol("UVW")
    off = sel.getcol("SMEAR_OFF")[..., 0]  # first correlation (XX or RR)
    on = sel.getcol("SMEAR_ON")[..., 0]
    sel.close()
    factor = np.real(on / np.where(np.abs(off) > 0, off, 1.0))  # both share the same phase
    blen_km = np.hypot(uvw[:, 0], uvw[:, 1]) / 1e3
    edges = np.linspace(0, blen_km.max() * 1.0001, 9)
    nchan = factor.shape[1]
    chans = (0, nchan // 2, nchan - 1)
    print(f"\nDATA_DESC_ID {d}  ({nchan} channels)   columns: median / worst")
    print("  baseline (km)   " + "   ".join(f"chan {c:>4d}         " for c in chans))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (blen_km >= lo) & (blen_km < hi)
        if m.any():
            cells = [f"{np.median(factor[m, c]):6.3f} / {np.min(factor[m, c]):6.3f}" for c in chans]
            print(f"  {lo:6.1f}-{hi:6.1f}   " + "     ".join(cells))

t.close()
