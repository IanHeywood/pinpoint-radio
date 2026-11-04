"""Write an InMemoryMS out as a real (minimal) Measurement Set with python-casacore.

Used by tests/test_optional_casacore.py to check that the casacore adapter behaves
exactly like the in-memory MS the rest of the MS tests use.
"""

from __future__ import annotations

import numpy as np


def write_casacore_ms(path, mem):  # pragma: no cover - needs python-casacore
    from casacore.tables import default_ms, makearrcoldesc, maketabdesc, table

    data_desc = makearrcoldesc("DATA", 0j, valuetype="complex", ndim=2, comment="synthetic data")
    t = default_ms(str(path), maketabdesc([data_desc]))
    n = mem.nrows()
    t.addrows(n)
    t.putcol("UVW", mem.cols["UVW"])
    t.putcol("FIELD_ID", mem.cols["FIELD_ID"])
    t.putcol("DATA_DESC_ID", mem.cols["DATA_DESC_ID"])
    t.putcol("EXPOSURE", mem.cols["EXPOSURE"])
    t.putcol("INTERVAL", mem.cols["EXPOSURE"])
    if "TIME" in mem.cols:
        t.putcol("TIME", mem.cols["TIME"])
    for r in range(n):
        t.putcell("DATA", r, mem.data_cols["DATA"][r])
        shp = mem.data_cols["DATA"][r].shape
        t.putcell("FLAG", r, np.zeros(shp, dtype=bool))
    t.close()

    with table(f"{path}/SPECTRAL_WINDOW", readonly=False, ack=False) as sw:
        sw.addrows(len(mem.spws))
        for i, (f, w) in enumerate(mem.spws):
            sw.putcell("CHAN_FREQ", i, np.asarray(f, float))
            sw.putcell("CHAN_WIDTH", i, np.asarray(w, float))
            sw.putcell("EFFECTIVE_BW", i, np.abs(np.asarray(w, float)))
            sw.putcell("RESOLUTION", i, np.abs(np.asarray(w, float)))
            sw.putcell("NUM_CHAN", i, len(f))
            sw.putcell("REF_FREQUENCY", i, float(np.mean(f)))
            sw.putcell("TOTAL_BANDWIDTH", i, float(np.sum(np.abs(w))))
    with table(f"{path}/POLARIZATION", readonly=False, ack=False) as pt:
        pt.addrows(len(mem.pols))
        for i, c in enumerate(mem.pols):
            pt.putcell("CORR_TYPE", i, np.asarray(c, np.int32))
            pt.putcell("NUM_CORR", i, len(c))
    with table(f"{path}/DATA_DESCRIPTION", readonly=False, ack=False) as dd:
        dd.addrows(len(mem.ddids))
        for i, (s, p) in enumerate(mem.ddids):
            dd.putcell("SPECTRAL_WINDOW_ID", i, s)
            dd.putcell("POLARIZATION_ID", i, p)
    with table(f"{path}/FIELD", readonly=False, ack=False) as ft:
        ft.addrows(len(mem.fields))
        for i, (ra, dec) in enumerate(mem.fields):
            d = np.array([[ra, dec]], dtype=float)
            for col in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
                ft.putcell(col, i, d)
    return str(path)
