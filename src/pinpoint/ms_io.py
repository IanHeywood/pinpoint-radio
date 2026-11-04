"""Writing predicted visibilities into Measurement Set columns.

All MS access goes through a tiny backend interface (see CasacoreMS below, and
pinpoint.testing.synth_ms.InMemoryMS used by the tests):

    meta()                       -> MSMeta (phase centres, spectral windows, correlations)
    nrows()
    read_scalar_col(name)        -> whole 1-D column (FIELD_ID, DATA_DESC_ID)
    read(name, rows)             -> array for the given rows
    write(name, rows, values)
    has_column(name)
    create_column_like(name, template="DATA")
    flush(); close()

Rows are processed per (FIELD_ID, DATA_DESC_ID) partition, so spectral windows with
different channel counts and different correlation setups are handled naturally, and
each field is predicted relative to its own phase centre.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .predict import predict_stokes_i, prepare_components, stokes_i_to_corr

PROTECTED_COLUMNS = {"DATA", "CORRECTED_DATA"}
MODES = ("overwrite", "add", "subtract")


@dataclass
class MSMeta:
    field_phase_dir: list  # [(ra, dec)] rad, per FIELD row
    ddid_spw: list  # per DATA_DESCRIPTION row
    ddid_pol: list
    spw_freqs: list  # per SPECTRAL_WINDOW row: array Hz
    spw_widths: list
    pol_corr_types: list  # per POLARIZATION row: list of casacore Stokes codes


@dataclass
class PredictReport:
    rows_predicted: int = 0
    rows_zeroed: int = 0
    columns_created: list = field(default_factory=list)
    t_read: float = 0.0
    t_compute: float = 0.0
    t_write: float = 0.0

    def summary(self):
        tot = self.t_read + self.t_compute + self.t_write
        return (f"rows predicted: {self.rows_predicted}, rows zero-filled: {self.rows_zeroed}, "
                f"created: {self.columns_created or 'none'}\n"
                f"time: read {self.t_read:.1f}s  compute {self.t_compute:.1f}s  write {self.t_write:.1f}s"
                f"  (total {tot:.1f}s)")


def _chunks(rows, n):
    for s in range(0, len(rows), n):
        yield rows[s:s + n]


def predict_into_ms(ms, skymodel, *, default_column="MODEL_DATA", override_column=None, mode="overwrite",
                    field_ids=None, chunk_rows=50_000, backend="auto", sign=1.0, freq_smearing=False,
                    time_smearing=False, allow_data_column=False, log=print) -> PredictReport:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    meta = ms.meta()
    fid = np.asarray(ms.read_scalar_col("FIELD_ID"))
    ddid = np.asarray(ms.read_scalar_col("DATA_DESC_ID"))
    present_fields = sorted(set(fid.tolist()))
    if field_ids is None:
        if len(present_fields) > 1:
            raise ValueError(f"MS has several fields {present_fields}; choose with --field "
                             "(the apparent spectrum is specific to one pointing)")
        field_ids = present_fields
    field_ids = set(int(f) for f in field_ids)
    by_col = skymodel.by_column(default_column, override_column)
    rep = PredictReport()

    for col, comps in by_col.items():
        if col in PROTECTED_COLUMNS and not allow_data_column:
            raise ValueError(f"refusing to write into {col}; pass allow_data_column=True "
                             "(CLI: --allow-data-column) if you really mean it")
        created = not ms.has_column(col)
        if created:
            ms.create_column_like(col, "DATA")
            rep.columns_created.append(col)
            if mode != "overwrite":
                log(f"note: {col} did not exist; created it and treated its prior contents as zero")
        log(f"{col}: {len(comps)} component(s), mode={mode}")
        sgn = -1.0 if mode == "subtract" else 1.0
        for f in present_fields:
            for d in sorted(set(ddid[fid == f].tolist())):
                rows = np.flatnonzero((fid == f) & (ddid == d))
                spw, pol = meta.ddid_spw[d], meta.ddid_pol[d]
                freqs = np.asarray(meta.spw_freqs[spw], dtype=np.float64)
                widths = np.asarray(meta.spw_widths[spw], dtype=np.float64)
                corr = meta.pol_corr_types[pol]
                if f not in field_ids:
                    if created:  # never leave an uninitialised column behind
                        for ch in _chunks(rows, chunk_rows):
                            ms.write(col, ch, np.zeros((len(ch), freqs.size, len(corr)), np.complex64))
                        rep.rows_zeroed += len(rows)
                    continue
                ra0, dec0 = meta.field_phase_dir[f]
                ca = prepare_components(comps, ra0, dec0, freqs)
                for ch in _chunks(rows, chunk_rows):
                    t0 = time.perf_counter()
                    uvw = ms.read("UVW", ch)
                    interval = ms.read("EXPOSURE", ch) if time_smearing else None
                    existing = None
                    if mode != "overwrite" and not created:
                        existing = ms.read(col, ch)
                    t1 = time.perf_counter()
                    vis = predict_stokes_i(uvw, freqs, ca, backend=backend, chan_width=widths, interval=interval,
                                           dec0=dec0, sign=sign, freq_smearing=freq_smearing,
                                           time_smearing=time_smearing)
                    out = stokes_i_to_corr(vis, corr, dtype=np.complex128)
                    if existing is not None:
                        out = existing.astype(np.complex128) + sgn * out
                    elif sgn < 0:
                        out = -out
                    out = out.astype(np.complex64)
                    t2 = time.perf_counter()
                    ms.write(col, ch, out)
                    t3 = time.perf_counter()
                    rep.t_read += t1 - t0
                    rep.t_compute += t2 - t1
                    rep.t_write += t3 - t2
                    rep.rows_predicted += len(ch)
        ms.flush()
    return rep


# ---------------- python-casacore backend ----------------

class CasacoreMS:  # pragma: no cover - exercised by tests/test_optional_casacore.py
    def __init__(self, path, readonly=False):
        from casacore.tables import table

        self._table = table
        self.path = str(path)
        self.t = table(self.path, readonly=readonly, ack=False)

    def _sub(self, name):
        return self._table(f"{self.path}::{name}", readonly=True, ack=False)

    def meta(self):
        with self._sub("FIELD") as ft:
            pd = ft.getcol("PHASE_DIR")  # (nfield, npoly, 2)
            phase = [(float(p[0, 0]), float(p[0, 1])) for p in pd]
        with self._sub("DATA_DESCRIPTION") as dd:
            ddid_spw = [int(x) for x in dd.getcol("SPECTRAL_WINDOW_ID")]
            ddid_pol = [int(x) for x in dd.getcol("POLARIZATION_ID")]
        with self._sub("SPECTRAL_WINDOW") as sw:
            freqs = [np.asarray(sw.getcell("CHAN_FREQ", i), dtype=np.float64) for i in range(sw.nrows())]
            widths = [np.asarray(sw.getcell("CHAN_WIDTH", i), dtype=np.float64) for i in range(sw.nrows())]
        with self._sub("POLARIZATION") as pt:
            corr = [list(map(int, pt.getcell("CORR_TYPE", i))) for i in range(pt.nrows())]
        return MSMeta(phase, ddid_spw, ddid_pol, freqs, widths, corr)

    def nrows(self):
        return self.t.nrows()

    def read_scalar_col(self, name):
        return self.t.getcol(name)

    def has_column(self, name):
        return name in self.t.colnames()

    @staticmethod
    def _contiguous(rows):
        return len(rows) > 0 and rows[-1] - rows[0] + 1 == len(rows)

    def read(self, name, rows):
        rows = np.asarray(rows)
        if self._contiguous(rows):
            return self.t.getcol(name, startrow=int(rows[0]), nrow=len(rows))
        sel = self.t.selectrows(rows.tolist())
        try:
            return sel.getcol(name)
        finally:
            sel.close()

    def write(self, name, rows, values):
        rows = np.asarray(rows)
        if self._contiguous(rows):
            self.t.putcol(name, values, startrow=int(rows[0]), nrow=len(rows))
            return
        sel = self.t.selectrows(rows.tolist())
        try:
            sel.putcol(name, values)
        finally:
            sel.close()

    def create_column_like(self, name, template="DATA"):
        from casacore.tables import makecoldesc, maketabdesc

        desc = self.t.getcoldesc(template)
        desc["comment"] = "pinpoint DFT model"
        dm = {k: v for k, v in self.t.getdminfo(template).items() if k != "SEQNR"}
        if "dysco" in str(dm.get("TYPE", "")).lower() or "SPEC" not in dm:
            # never inherit lossy compression (Dysco) for a model column
            dm = {"TYPE": "TiledShapeStMan", "SPEC": {"DEFAULTTILESHAPE": np.array([4, 64, 128], dtype=np.int32)}}
        dm["NAME"] = f"{name}_dm"
        # bind the new column to the new storage manager, not to DATA's
        desc["dataManagerGroup"] = dm["NAME"]
        desc["dataManagerType"] = dm["TYPE"]
        self.t.addcols(maketabdesc(makecoldesc(name, desc)), dm)

    def flush(self):
        self.t.flush()

    def close(self):
        self.t.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
