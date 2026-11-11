"""Synthetic interferometer geometry and an in-memory stand-in for a Measurement Set.

`uvw_from_hour_angle` is the textbook relation (Thompson, Moran & Swenson eq. 4.1)
between equatorial baseline coordinates (X, Y, Z) and (u, v, w) for a phase centre at
hour angle H and declination d0. It's used to make realistic uvw tracks and, in the
time-smearing test, to brute-force average visibilities over an integration.

`InMemoryMS` implements the same small interface as pinpoint.ms_io.CasacoreMS, so all
of the MS-writing logic (row selection, multiple spectral windows, correlation
mapping, write modes, column creation) can be tested without python-casacore.
"""

from __future__ import annotations

import numpy as np


def uvw_from_hour_angle(xyz, H, dec0):
    """xyz: (nbl, 3) equatorial baselines (m); H: hour angle(s) (rad). Returns (..., nbl, 3)."""
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    H = np.asarray(H)[..., None]
    sH, cH = np.sin(H), np.cos(H)
    sd, cd = np.sin(dec0), np.cos(dec0)
    u = sH * X + cH * Y
    v = -sd * cH * X + sd * sH * Y + cd * Z
    w = cd * cH * X - cd * sH * Y + sd * Z
    return np.stack([u, v, w], axis=-1)


def random_array(nant=16, extent_m=8000.0, lat=-30.7 * np.pi / 180, rng=None):
    """Antenna positions in a local ENU disc, converted to equatorial XYZ (m)."""
    rng = rng or np.random.default_rng(0)
    r = extent_m * np.sqrt(rng.uniform(0, 1, nant)) / 2
    th = rng.uniform(0, 2 * np.pi, nant)
    e, n = r * np.cos(th), r * np.sin(th)
    up = rng.normal(0, 5, nant)
    X = -np.sin(lat) * n + np.cos(lat) * up
    Y = e
    Z = np.cos(lat) * n + np.sin(lat) * up
    return np.stack([X, Y, Z], axis=1)


def baselines(ant_xyz):
    a1, a2 = np.triu_indices(len(ant_xyz), 1)
    return a1, a2, ant_xyz[a2] - ant_xyz[a1]


class InMemoryMS:
    """Minimal in-memory Measurement Set.

    spws: list of (chan_freqs array, chan_width array)
    pols: list of corr_type lists (casacore codes, e.g. [9,10,11,12] for XX,XY,YX,YY)
    ddids: list of (spw_id, pol_id)
    fields: list of (ra, dec) phase centres (rad)
    """

    def __init__(self, uvw, field_id, ddid, exposure, spws, pols, ddids, fields, time=None):
        self.cols = {
            "UVW": np.asarray(uvw, dtype=np.float64),
            "FIELD_ID": np.asarray(field_id, dtype=np.int32),
            "DATA_DESC_ID": np.asarray(ddid, dtype=np.int32),
            "EXPOSURE": np.asarray(exposure, dtype=np.float64),
            "INTERVAL": np.asarray(exposure, dtype=np.float64),
            "FLAG_ROW": np.zeros(len(field_id), dtype=bool),
        }
        if time is not None:
            self.cols["TIME"] = np.asarray(time, dtype=np.float64)
        self.spws = spws
        self.pols = pols
        self.ddids = ddids
        self.fields = fields
        self.nrow = len(field_id)
        # variable-shape data columns: dict row -> array, emulated per row
        self.data_cols: dict[str, list] = {}
        self.add_data_column("DATA", fill=None)
        # standard MS columns used by fit-uv: per-row WEIGHT (ncorr) and FLAG (nchan, ncorr)
        self.data_cols["WEIGHT"] = [np.ones(self.row_shape(r)[1], dtype=np.float32) for r in range(self.nrow)]
        self.data_cols["FLAG"] = [np.zeros(self.row_shape(r), dtype=bool) for r in range(self.nrow)]
        self.created = []

    # --- helpers for tests ---
    def row_shape(self, r):
        spw, pol = self.ddids[int(self.cols["DATA_DESC_ID"][r])]
        return (len(self.spws[spw][0]), len(self.pols[pol]))

    def add_data_column(self, name, fill=0.0):
        rows = []
        rng = np.random.default_rng(len(self.data_cols))
        for r in range(self.nrow):
            shp = self.row_shape(r)
            if fill is None:
                rows.append((rng.normal(size=shp) + 1j * rng.normal(size=shp)).astype(np.complex64))
            else:
                rows.append(np.full(shp, fill, dtype=np.complex64))
        self.data_cols[name] = rows

    # --- MS backend interface ---
    def meta(self):
        from ..ms_io import MSMeta

        return MSMeta(
            field_phase_dir=[tuple(f) for f in self.fields],
            ddid_spw=[d[0] for d in self.ddids],
            ddid_pol=[d[1] for d in self.ddids],
            spw_freqs=[np.asarray(s[0], dtype=np.float64) for s in self.spws],
            spw_widths=[np.asarray(s[1], dtype=np.float64) for s in self.spws],
            pol_corr_types=[list(p) for p in self.pols],
        )

    def nrows(self):
        return self.nrow

    def read_scalar_col(self, name):
        return self.cols[name].copy()

    def has_column(self, name):
        return name in self.data_cols or name in self.cols

    def create_column_like(self, name, template="DATA"):
        self.add_data_column(name, fill=np.nan)  # deliberately garbage until written
        self.created.append(name)

    def read(self, name, rows):
        rows = np.asarray(rows)
        if name in self.cols:
            return self.cols[name][rows].copy()
        return np.stack([self.data_cols[name][r] for r in rows])

    def write(self, name, rows, values):
        for i, r in enumerate(np.asarray(rows)):
            assert self.data_cols[name][r].shape == values[i].shape, "shape mismatch writing row"
            self.data_cols[name][r] = np.asarray(values[i], dtype=self.data_cols[name][r].dtype).copy()

    def flush(self):
        pass

    def close(self):
        pass


def make_inmemory_ms(*, nant=8, ntime=6, dec0=-30 * np.pi / 180, ra0=1.0, int_time=8.0,
                     spws=None, pols=None, ddids=None, fields=None, rng=None, extent_m=8000.0,
                     lat=-30.7 * np.pi / 180, hour_angle_range=(-0.5, 0.5)):
    """Rows ordered by time, then ddid, then baseline (a typical MS layout).
    hour_angle_range is in radians; lat is the array latitude (rad)."""
    rng = rng or np.random.default_rng(1)
    ant = random_array(nant, extent_m, lat=lat, rng=rng)
    _, _, bl = baselines(ant)
    if spws is None:
        spws = [(np.linspace(0.9e9, 1.0e9, 16), np.full(16, 0.1e9 / 15))]
    if pols is None:
        pols = [[9, 10, 11, 12]]
    if ddids is None:
        ddids = [(0, 0)]
    if fields is None:
        fields = [(ra0, dec0)]
    H = np.linspace(hour_angle_range[0], hour_angle_range[1], ntime)
    uvw, fid, did, exp, tim = [], [], [], [], []
    for it, h in enumerate(H):
        for f, (_, fdec) in enumerate(fields):
            for d in range(len(ddids)):
                uvw.append(uvw_from_hour_angle(bl, h, fdec))
                n = len(bl)
                fid += [f] * n
                did += [d] * n
                exp += [int_time] * n
                tim += [it * int_time] * n
    uvw = np.concatenate(uvw)
    return InMemoryMS(uvw, fid, did, exp, spws, pols, ddids, fields, time=tim)


def make_emerlin_like_ms(*, ntime=48, int_time=8.0, nchan=16, chan_width=1e6, fmin=1.3e9, fmax=1.7e9,
                         ra0=260.0 * np.pi / 180, dec0=65.8 * np.pi / 180, rng=None):
    """7 antennas spread over ~217 km at latitude 52 deg, a long track on a high-declination
    field, linear feeds. Channel spacing can exceed chan_width (i.e. a sub-sampled band)."""
    freqs = np.linspace(fmin, fmax, nchan)
    return make_inmemory_ms(nant=7, ntime=ntime, ra0=ra0, dec0=dec0, int_time=int_time,
                            spws=[(freqs, np.full(nchan, chan_width))], extent_m=217e3,
                            lat=52.0 * np.pi / 180, hour_angle_range=(-2.5, 2.5),
                            rng=rng or np.random.default_rng(7))


def fill_column(ms, comps, column="DATA", *, sigma=0.0, rng=None, time_smearing=True, freq_smearing=True,
                add_to=None):
    """Write the (smeared) visibilities of `comps` into `column` of an InMemoryMS, optionally
    adding complex Gaussian noise (sigma per real and imaginary part, every correlation)
    and optionally adding to the current contents of column `add_to`. WEIGHT is set to 1/sigma^2."""
    from ..predict import predict_reference, prepare_components, stokes_i_to_corr

    rng = rng or np.random.default_rng(0)
    meta = ms.meta()
    if not ms.has_column(column):
        ms.add_data_column(column, fill=0.0)
    for f in sorted(set(ms.cols["FIELD_ID"].tolist())):
        ra0, dec0 = meta.field_phase_dir[f]
        for d in sorted(set(ms.cols["DATA_DESC_ID"].tolist())):
            rows = np.flatnonzero((ms.cols["FIELD_ID"] == f) & (ms.cols["DATA_DESC_ID"] == d))
            spw, pol = meta.ddid_spw[d], meta.ddid_pol[d]
            freqs, widths = meta.spw_freqs[spw], meta.spw_widths[spw]
            ca = prepare_components(comps, ra0, dec0, freqs)
            v = predict_reference(ms.cols["UVW"][rows], freqs, ca, chan_width=widths,
                                  interval=ms.cols["EXPOSURE"][rows], dec0=dec0,
                                  time_smearing=time_smearing, freq_smearing=freq_smearing)
            out = stokes_i_to_corr(v, meta.pol_corr_types[pol], dtype=np.complex128)
            if sigma > 0:
                out = out + sigma * (rng.normal(size=out.shape) + 1j * rng.normal(size=out.shape))
            if add_to is not None:
                out = out + ms.read(add_to, rows)
            for i, r in enumerate(rows):
                ms.data_cols[column][r] = out[i].astype(np.complex64)
                if sigma > 0:
                    ms.data_cols["WEIGHT"][r] = np.full(out.shape[2], 1.0 / sigma**2, dtype=np.float32)
