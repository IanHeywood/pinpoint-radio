"""Visibility-domain refinement of a sky model, with exact smearing in the model.

Why: when a source is strongly time/bandwidth smeared in the data (e.g. e-MERLIN with
8 s integrations a few arcmin from the phase centre), the restored image shows a smeared
source and an image-plane fit returns biased flux, spectrum and shape. Here the model is
compared with the visibilities themselves, using the same predictor (and the same
smearing factors) as `pinpoint predict`, so the fitted parameters describe the source
before smearing.

What is fitted (each optional), per component:
  * position: offsets (mas, East/North) from the starting position
  * spectrum: all spectral coefficients (number of terms may be changed)
  * shape   : Gaussian components only, via a Cholesky factor (mas) so it can shrink to a point

Data used:  D - R  where D = data column (default CORRECTED_DATA) and R = an optional
column holding the rest-of-field model (default MODEL_DATA, e.g. from wsclean -predict on
masked model images). Only parallel hands (XX/YY or RR/LL) are used, each as a measurement
of Stokes I. FLAG / FLAG_ROW are honoured; weights come from WEIGHT_SPECTRUM if present,
else WEIGHT.

The fit works in the ORIGINAL phase-centre frame. It does not phase-rotate to the source:
the decorrelation happened at averaging time relative to the original phase centre, so
rotating afterwards would not undo it, and the exact model is only exact in that frame.
For speed, use every Nth channel / row (`chan_step`, `row_step`): sub-sampling keeps the
model exact, whereas averaging would add further smearing.

Optimiser: Levenberg-Marquardt with normal equations accumulated in row chunks (so memory
stays small) and forward-difference derivatives. Errors come from the inverse normal
matrix scaled by the reduced chi-squared (so the absolute scale of the MS weights does not
matter).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .constants import ARCSEC, C, DEG
from .coords import lmn_to_radec, radec_to_lmn
from .ms_io import integration_times
from .predict import predict_stokes_i, prepare_components, stokes_i_weights
from .shapes import cov_to_shape, shape_to_cov
from .skymodel import Component, SkyModel, Source
from .spectral import Spectrum

MAS = 1e-3 * ARCSEC


@dataclass
class UVFitOptions:
    data_column: str = "CORRECTED_DATA"
    subtract_column: str | None = "MODEL_DATA"
    field_id: int | None = None
    chan_step: int = 1
    row_step: int = 1
    time_smearing: bool = True
    freq_smearing: bool = True
    fit_position: bool = True
    fit_spectrum: bool = True
    fit_shape: bool = True
    nterms: int | None = None  # None = keep the starting model's number of terms
    max_iter: int = 50
    chunk_rows: int = 20_000
    backend: str = "auto"
    sign: float = 1.0


@dataclass
class UVPartition:
    ra0: float
    dec0: float
    freqs: np.ndarray
    widths: np.ndarray
    uvw: np.ndarray
    interval: np.ndarray
    W: np.ndarray  # (n, nc) summed weight over parallel hands
    Rbar: np.ndarray  # (n, nc) complex: weighted mean over parallel hands of (D - R)
    chi2_const: float  # sum_corr w |d - Rbar|^2: the part of chi2 no model can remove
    nmeas: int  # number of (real) measurements with non-zero weight


# ---------------------------------------------------------------- data

def load_uv_data(ms, opts: UVFitOptions, log=print) -> list[UVPartition]:
    meta = ms.meta()
    fid = np.asarray(ms.read_scalar_col("FIELD_ID"))
    ddid = np.asarray(ms.read_scalar_col("DATA_DESC_ID"))
    fields = sorted(set(fid.tolist()))
    f = opts.field_id
    if f is None:
        if len(fields) > 1:
            raise ValueError(f"MS has several fields {fields}; choose one with --field")
        f = fields[0]
    if not ms.has_column(opts.data_column):
        raise ValueError(f"data column {opts.data_column!r} not in MS")
    if opts.subtract_column is not None and not ms.has_column(opts.subtract_column):
        raise ValueError(f"subtract column {opts.subtract_column!r} not in MS "
                         "(predict the rest-of-field model first, or use --no-subtract)")
    has_ws = ms.has_column("WEIGHT_SPECTRUM")
    has_fr = ms.has_column("FLAG_ROW")
    ra0, dec0 = meta.field_phase_dir[f]
    parts = []
    for d in sorted(set(ddid[fid == f].tolist())):
        rows_all = np.flatnonzero((fid == f) & (ddid == d))[::opts.row_step]
        spw, pol = meta.ddid_spw[d], meta.ddid_pol[d]
        ci = np.arange(0, len(meta.spw_freqs[spw]), opts.chan_step)
        freqs = np.asarray(meta.spw_freqs[spw], dtype=np.float64)[ci]
        widths = np.asarray(meta.spw_widths[spw], dtype=np.float64)[ci]
        par = np.flatnonzero(stokes_i_weights(meta.pol_corr_types[pol]) > 0)
        Ws, WRs, uvws, ivs = [], [], [], []
        s2, nmeas = 0.0, 0
        for s in range(0, len(rows_all), opts.chunk_rows):
            rows = rows_all[s:s + opts.chunk_rows]
            vis = np.asarray(ms.read(opts.data_column, rows))[:, ci][:, :, par].astype(np.complex128)
            if opts.subtract_column is not None:
                vis = vis - np.asarray(ms.read(opts.subtract_column, rows))[:, ci][:, :, par]
            flag = np.asarray(ms.read("FLAG", rows))[:, ci][:, :, par]
            if has_fr:
                flag = flag | np.asarray(ms.read("FLAG_ROW", rows))[:, None, None]
            if has_ws:
                w = np.asarray(ms.read("WEIGHT_SPECTRUM", rows), dtype=np.float64)[:, ci][:, :, par]
            else:
                w = np.broadcast_to(np.asarray(ms.read("WEIGHT", rows), dtype=np.float64)[:, None, par],
                                    vis.shape).copy()
            bad = flag | ~np.isfinite(vis) | ~np.isfinite(w) | (w < 0)
            w = np.where(bad, 0.0, w)
            vis = np.where(bad, 0.0, vis)
            keep = w.sum(axis=(1, 2)) > 0
            if not keep.any():
                continue
            vis, w, rows = vis[keep], w[keep], rows[keep]
            Wsum = w.sum(axis=2)
            rbar = (w * vis).sum(axis=2) / np.where(Wsum > 0, Wsum, 1.0)
            Ws.append(Wsum.astype(np.float32))  # single precision halves memory; sums stay float64
            WRs.append(rbar.astype(np.complex64))
            s2 += float(np.sum(w * np.abs(vis - rbar[:, :, None]) ** 2))
            nmeas += 2 * int(np.count_nonzero(w))
            uvws.append(np.asarray(ms.read("UVW", rows), dtype=np.float64))
            ivs.append(integration_times(ms, rows))
        if not Ws:
            continue
        parts.append(UVPartition(ra0, dec0, freqs, widths, np.concatenate(uvws), np.concatenate(ivs),
                                 np.concatenate(Ws), np.concatenate(WRs), s2, nmeas))
    if not parts:
        raise ValueError("no unflagged data with non-zero weight for this field")
    nvis = sum(p.W.size for p in parts)
    log(f"loaded {len(parts)} spectral window(s), {sum(len(p.uvw) for p in parts)} rows, "
        f"{nvis} (row, channel) samples")
    return parts


def resolution(parts) -> float:
    """Approximate synthesised-beam size (rad) of the loaded data."""
    bmax = max(np.max(np.hypot(p.uvw[:, 0], p.uvw[:, 1])) for p in parts)
    numax = max(np.max(p.freqs) for p in parts)
    return C / (numax * max(bmax, 1.0))


# ---------------------------------------------------------------- parameters

class _Layout:
    def __init__(self, comps, opts):
        self.start = comps
        self.opts = opts
        self.blocks = []  # (component index, kind, slice)
        idx = 0
        self.nterms = []
        for k, c in enumerate(comps):
            nt = opts.nterms or len(c.spectrum.coeffs)
            self.nterms.append(nt)
            if opts.fit_position:
                self.blocks.append((k, "pos", slice(idx, idx + 2)))
                idx += 2
            if opts.fit_spectrum:
                self.blocks.append((k, "spec", slice(idx, idx + nt)))
                idx += nt
            if c.type == "gauss" and opts.fit_shape:
                self.blocks.append((k, "shape", slice(idx, idx + 3)))
                idx += 3
        self.n = idx

    def initial(self):
        p = np.zeros(self.n)
        for k, kind, s in self.blocks:
            c = self.start[k]
            if kind == "spec":
                co = list(c.spectrum.coeffs)[:self.nterms[k]]
                co += [0.0] * (self.nterms[k] - len(co))
                p[s] = co
            elif kind == "shape":
                cov = shape_to_cov(max(c.maj, 1e-3 * MAS) / MAS, max(c.min, 1e-3 * MAS) / MAS, c.pa)
                L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
                p[s] = (L[0, 0], L[1, 0], L[1, 1])
        return p

    def components(self, p):
        out = []
        for k, c in enumerate(self.start):
            ra, dec = c.ra, c.dec
            coeffs = list(c.spectrum.coeffs)[:self.nterms[k]]
            coeffs += [0.0] * (self.nterms[k] - len(coeffs))
            maj, mn, pa = c.maj, c.min, c.pa
            for kk, kind, s in self.blocks:
                if kk != k:
                    continue
                if kind == "pos":
                    ra, dec = lmn_to_radec(p[s][0] * MAS, p[s][1] * MAS, c.ra, c.dec)
                    ra, dec = float(ra), float(dec)
                elif kind == "spec":
                    coeffs = list(p[s])
                elif kind == "shape":
                    a, b, cc = p[s]
                    cov = np.array([[a * a, a * b], [a * b, b * b + cc * cc]])
                    maj, mn, pa = cov_to_shape(cov)
                    maj, mn = maj * MAS, mn * MAS
            out.append(Component(c.name, c.type, ra, dec, Spectrum(c.spectrum.kind, c.spectrum.ref_freq, coeffs),
                                 maj=maj, min=mn, pa=pa))
        return out

    def steps(self, p, res):
        h = np.empty(self.n)
        for k, kind, s in self.blocks:
            if kind == "pos":
                h[s] = 1e-3 * res / MAS
            elif kind == "spec":
                flux_scale = 1e-6 * max(abs(p[s.start]), 1e-6)  # Jy-valued coefficients
                h[s] = flux_scale if self.start[k].spectrum.kind == "pol" else 1e-5  # logpol: dimensionless
                h[s.start] = flux_scale
            elif kind == "shape":
                h[s] = 1e-2 * res / MAS
        return h


# ---------------------------------------------------------------- model / normal equations

def _model_chunk(part, sl, comps, opts):
    ca = prepare_components(comps, part.ra0, part.dec0, part.freqs)
    return predict_stokes_i(part.uvw[sl], part.freqs, ca, backend=opts.backend, chan_width=part.widths,
                            interval=part.interval[sl], dec0=part.dec0, sign=opts.sign,
                            freq_smearing=opts.freq_smearing, time_smearing=opts.time_smearing)


def _chi2(parts, comps, opts):
    chi2 = 0.0
    for part in parts:
        chi2 += part.chi2_const
        for s in range(0, len(part.uvw), opts.chunk_rows):
            sl = slice(s, s + opts.chunk_rows)
            m = _model_chunk(part, sl, comps, opts)
            chi2 += float(np.sum(part.W[sl] * np.abs(part.Rbar[sl] - m) ** 2))
    return chi2


def _normal_equations(parts, lay, p, opts, res):
    h = lay.steps(p, res)
    comps0 = lay.components(p)
    comps_j = []
    for j in range(lay.n):
        pj = p.copy()
        pj[j] += h[j]
        comps_j.append(lay.components(pj))
    A = np.zeros((lay.n, lay.n))
    b = np.zeros(lay.n)
    chi2 = 0.0
    for part in parts:
        chi2 += part.chi2_const
        for s in range(0, len(part.uvw), opts.chunk_rows):
            sl = slice(s, s + opts.chunk_rows)
            W = part.W[sl]
            m0 = _model_chunk(part, sl, comps0, opts)
            r = W * (part.Rbar[sl] - m0)  # sum_corr w (d - m)
            chi2 += float(np.sum(W * np.abs(part.Rbar[sl] - m0) ** 2))
            D = [(_model_chunk(part, sl, comps_j[j], opts) - m0) / h[j] for j in range(lay.n)]
            for j in range(lay.n):
                b[j] += float(np.sum(np.real(np.conj(D[j]) * r)))
                for k in range(j, lay.n):
                    A[j, k] += float(np.sum(W * np.real(np.conj(D[j]) * D[k])))
    A = A + np.triu(A, 1).T
    return A, b, chi2


# ---------------------------------------------------------------- results

@dataclass
class UVFitResult:
    model: SkyModel
    chi2_red: float
    iterations: int
    converged: bool
    covariance: np.ndarray = field(repr=False)
    component_report: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)  # rows of (bl_lo_km, bl_hi_km, data, model, ratio)


def _smearing_alignment_warnings(comps, ra0, dec0, res):
    out = []
    for c in comps:
        if c.type != "gauss" or c.maj < 0.1 * res or c.maj < 1.2 * max(c.min, 1e-30):
            continue
        l, m, _ = radec_to_lmn(c.ra, c.dec, ra0, dec0)
        if np.hypot(l, m) == 0:
            continue
        radial = np.arctan2(l, m) % np.pi  # PA (N through E) of the direction to the phase centre
        for name, ang in (("radial (bandwidth-smearing)", radial), ("tangential (time-smearing)", (radial + np.pi / 2) % np.pi)):
            d = abs((c.pa - ang + np.pi / 2) % np.pi - np.pi / 2)
            if d < 10 * DEG:
                out.append(f"{c.name}: fitted elongation (PA {np.rad2deg(c.pa):.1f} deg, axis ratio "
                           f"{c.maj / c.min:.2f}) lies within {np.rad2deg(d):.1f} deg of the {name} direction - "
                           "possibly unmodelled or mis-modelled smearing rather than real structure "
                           "(real structure can also line up by coincidence)")
    return out


def amplitude_diagnostic(parts, comps, opts, nbins=8):
    """Data vs model, phase-rotated to the brightest component and vector-averaged in
    baseline-length bins (rotation is used ONLY here, for display)."""
    if not comps:
        return []
    ref = comps[int(np.argmax([abs(c.spectrum.coeffs[0]) for c in comps]))]
    bl_all = np.concatenate([np.hypot(p.uvw[:, 0], p.uvw[:, 1]) for p in parts]) / 1e3
    edges = np.linspace(0, bl_all.max() * 1.0001, nbins + 1)
    sums = np.zeros((nbins, 3))  # weight, weighted data, weighted model (real parts after rotation)
    for part in parts:
        l, m, n = radec_to_lmn(ref.ra, ref.dec, part.ra0, part.dec0)
        for s in range(0, len(part.uvw), opts.chunk_rows):
            sl = slice(s, s + opts.chunk_rows)
            uvw = part.uvw[sl]
            tau = uvw[:, 0] * l + uvw[:, 1] * m + uvw[:, 2] * (n - 1)
            rot = np.exp(-1j * opts.sign * 2 * np.pi * tau[:, None] * part.freqs[None, :] / C)
            mod = _model_chunk(part, sl, comps, opts)
            W = part.W[sl]
            dat = part.Rbar[sl]
            idx = np.clip(np.digitize(np.hypot(uvw[:, 0], uvw[:, 1]) / 1e3, edges) - 1, 0, nbins - 1)
            for i in range(nbins):
                sel = idx == i
                if sel.any():
                    w = W[sel]
                    sums[i, 0] += w.sum()
                    sums[i, 1] += np.sum(w * np.real(dat[sel] * rot[sel]))
                    sums[i, 2] += np.sum(w * np.real(mod[sel] * rot[sel]))
    rows = []
    for i in range(nbins):
        if sums[i, 0] > 0:
            d, mo = sums[i, 1] / sums[i, 0], sums[i, 2] / sums[i, 0]
            rows.append((edges[i], edges[i + 1], d, mo, d / mo if mo != 0 else np.nan))
    return rows


def fit_uv(ms, start: SkyModel, opts: UVFitOptions | None = None, log=print) -> UVFitResult:
    opts = opts or UVFitOptions()
    parts = load_uv_data(ms, opts, log=log)
    res = resolution(parts)
    srcs = start.sources
    comps = [c for s in srcs for c in s.components]
    lay = _Layout(comps, opts)
    if lay.n == 0:
        raise ValueError("nothing to fit: all parameter groups are fixed")
    p = lay.initial()
    nmeas = sum(pt.nmeas for pt in parts)
    lam = 1e-3
    A, b, chi2 = _normal_equations(parts, lay, p, opts, res)
    log(f"start: chi2_red = {chi2 / max(nmeas - lay.n, 1):.4f}  ({lay.n} free parameters)")
    converged = False
    it = 0
    for it in range(1, opts.max_iter + 1):
        dA = np.diag(np.diag(A))
        try:
            delta = np.linalg.solve(A + lam * dA + 1e-30 * np.eye(lay.n), b)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A + lam * dA, b, rcond=None)[0]
        chi2_new = _chi2(parts, lay.components(p + delta), opts)
        if chi2_new < chi2:
            p = p + delta
            lam = max(lam / 10, 1e-12)
            rel = (chi2 - chi2_new) / max(chi2, 1e-300)
            A, b, chi2 = _normal_equations(parts, lay, p, opts, res)
            try:
                cov_un = np.linalg.pinv(A)
                small = np.all(delta**2 <= 1e-6 * np.abs(np.diag(cov_un)) * max(chi2 / max(nmeas - lay.n, 1), 1e-12))
            except np.linalg.LinAlgError:
                small = False
            if rel < 1e-12 or small:
                converged = True
                break
        else:
            lam *= 10
            if lam > 1e12:
                converged = True  # no further improvement possible
                break
    if not converged:
        warnings.warn(f"fit-uv did not converge in {opts.max_iter} iterations")
    dof = max(nmeas - lay.n, 1)
    chi2_red = chi2 / dof
    cov = np.linalg.pinv(A) * chi2_red
    log(f"end:   chi2_red = {chi2_red:.4f}  after {it} iteration(s)")

    final = lay.components(p)
    rep = []
    for k, (c0, c1) in enumerate(zip(comps, final)):
        r = {"name": c1.name, "type": c1.type}
        errs = {}
        for kk, kind, s in lay.blocks:
            if kk != k:
                continue
            e = np.sqrt(np.clip(np.diag(cov)[s], 0, None))
            if kind == "pos":
                r["shift_mas"] = (float(p[s][0]), float(p[s][1]))
                r["shift_err_mas"] = (float(e[0]), float(e[1]))
                errs["ra_arcsec"], errs["dec_arcsec"] = float(e[0]) * 1e-3, float(e[1]) * 1e-3
            elif kind == "spec":
                c1.spectrum.coeff_errors = [float(x) for x in e]
            elif kind == "shape":
                draws = np.random.default_rng(1).multivariate_normal(p[s], cov[s, s], size=300, check_valid="ignore")
                shp = []
                for a, bb, cc in draws:
                    shp.append(cov_to_shape(np.array([[a * a, a * bb], [a * bb, bb * bb + cc * cc]])))
                shp = np.array(shp)
                errs["maj_arcsec"] = float(np.std(shp[:, 0])) * 1e-3
                errs["min_arcsec"] = float(np.std(shp[:, 1])) * 1e-3
                dpa = (shp[:, 2] - c1.pa + np.pi / 2) % np.pi - np.pi / 2
                errs["pa_deg"] = float(np.rad2deg(np.std(dpa)))
        c1.errors = errs
        r["flux_ref"] = float(c1.spectrum.coeffs[0])
        r["flux_start"] = float(c0.spectrum.coeffs[0])
        rep.append(r)

    # rebuild the sky model with the same source grouping / columns
    out_sources, i = [], 0
    for s in srcs:
        n = len(s.components)
        out_sources.append(Source(s.name, final[i:i + n], s.column))
        i += n
    meta = dict(start.meta)
    meta["fit_uv"] = {"data_column": opts.data_column, "subtract_column": opts.subtract_column,
                      "time_smearing": opts.time_smearing, "freq_smearing": opts.freq_smearing,
                      "chan_step": opts.chan_step, "row_step": opts.row_step, "chi2_red": float(chi2_red)}
    model = SkyModel(out_sources, meta)
    warn = _smearing_alignment_warnings(final, parts[0].ra0, parts[0].dec0, res)
    diag = amplitude_diagnostic(parts, final, opts)
    return UVFitResult(model, float(chi2_red), it, converged, cov, rep, warn, diag)


def format_uv_report(r: UVFitResult) -> str:
    lines = [f"reduced chi2 {r.chi2_red:.4f}, {r.iterations} iteration(s), "
             f"{'converged' if r.converged else 'NOT converged'}"]
    for c in r.component_report:
        s = f"  {c['name']:14s} {c['type']:5s} flux@ref {c['flux_start']:.6g} -> {c['flux_ref']:.6g} Jy"
        if "shift_mas" in c:
            (dx, dy), (ex, ey) = c["shift_mas"], c["shift_err_mas"]
            s += f"   moved {dx:+.3f} +/- {ex:.3f} mas E, {dy:+.3f} +/- {ey:.3f} mas N"
        lines.append(s)
    for w in r.warnings:
        lines.append("  WARNING: " + w)
    if r.diagnostics:
        lines.append("  data vs model (rotated to brightest component, vector-averaged real part):")
        lines.append("    baseline (km)        data      model   data/model")
        for lo, hi, d, m, q in r.diagnostics:
            lines.append(f"    {lo:7.1f}-{hi:7.1f}  {d:9.5f}  {m:9.5f}   {q:8.4f}")
    return "\n".join(lines)
