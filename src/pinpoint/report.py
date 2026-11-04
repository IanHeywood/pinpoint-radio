"""Human-readable fit summaries and optional diagnostic plots."""

from __future__ import annotations

import os

import numpy as np

from .constants import ARCSEC, DEG


def sexagesimal(ra, dec, prec=5):
    h = (np.rad2deg(ra) % 360) / 15
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = ((h - hh) * 60 - mm) * 60
    d = abs(np.rad2deg(dec))
    sign = "-" if dec < 0 else "+"
    dd = int(d)
    dm = int((d - dd) * 60)
    ds = ((d - dd) * 60 - dm) * 60
    return f"{hh:02d}:{mm:02d}:{ss:0{3 + prec}.{prec}f} {sign}{dd:02d}:{dm:02d}:{ds:0{2 + prec}.{prec - 1}f}"


def summarise(result) -> str:
    lines = [f"Source {result.name}  (column: {result.column or 'default'})  "
             f"pixels used: {result.npix}  reduced chi2: {result.chi2_red:.3f}  error scale: {result.error_scale:.2f}"]
    for i, c in enumerate(result.components):
        sp = c.spectral.spectrum
        lines.append(f"  [{i}] {c.type:5s} {sexagesimal(c.ra, c.dec)}   "
                     f"+/- {c.de_err_arcsec * 1e3:.3f} mas (E), {c.dn_err_arcsec * 1e3:.3f} mas (N)")
        if c.type == "gauss":
            lines.append(f"       maj {c.maj / ARCSEC:.4f}\" min {c.min / ARCSEC:.4f}\" pa {c.pa / DEG:.2f} deg "
                         f"(+/- {c.shape_err[0]:.4f}\", {c.shape_err[1]:.4f}\", {c.shape_err[2]:.2f} deg)")
        coeffs = ", ".join(f"{x:.6g}" for x in sp.coeffs)
        errs = ", ".join(f"{x:.2g}" for x in (sp.coeff_errors or []))
        lines.append(f"       spectrum {sp.kind} @ {sp.ref_freq / 1e6:.3f} MHz: [{coeffs}] +/- [{errs}]  "
                     f"(chi2 {c.spectral.chi2:.2f} / dof {c.spectral.dof})")
        if c.band_offsets_arcsec is not None:
            chi = c.band_offsets_arcsec / c.band_offsets_err
            lines.append(f"       per-band position scatter: max |offset/err| = {np.nanmax(np.abs(chi)):.2f}")
    return "\n".join(lines)


def plot_result(result, outdir):  # pragma: no cover - plotting
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    os.makedirs(outdir, exist_ok=True)
    nb = result.data.shape[0]
    show = sorted(set([0, nb // 2, nb - 1]))
    fig, axes = plt.subplots(len(show) + 1, 3, figsize=(11, 3.4 * (len(show) + 1)))
    axes = np.atleast_2d(axes)
    m = result.cutout.mask
    for row, b in enumerate(show):
        d, mod = result.data[b], result.model[b]
        res = np.where(m, d - mod, np.nan)
        vmax = np.nanmax(np.abs(d))
        for ax, img, title in zip(axes[row], (d, mod, res),
                                  (f"data  {result.freqs[b] / 1e6:.1f} MHz", "model", "residual (fit pixels)")):
            lim = vmax if title != "residual (fit pixels)" else np.nanmax(np.abs(res))
            im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[-1, 0]
    f = result.freqs / 1e6
    nu = np.linspace(result.freqs.min(), result.freqs.max(), 200)
    for i, c in enumerate(result.components):
        ax.errorbar(f, c.band_flux, c.band_flux_err, fmt="o", ms=3, label=f"c{i} bands")
        ax.plot(nu / 1e6, c.spectral.spectrum(nu), lw=1)
    ax.set_xlabel("MHz")
    ax.set_ylabel("Jy")
    ax.set_title("apparent spectrum", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[-1, 1]
    for i, c in enumerate(result.components):
        if c.band_offsets_arcsec is not None:
            ax.errorbar(f, c.band_offsets_arcsec[:, 0] * 1e3, c.band_offsets_err[:, 0] * 1e3, fmt="o", ms=3, label=f"c{i} E")
            ax.errorbar(f, c.band_offsets_arcsec[:, 1] * 1e3, c.band_offsets_err[:, 1] * 1e3, fmt="s", ms=3, label=f"c{i} N")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("per-band position - joint (mas)", fontsize=9)
    ax.set_xlabel("MHz")
    ax.legend(fontsize=7)
    axes[-1, 2].axis("off")
    fig.tight_layout()
    path = os.path.join(outdir, f"{result.name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
