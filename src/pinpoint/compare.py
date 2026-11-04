"""Compare two sky models (e.g. the fit, and a re-fit after imaging the predicted column)."""

from __future__ import annotations

import numpy as np

from .constants import ARCSEC
from .coords import angular_separation, radec_to_lmn


def compare_models(a, b):
    """For each component of `a`, the nearest component of `b`. Returns a list of dicts."""
    bcomps = list(b.components())
    rows = []
    for ca in a.components():
        if not bcomps:
            break
        seps = [angular_separation(ca.ra, ca.dec, cb.ra, cb.dec) for cb in bcomps]
        cb = bcomps[int(np.argmin(seps))]
        dl, dm, _ = radec_to_lmn(cb.ra, cb.dec, ca.ra, ca.dec)
        nu = ca.spectrum.ref_freq
        row = {
            "a": ca.name, "b": cb.name,
            "dRA_mas": float(dl / ARCSEC * 1e3), "dDec_mas": float(dm / ARCSEC * 1e3),
            "sep_mas": float(np.min(seps) / ARCSEC * 1e3),
            "flux_ratio": float(cb.spectrum(nu) / ca.spectrum(nu)),
        }
        if ca.errors.get("ra_arcsec"):
            row["dRA_sigma"] = row["dRA_mas"] / 1e3 / ca.errors["ra_arcsec"]
            row["dDec_sigma"] = row["dDec_mas"] / 1e3 / ca.errors["dec_arcsec"]
        if ca.type == "gauss" and cb.type == "gauss":
            row["maj_ratio"] = cb.maj / ca.maj
            row["min_ratio"] = cb.min / ca.min
            row["dPA_deg"] = float(np.rad2deg((cb.pa - ca.pa + np.pi / 2) % np.pi - np.pi / 2))
        rows.append(row)
    return rows


def format_comparison(rows):
    lines = []
    for r in rows:
        s = (f"{r['a']} -> {r['b']}: dRA {r['dRA_mas']:+.3f} mas, dDec {r['dDec_mas']:+.3f} mas, "
             f"flux ratio {r['flux_ratio']:.5f}")
        if "dRA_sigma" in r:
            s += f"  ({r['dRA_sigma']:+.2f}, {r['dDec_sigma']:+.2f} sigma)"
        if "maj_ratio" in r:
            s += f", maj x{r['maj_ratio']:.4f}, min x{r['min_ratio']:.4f}, dPA {r['dPA_deg']:+.2f} deg"
        lines.append(s)
    return "\n".join(lines)
