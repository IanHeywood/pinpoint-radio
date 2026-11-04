"""The sky-model file: the hand-off between fitting and prediction.

It is plain YAML in human units so it can be inspected and hand-edited:

  sources:
    - name: 3C123
      column: DIR1_DATA
      components:
        - name: 3C123_c0
          type: gauss            # or point
          ra_deg: 69.2682...
          dec_deg: 29.6705...
          maj_arcsec: 1.23       # FWHM, gauss only
          min_arcsec: 0.80
          pa_deg: 35.0           # North through East
          spectrum: {kind: logpol, ref_freq_hz: 1.284e9, coeffs: [12.3, -0.71, -0.05]}

Internally everything is radians / Hz / Jy.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import yaml

from .constants import ARCSEC, DEG
from .spectral import Spectrum

COMPONENT_TYPES = ("point", "gauss")


@dataclass
class Component:
    name: str
    type: str
    ra: float  # rad
    dec: float  # rad
    spectrum: Spectrum
    maj: float = 0.0  # FWHM rad
    min: float = 0.0  # FWHM rad
    pa: float = 0.0  # rad, North through East
    errors: dict = field(default_factory=dict)  # optional 1-sigma errors, human units

    def __post_init__(self):
        if self.type not in COMPONENT_TYPES:
            raise ValueError(f"component type {self.type!r} not in {COMPONENT_TYPES}")
        if self.type == "point":
            self.maj = self.min = self.pa = 0.0
        if self.min > self.maj:
            self.maj, self.min = self.min, self.maj
            self.pa = (self.pa + np.pi / 2) % np.pi


@dataclass
class Source:
    name: str
    components: list[Component]
    column: str | None = None  # MS column to predict into; None = use CLI default


@dataclass
class SkyModel:
    sources: list[Source]
    meta: dict = field(default_factory=dict)

    def components(self):
        for s in self.sources:
            yield from s.components

    def by_column(self, default_column=None, override=None):
        """Map column name -> list of components."""
        out: dict[str, list[Component]] = {}
        for s in self.sources:
            col = override or s.column or default_column
            if col is None:
                raise ValueError(f"source {s.name!r} has no column and no default column was given")
            out.setdefault(col, []).extend(s.components)
        return out

    # ---- serialisation ----
    def to_dict(self):
        def comp(c: Component):
            d = {
                "name": c.name,
                "type": c.type,
                "ra_deg": float(c.ra / DEG),
                "dec_deg": float(c.dec / DEG),
            }
            if c.type == "gauss":
                d.update(maj_arcsec=float(c.maj / ARCSEC), min_arcsec=float(c.min / ARCSEC),
                         pa_deg=float(c.pa / DEG))
            sp = {"kind": c.spectrum.kind, "ref_freq_hz": float(c.spectrum.ref_freq),
                  "coeffs": [float(x) for x in c.spectrum.coeffs]}
            if c.spectrum.coeff_errors is not None:
                sp["coeff_errors"] = [float(x) for x in c.spectrum.coeff_errors]
            d["spectrum"] = sp
            if c.errors:
                d["errors"] = {k: float(v) for k, v in c.errors.items()}
            return d

        meta = dict(self.meta)
        meta.setdefault("created", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        from . import __version__
        meta.setdefault("pinpoint_version", __version__)
        return {
            "meta": meta,
            "sources": [
                {"name": s.name, "column": s.column, "components": [comp(c) for c in s.components]}
                for s in self.sources
            ],
        }

    @classmethod
    def from_dict(cls, d):
        sources = []
        for s in d.get("sources", []):
            comps = []
            for c in s["components"]:
                sp = c["spectrum"]
                spec = Spectrum(sp["kind"], float(sp["ref_freq_hz"]), sp["coeffs"], sp.get("coeff_errors"))
                comps.append(Component(
                    name=c.get("name", f"{s['name']}_c{len(comps)}"),
                    type=c["type"],
                    ra=float(c["ra_deg"]) * DEG,
                    dec=float(c["dec_deg"]) * DEG,
                    spectrum=spec,
                    maj=float(c.get("maj_arcsec", 0.0)) * ARCSEC,
                    min=float(c.get("min_arcsec", 0.0)) * ARCSEC,
                    pa=float(c.get("pa_deg", 0.0)) * DEG,
                    errors=dict(c.get("errors", {})),
                ))
            sources.append(Source(s["name"], comps, s.get("column")))
        return cls(sources, dict(d.get("meta", {})))

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=None)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
