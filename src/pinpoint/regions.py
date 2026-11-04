"""Minimal, dependency-free DS9 region parser.

Supported:
  * coordinate systems: fk5 / icrs / j2000 (sky) and image (1-based pixels)
  * shapes: circle, ellipse, box, polygon
  * exclusion regions (leading '-'): pixels inside them are excluded from every fit
  * sky sizes with units: "  (arcsec), '  (arcmin), d (deg); bare numbers = degrees
  * sexagesimal (hh:mm:ss, dd:mm:ss) or decimal-degree centres

The region's text tag says what to fit and where to predict it:
    circle(...)  # text={gauss}
    circle(...)  # text={point+gauss col=DIR1_DATA name=3C123}
Components default to a single point source if no text is given.

DS9 angle convention: degrees counter-clockwise from the image +x axis. With the
usual orientation (North up, East left) +x is West, so the first ellipse/box axis
points along (East, North) = (-cos a, sin a).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .constants import ARCSEC, DEG
from .coords import radec_to_lmn

SKY_FRAMES = {"fk5", "icrs", "j2000", "fk4"}
SHAPES = {"circle", "ellipse", "box", "polygon"}
COMPONENT_WORDS = {"point", "gauss"}


@dataclass
class Region:
    shape: str
    frame: str  # "sky" or "image"
    centre: tuple[float, float]  # (ra, dec) rad for sky; (x, y) 0-based pixels for image
    sizes: tuple = ()  # sky: rad; image: pixels
    angle: float = 0.0  # rad, DS9 convention
    vertices: np.ndarray | None = None  # polygon: (N,2) of (ra,dec) or (x,y)
    components: list[str] = field(default_factory=lambda: ["point"])
    column: str | None = None
    name: str = ""
    exclude: bool = False

    # -- geometry --
    def bounding_radius(self) -> float:
        """Radius (rad for sky, pixels for image) enclosing the region."""
        if self.shape == "circle":
            return self.sizes[0]
        if self.shape == "ellipse":
            return max(self.sizes)
        if self.shape == "box":
            return 0.5 * float(np.hypot(*self.sizes))
        if self.frame == "sky":
            l, m, _ = radec_to_lmn(self.vertices[:, 0], self.vertices[:, 1], *self.centre)
            return float(np.max(np.hypot(l, m)))
        d = self.vertices - np.asarray(self.centre)
        return float(np.max(np.hypot(d[:, 0], d[:, 1])))

    def contains(self, ra, dec, x, y):
        """Boolean mask: which of the given pixels (with sky coords ra, dec and
        0-based pixel coords x, y) lie inside the region."""
        if self.frame == "sky":
            a, b, _ = radec_to_lmn(ra, dec, *self.centre)  # East, North offsets
            ax1 = (-np.cos(self.angle), np.sin(self.angle))
            ax2 = (np.sin(self.angle), np.cos(self.angle))
            if self.vertices is not None:
                vl, vm, _ = radec_to_lmn(self.vertices[:, 0], self.vertices[:, 1], *self.centre)
                poly = np.stack([vl, vm], axis=1)
        else:
            a = np.asarray(x, dtype=np.float64) - self.centre[0]
            b = np.asarray(y, dtype=np.float64) - self.centre[1]
            ax1 = (np.cos(self.angle), np.sin(self.angle))
            ax2 = (-np.sin(self.angle), np.cos(self.angle))
            if self.vertices is not None:
                poly = self.vertices - np.asarray(self.centre)
        if self.shape == "circle":
            return a * a + b * b <= self.sizes[0] ** 2
        p = a * ax1[0] + b * ax1[1]
        q = a * ax2[0] + b * ax2[1]
        if self.shape == "ellipse":
            return (p / self.sizes[0]) ** 2 + (q / self.sizes[1]) ** 2 <= 1.0
        if self.shape == "box":
            return (np.abs(p) <= 0.5 * self.sizes[0]) & (np.abs(q) <= 0.5 * self.sizes[1])
        if self.shape == "polygon":
            return _point_in_polygon(a, b, poly)
        raise ValueError(self.shape)


def _point_in_polygon(px, py, poly):
    px = np.asarray(px)
    py = np.asarray(py)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    return inside


# ---------------- parsing ----------------

def _parse_ra(tok: str) -> float:
    tok = tok.strip()
    if ":" in tok or "h" in tok:
        parts = [float(p) for p in re.split(r"[:hms]", tok) if p != ""]
        h = parts[0] + (parts[1] if len(parts) > 1 else 0) / 60 + (parts[2] if len(parts) > 2 else 0) / 3600
        return h * 15 * DEG
    return float(tok.rstrip("d")) * DEG


def _parse_dec(tok: str) -> float:
    tok = tok.strip()
    if ":" in tok or "d" in tok[:-1]:
        sign = -1.0 if tok.startswith("-") else 1.0
        parts = [abs(float(p)) for p in re.split(r"[:dms']", tok.lstrip("+-")) if p != ""]
        v = parts[0] + (parts[1] if len(parts) > 1 else 0) / 60 + (parts[2] if len(parts) > 2 else 0) / 3600
        return sign * v * DEG
    return float(tok.rstrip("d")) * DEG


def _parse_size(tok: str, frame: str) -> float:
    tok = tok.strip()
    if frame == "image":
        return float(tok.rstrip("i"))
    if tok.endswith('"'):
        return float(tok[:-1]) * ARCSEC
    if tok.endswith("'"):
        return float(tok[:-1]) * 60 * ARCSEC
    if tok.endswith("d"):
        return float(tok[:-1]) * DEG
    if tok.endswith("r"):
        return float(tok[:-1])
    return float(tok) * DEG


def _parse_text(comment: str):
    m = re.search(r"text\s*=\s*\{([^}]*)\}", comment) or re.search(r'text\s*=\s*"([^"]*)"', comment)
    comps, column, name = ["point"], None, ""
    if not m:
        return comps, column, name
    words = m.group(1).split()
    spec = [w for w in words if "=" not in w]
    kv = dict(w.split("=", 1) for w in words if "=" in w)
    if spec:
        comps = []
        for part in "+".join(spec).split("+"):
            part = part.strip().lower()
            if not part:
                continue
            mm = re.fullmatch(r"(\d+)\*?(point|gauss)", part)
            if mm:
                comps += [mm.group(2)] * int(mm.group(1))
            elif part in COMPONENT_WORDS:
                comps.append(part)
            else:
                raise ValueError(f"unknown component type {part!r} in region text {m.group(1)!r}")
    column = kv.get("col") or kv.get("column")
    name = kv.get("name", "")
    return comps, column, name


def parse_region_string(text: str) -> list[Region]:
    regions: list[Region] = []
    frame = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.lower().startswith("global"):
            continue
        code_part, _, comment_part = line.partition("#")
        stmts = [s.strip() for s in code_part.split(";") if s.strip()]
        for i, code in enumerate(stmts):
            comment = comment_part if i == len(stmts) - 1 else ""
            low = code.lower()
            if low in SKY_FRAMES:
                frame = "sky"
                continue
            if low in ("image", "physical"):
                frame = "image"
                continue
            if low.startswith("global"):
                continue
            exclude = code.startswith("-")
            code = code.lstrip("+-").strip()
            mm = re.fullmatch(r"(\w+)\s*\((.*)\)", code)
            if not mm:
                continue
            shape = mm.group(1).lower()
            if shape not in SHAPES:
                raise ValueError(f"unsupported region shape {shape!r}")
            if frame is None:
                raise ValueError("region file has no coordinate system line (fk5 / icrs / image)")
            args = [a for a in re.split(r"\s*,\s*", mm.group(2).strip()) if a]
            regions.append(_build(shape, frame, args, comment, exclude, len(regions)))
    return regions


def _build(shape, frame, args, comment, exclude, idx) -> Region:
    def centre(a, b):
        if frame == "sky":
            return (_parse_ra(a), _parse_dec(b))
        return (float(a) - 1.0, float(b) - 1.0)

    comps, column, name = _parse_text(comment)
    if shape == "polygon":
        pts = [centre(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]
        verts = np.array(pts)
        if frame == "sky":
            vec = np.stack([np.cos(verts[:, 1]) * np.cos(verts[:, 0]),
                            np.cos(verts[:, 1]) * np.sin(verts[:, 0]),
                            np.sin(verts[:, 1])], axis=1).mean(axis=0)
            c = (float(np.arctan2(vec[1], vec[0]) % (2 * np.pi)), float(np.arctan2(vec[2], np.hypot(vec[0], vec[1]))))
        else:
            c = tuple(verts.mean(axis=0))
        reg = Region(shape, frame, c, vertices=verts)
    else:
        c = centre(args[0], args[1])
        if shape == "circle":
            sizes = (_parse_size(args[2], frame),)
            angle = 0.0
        else:
            sizes = (_parse_size(args[2], frame), _parse_size(args[3], frame))
            angle = float(args[4]) * DEG if len(args) > 4 else 0.0
        reg = Region(shape, frame, c, sizes, angle)
    reg.components, reg.column, reg.exclude = comps, column, exclude
    reg.name = name or f"src{idx}"
    return reg


def load_regions(path) -> tuple[list[Region], list[Region]]:
    """Return (sources, exclusions)."""
    with open(path) as f:
        regs = parse_region_string(f.read())
    return [r for r in regs if not r.exclude], [r for r in regs if r.exclude]
