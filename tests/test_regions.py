"""Stage: region files (pinpoint/regions.py, pinpoint/images.region_cutout).

What these tests establish:
  * DS9 syntax variants parse (sexagesimal / degrees, units, 'fk5;' prefixes);
  * the text tag sets components, target column and name;
  * exclusion regions are recognised;
  * masks select the right pixels for circle / ellipse / box / polygon, in both sky
    and image coordinates, and the ellipse angle follows DS9's convention;
  * cutouts are centred on the region and honour exclusions.
"""

import numpy as np
import pytest

from pinpoint.constants import ARCSEC, DEG
from pinpoint.coords import SinGeometry, angular_separation
from pinpoint.images import region_cutout
from pinpoint.regions import parse_region_string

RA0, DEC0 = 150.0 * DEG, -30.0 * DEG


def geom():
    return SinGeometry(200, 200, RA0, DEC0, -1.0 * ARCSEC, 1.0 * ARCSEC, 101.0, 101.0)


def test_parse_sexagesimal_and_units():
    txt = """# Region file format: DS9 version 4.1
global color=green
fk5
circle(10:00:00.000,-30:00:00.00,15")  # text={gauss col=DIR1_DATA name=bright}
ellipse(150.01,-30.01,10",0.1',45) # text={point+gauss}
fk5;box(150.0d,-29.99d,20",10",0)
-circle(150.0,-30.0,2")
"""
    regs = parse_region_string(txt)
    assert len(regs) == 4
    c = regs[0]
    assert c.shape == "circle" and np.isclose(c.sizes[0], 15 * ARCSEC)
    assert angular_separation(c.centre[0], c.centre[1], RA0, DEC0) < 1e-3 * ARCSEC
    assert c.components == ["gauss"] and c.column == "DIR1_DATA" and c.name == "bright"
    e = regs[1]
    assert e.components == ["point", "gauss"] and np.isclose(e.sizes[1], 6 * ARCSEC)
    assert np.isclose(e.angle, 45 * DEG)
    assert regs[2].shape == "box" and regs[2].components == ["point"]
    assert regs[3].exclude


def test_multiplicity_syntax():
    r = parse_region_string('fk5\ncircle(150,-30,5") # text={2point+gauss}')[0]
    assert r.components == ["point", "point", "gauss"]


def test_bad_component_word():
    with pytest.raises(ValueError):
        parse_region_string('fk5\ncircle(150,-30,5") # text={banana}')


def test_missing_frame():
    with pytest.raises(ValueError):
        parse_region_string('circle(150,-30,5")')


def _mask(reg):
    g = geom()
    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    ra, dec = g.pix2world(xx, yy)
    return reg.contains(ra, dec, xx, yy)


def test_circle_area_sky():
    reg = parse_region_string('fk5\ncircle(150,-30,20")')[0]
    m = _mask(reg)
    assert abs(m.sum() - np.pi * 20**2) / (np.pi * 20**2) < 0.02
    assert m[100, 100]  # centre pixel (0-based 100 == CRPIX 101)


def test_ellipse_angle_convention_sky():
    # angle 0: first axis along image +x (West-East line). 30" x 5".
    r0 = parse_region_string('fk5\nellipse(150,-30,30",5",0)')[0]
    m = _mask(r0)
    assert m[100, 125] and m[100, 75] and not m[125, 100]
    # angle 90: first axis along +y (North)
    r90 = parse_region_string('fk5\nellipse(150,-30,30",5",90)')[0]
    m = _mask(r90)
    assert m[125, 100] and m[75, 100] and not m[100, 125]


def test_box_and_polygon_image_frame():
    b = parse_region_string("image\nbox(101,101,21,11,0)")[0]
    m = _mask(b)
    assert m.sum() == 21 * 11
    p = parse_region_string("image\npolygon(91,91,111,91,111,111,91,111)")[0]
    m = _mask(p)
    assert 380 <= m.sum() <= 441 and m[100, 100]


def test_cutout_and_exclusion():
    g = geom()
    regs = parse_region_string('fk5\ncircle(150,-30,10")\n-circle(150,-30,2")')
    cut = region_cutout(regs[0], g, (200, 200), exclusions=[regs[1]])
    assert cut.mask.sum() > 250
    # the excluded centre pixel is not used
    iy, ix = 100 - cut.ys.start, 100 - cut.xs.start
    assert not cut.mask[iy, ix]
    # offsets are relative to the region centre
    assert abs(cut.de[iy, ix]) < 1e-9 and abs(cut.dn[iy, ix]) < 1e-9
