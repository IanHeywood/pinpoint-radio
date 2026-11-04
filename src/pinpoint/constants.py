"""Physical constants used throughout. Single source of truth."""

import math

C = 299_792_458.0  # speed of light, m/s
OMEGA_EARTH = 7.292115e-5  # sidereal rotation rate, rad/s
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
ARCSEC = math.pi / (180.0 * 3600.0)  # radians per arcsec
DEG = math.pi / 180.0
