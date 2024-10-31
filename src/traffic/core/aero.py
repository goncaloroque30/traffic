from typing import Tuple

import numpy as np

from .types import array

p0 = 101325.0  # Pa     Sea level pressure ISA
T0 = 288.15  # K   Sea level temperature ISA
rho0 = 1.225  # kg/m3  Sea level density ISA

R = 287.05287  # Used in wikipedia table: checked with 11000 m
beta = -0.0065  # [K/m] ISA temp gradient below tropopause
Tstrat = 216.65  # K Stratosphere temperature (until alt=22km)


def vatmos(
    h: array, Tb: float | array = T0, pb: float | array = p0
) -> Tuple[array, array, array]:  # h [m], Tb [K], pb [Pa]
    """International Standard Atmosphere calculator

    :param h:  altitude in meters 0.0 < h < 84852.
        (will be clipped when outside range, integer input allowed)

    :param Tb: the sea level temperature in Kelvin. By default, the ISA sea
    level temperature.

    :param pb: the sea level pressure in Pa. By default, the ISA sea level
    pressure.

    :return:
        - the pressure (in Pa)
        - the air density :math:`\\rho` (in kg/m3)
        - the temperature (in K)
    """
    # Base Density
    rhob = pb / (R * Tb)

    # Temp
    T = vtemp(h, Tb)

    # Density
    rhotrop = rhob * (T / Tb) ** 4.256848030018761  # = -(g0*M/((R*)*beta))-1
    dhstrat = np.maximum(0.0, h - 11000.0)
    rho = rhotrop * np.exp(-dhstrat / 6341.552161)  # = *g0/(287.05*216.65))

    # Pressure
    p = rho * R * T

    return p, rho, T


def vtemp(h: array, Tb: float | array = T0) -> array:  # h [m], Tb [K]
    """Temperature only version of ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)
    :param Tb: the base temperature in Kelvin

    :return: the temperature (in K)

    """
    T = np.maximum(Tb + beta * h, Tstrat)
    return T


def vtempbase(h: array, t: array) -> array:
    return T0 + t - vtemp(h)


def vtas2casw(tas: array, p: array, rho: array) -> array:
    qdyn = p * ((1.0 + rho * tas * tas / (7.0 * p)) ** 3.5 - 1.0)
    cas = np.sqrt(7.0 * p0 / rho0 * ((qdyn / p0 + 1.0) ** (2.0 / 7.0) - 1.0))

    # cope with negative speed
    cas = np.where(tas < 0, -1 * cas, cas)
    return cas  # type: ignore
