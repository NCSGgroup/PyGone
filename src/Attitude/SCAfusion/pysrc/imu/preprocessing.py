from pysrc.core.settings import SatID, Mission, IMUID
from pysrc.core.geometry import GeoMathKit
import numpy as np
from enum import Enum
from scipy import interpolate
import os

class IMUinterpOption(Enum):
    OneHZ=1
    TwoHz=2
    FourHz=3
    EightHz=4


class IMU1Bprocess:
    """
    A further process of HUST-IMU 1B data, as a preparation for the following attitude fusion (kalman)
    """

    # def __init__(self, sat: SatID, mission: Mission, data: str, disable: IMUID=None):
    #
    #
    # pass