"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/4
@Description:
"""
from enum import Enum


class Level(Enum):
    L1A = 0
    L1B = 1


class Payload(Enum):
    IMU = 0
    SCA = 1
    ACC = 2


class SatID(Enum):
    A = 0
    B = 1
    C = 2
    D = 4


class Mission(Enum):
    GRACE = 0
    GRACE_FO = 1


class IMUID(Enum):
    No_1 = 1
    No_2 = 2
    No_3 = 3
    No_4 = 4


class SCAID(Enum):
    No_1 = 1
    No_2 = 2
    No_3 = 3
