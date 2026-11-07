"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time: 2021/9/9
@Description:
"""
from quaternion import Quaternion
import numpy as np
import os
from pysrc.core.settings import SatID
from quaternion.Quaternion import Quat


class Pointing:
    """
    Pointing variation (roll, pitch, and yaw, called RPY) can be derived from a
    matrix rotating the K-frame into the LOS-frame.

    Reference:
    1. Characteristics and accuracies of the GRACE inter-satellite pointing,
       Tamara Bandikova et al., 2012.
    """

    def __init__(self, SCA_attitude: np.ndarray, sat: SatID):
        """
        :param SCA_attitude: N x 5 array; the first column is time and the
            remaining columns are quaternions.
        """
        self.__SCA = SCA_attitude
        self.__pc = self.__loadPC()
        self.__LOS = None
        self.__sat = sat

    def __loadPC(self):
        """Load KBR phase centers for the supported spacecraft."""
        return {
            "C": np.array([1.4443985, -0.00017, 0.000448]),
            "D": np.array([1.4444575, 0.000054, 0.000230]),
        }

    def loadLos(self, date: str, dataDir: str, version="HUST01"):
        file_dir = os.path.join(dataDir, date)
        file_name = "IF2LOS_%s_%s_%s.npy" % (date, version, self.__sat.name)
        self.__LOS = np.load(os.path.join(file_dir, file_name))
        return self

    def getRPY(self, sat: SatID):
        SCA_time = self.__SCA[:, 0]
        LOS_time = self.__LOS[:, 0]

        SCA_index, LOS_index = [], []
        i, j = 0, 0
        while True:
            if i == len(SCA_time) or j == len(LOS_time):
                break

            if np.abs(SCA_time[i] - LOS_time[j]) < 0.0001:
                SCA_index.append(i)
                LOS_index.append(j)
                i += 1
                j += 1
            elif SCA_time[i] - LOS_time[j] > 0.0001:
                j += 1
            elif LOS_time[j] - SCA_time[i] > 0.0001:
                i += 1

        SCA = self.__SCA[SCA_index][:, [1, 2, 3, 4]]
        LOS = self.__LOS[LOS_index][:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]

        time = LOS_time[LOS_index]
        R_IN2LOS = LOS.reshape((len(time), 3, 3))

        # Input quaternions use the JPL convention; transpose the Hamilton
        # convention rotation matrix returned by the quaternion helper.
        q = SCA[:, [1, 2, 3, 0]]
        q = Quaternion.normalize(q)
        R_SRF_T = Quat(q=q).transform

        PC = self.__pc[sat.name]
        PC = np.array([PC] * len(time))[:, :, None]
        e1 = R_SRF_T @ PC
        e1 = e1[:, :, 0]
        e1 = e1 / np.sqrt(e1[:, 0] ** 2 + e1[:, 1] ** 2 + e1[:, 2] ** 2)[:, None]
        ySRF = R_SRF_T[:, :, 1]
        e3 = np.cross(e1, ySRF)
        e2 = np.cross(e3, e1)

        R_IN2KF_T = np.zeros((len(time), 3, 3))
        R_IN2KF_T[:, :, 0] = e1
        R_IN2KF_T[:, :, 1] = e2
        R_IN2KF_T[:, :, 2] = e3

        R_KF2LOS = R_IN2LOS @ R_IN2KF_T

        roll = -np.arctan(R_KF2LOS[:, 1, 2] / R_KF2LOS[:, 2, 2])
        pitch = np.arcsin(R_KF2LOS[:, 0, 2])
        yaw = -np.arctan(R_KF2LOS[:, 0, 1] / R_KF2LOS[:, 0, 0])

        return time, roll, pitch, yaw
