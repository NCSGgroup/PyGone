"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/9/9
@Description:
"""
import Quaternion
import numpy as np
import os
from Setting import SatID
from Quaternion import Quat


class Pointing:
    """
    Pointing variation (roll, pitch, and yaw, called RPY) can be derived from a matrix rotating the K-frame into the
    LOS-frame.

    Reference:
    1. Characteristics and accuracies of the GRACE inter-satellite pointing, Tamara Bandikova et al, 2012.
    """

    def __init__(self, SCA_attitude: np.ndarray):
        """
        :param SCA_attitude: N*5, and the first column denotes the time epoch, and the rest column are quaternions
        """
        self.__SCA = SCA_attitude
        self.__pc = self.__loadPC()
        self.__LOS = None
        pass

    def __loadPC(self):
        '''
        load kbr phase center, which should be retrieved from the VKB1B files. Here, this is simplified by
        a direct assignment.
        :return:
        '''

        PC = {
            'C': np.array([1.4443985, -0.00017, 0.000448]),
            'D': np.array([1.4444575, 0.000054, 0.000230])
        }
        return PC

    def loadLos(self, date: str, dataDir: str, version='HUST01'):
        fileDir = os.path.join(dataDir, '%s/' % date)
        filename = 'IF2LOS_%s_%s.npy' % (date, version)

        self.__LOS = np.load(str(fileDir) + filename)
        return self

    def getRPY(self, sat: SatID):

        SCA_time = self.__SCA[:, 0]
        LOS_time = self.__LOS[:, 0]

        '''find the common time'''
        SCA_index, LOS_index = [], []
        '''find the common time'''
        i, j = 0, 0
        while 1:
            if i == len(SCA_time) or j == len(LOS_time):
                break

            if np.fabs(SCA_time[i] - LOS_time[j]) < 0.0001:
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

        '''Convert SCA into rotation matrix, notice that the quaternions are in the JPL convention'''
        '''So, the matrix is a transpose if using the Hamilton convention like here '''
        q=SCA[:, [1, 2, 3, 0]]
        q = Quaternion.normalize(q)
        RsrfT = Quat(q=q).transform

        PC = self.__pc[sat.name]
        PC = np.array([PC, ] * len(time))[:, :, None]
        e1 = RsrfT @ PC
        e1 = e1[:, :, 0]
        e1 = e1 / np.sqrt(e1[:, 0] ** 2 + e1[:, 1] ** 2 + e1[:, 2] ** 2)[:, None]
        ySRF = RsrfT[:, :, 1]
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
