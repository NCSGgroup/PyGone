"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/21
@Description:
"""
import os

from Setting import Mission, SatID, IMUID, SCAID
from GeoMathKit import GeoMathKit
import numpy as np
import datetime
from GetInstrument import GetInstrument_L1B


class LOS:
    """
    Calculate the Line Of Sight Frame of the satellite.
    Notice: LOS with respect to the inertial frame.

    Reference:
    1. Characteristics and accuracies of the GRACE inter-satellite pointing, Tamara Bandikova et al, 2012.
    2. Analyzing and monitoring GRACE-FO star camera performance in a changing environment, 2020, master thesis
    """

    def __init__(self, mission: Mission = Mission.GRACE_FO, date: str = '2019-08-06'):
        self.mission = mission
        self.date = datetime.datetime.strptime(date, "%Y-%m-%d")
        self._dataDir = None
        self.__L1B = GetInstrument_L1B(mission=mission, date=date)

        pass

    def configDir(self, DataDir: str):
        self._dataDir = os.path.join(DataDir, self.date.strftime("%Y-%m"),
                                     'gracefo_1B_%s_RL04.ascii.noLRI.tgz_files' % (self.date.strftime("%Y-%m-%d")))
        self.__L1B.configDir(DataDir)
        return self

    @DeprecationWarning
    def getLOS_normal(self, DataDir: str, version='HUST01'):
        """
        calculate and save the result
        :param DataDir: Output directory
        :return:
        """

        '''get orbit data in Inertial Frame (IF)'''
        GNI_C = self.__L1B.getGNI(sat=SatID.C)
        GNI_D = self.__L1B.getGNI(sat=SatID.D)

        '''compute the rotation matrix from IF to LOS-frame'''
        R = []
        i, j = 0, 0
        while 1:
            if i == len(GNI_C[0]) or j == len(GNI_D[0]):
                break

            if np.fabs(GNI_C[0][i] - GNI_D[0][j]) < 0.0001:
                '''computation with formula in ref.1'''
                r1 = GNI_C[[1, 2, 3]][:, i]
                r2 = GNI_D[[1, 2, 3]][:, j]
                e1 = (r2 - r1) / np.linalg.norm(r1 - r2)
                cross = np.cross(r1, r2)
                e2 = cross / np.linalg.norm(cross)
                e3 = np.cross(e1, e2)
                R.append(np.array([e1, e2, e3]))
                i += 1
                j += 1
            else:
                if GNI_C[0][i] > GNI_D[0][j] + 0.001:
                    '''computation with formula in ref 2. Sat.D is used'''
                    r = GNI_D[[1, 2, 3]][:, j]
                    v = GNI_D[[4, 5, 6]][:, j]
                    e1 = -r / np.linalg.norm(r)
                    cross = np.cross(r, v)
                    e2 = cross / np.linalg.norm(cross)
                    e3 = np.cross(e1, e2)
                    R.append(np.array([e1, e2, e3]))
                    j += 1
                else:
                    '''computation with formula in ref 2. Sat.C is used'''
                    r = GNI_C[[1, 2, 3]][:, i]
                    v = GNI_C[[4, 5, 6]][:, i]
                    e1 = -r / np.linalg.norm(r)
                    cross = np.cross(r, v)
                    e2 = cross / np.linalg.norm(cross)
                    e3 = np.cross(e1, e2)
                    R.append(np.array([e1, e2, e3]))
                    i += 1

        '''record the result'''
        fileDir = os.path.join(DataDir, '%s/' % self.date.strftime("%Y-%m-%d"))
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'IF2LOS_%s_%s.npy' % (self.date.strftime("%Y-%m-%d"), version)
        np.save(str(fileDir) + filename, np.array(R))

        pass

    def getLOS(self, DataDir: str, version='HUST01'):
        """
        calculate and save the result in an efficient way (matrix)
        :param DataDir: Output directory
        :return:
        """

        '''get orbit data in Inertial Frame (IF)'''
        GNI_C = self.__L1B.getGNI(sat=SatID.C)
        GNI_D = self.__L1B.getGNI(sat=SatID.D)
        if len(GNI_D) == 0:
            self.getLOS_single(sat=SatID.C, DataDir=DataDir, version=version)
            return
        elif len(GNI_C) == 0:
            self.getLOS_single(sat=SatID.D, DataDir=DataDir, version=version)
            return

        '''compute the rotation matrix from IF to LOS-frame'''
        C_index, D_index = [], []
        '''find the common time'''
        i, j = 0, 0
        while 1:
            if i == len(GNI_C[0]) or j == len(GNI_D[0]):
                break

            if np.fabs(GNI_C[0][i] - GNI_D[0][j]) < 0.0001:
                C_index.append(i)
                D_index.append(j)

                i += 1
                j += 1

            elif GNI_C[0][i] - GNI_D[0][j] > 0.0001:
                j += 1
            elif GNI_D[0][j] - GNI_C[0][i] > 0.0001:
                i += 1

        r1 = GNI_C[[1, 2, 3]][:, C_index]
        r2 = GNI_D[[1, 2, 3]][:, D_index]

        e1 = (r2 - r1).T
        e1 = e1 / np.sqrt(e1[:, 0] ** 2 + e1[:, 1] ** 2 + e1[:, 2] ** 2)[:, None]
        e2 = np.cross(r2.T, r1.T)
        e2 = e2 / np.sqrt(e2[:, 0] ** 2 + e2[:, 1] ** 2 + e2[:, 2] ** 2)[:, None]
        e3 = np.cross(e1, e2)

        '''record the result'''
        R = np.zeros((len(e1[:, 0]), 10))
        R[:, 0] = GNI_C[0][C_index]
        R[:, 1:4] = e1
        R[:, 4:7] = e2
        R[:, 7:] = e3
        fileDir = os.path.join(DataDir, '%s/' % self.date.strftime("%Y-%m-%d"))
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'IF2LOS_%s_%s.npy' % (self.date.strftime("%Y-%m-%d"), version)
        np.save(str(fileDir) + filename, R)

        pass

    def getLOS_single(self, sat: SatID, DataDir: str, version='HUST01'):
        """
        calculate and save the result in an efficient way (matrix)
        :param DataDir: Output directory
        :return:
        """

        '''get orbit data in Inertial Frame (IF)'''
        GNI = self.__L1B.getGNI(sat=sat)

        r = GNI[[1, 2, 3]]
        v = GNI[[4, 5, 6]]
        t = GNI[0]

        e3 = -r.T
        e3 = e3 / np.sqrt(e3[:, 0] ** 2 + e3[:, 1] ** 2 + e3[:, 2] ** 2)[:, None]
        e2 = np.cross(r.T, v.T)
        e2 = e2 / np.sqrt(e2[:, 0] ** 2 + e2[:, 1] ** 2 + e2[:, 2] ** 2)[:, None]
        e1 = np.cross(e2, e3)

        '''record the result'''
        R = np.zeros((len(e1[:, 0]), 10))
        R[:, 0] = t
        R[:, 1:4] = e1
        R[:, 4:7] = e2
        R[:, 7:] = e3
        fileDir = os.path.join(DataDir, '%s/' % self.date.strftime("%Y-%m-%d"))
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'IF2LOS_%s_%s.npy' % (self.date.strftime("%Y-%m-%d"), version)
        np.save(str(fileDir) + filename, R)

        pass


def demo():
    los = LOS(mission=Mission.GRACE_FO, date='2019-01-03').configDir('../data/GRACE_FO/RL04/L1B')
    los.getLOS('../result/product/GRACE_FO/RL04/L1B/')
    pass


if __name__ == '__main__':
    demo()
