"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/5
@Description:
"""
import os

from Setting import Mission, SatID, IMUID, SCAID
from GeoMathKit import GeoMathKit
import numpy as np
import datetime


class GetInstrument_L1A:
    """
    This only specifies to GRACE-FO.
    """

    def __init__(self, mission: Mission = Mission.GRACE_FO, date: str = '2019-08-06'):
        self.mission = mission
        self.date = datetime.datetime.strptime(date, "%Y-%m-%d")
        self._dataDir = None
        pass

    def configDir(self, DataDir: str):
        self._dataDir = os.path.join(DataDir, self.date.strftime("%Y-%m"),
                                     'gracefo_1A_%s_RL04.ascii.noLRI.tgz_files' % (self.date.strftime("%Y-%m-%d")))
        return self

    def getIMU(self, sat: SatID):
        file = 'IMU1A_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 1, 5, 4), unpack=True)

        imu = {}
        imu[IMUID.No_1.name] = res[0:3, np.fabs(res[-1, :] - 1) < 0.001]
        imu[IMUID.No_2.name] = res[0:3, np.fabs(res[-1, :] - 2) < 0.001]
        imu[IMUID.No_3.name] = res[0:3, np.fabs(res[-1, :] - 3) < 0.001]
        imu[IMUID.No_4.name] = res[0:3, np.fabs(res[-1, :] - 4) < 0.001]
        return imu

    def getSCA(self, sat: SatID):
        file = 'SCA1A_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 1, 5, 6, 7, 8, 11, 14, 15, 3), unpack=True)
        if len(res) == 0:
            '''
            It is an empty file
            '''
            return None
        sca = {}
        sca[SCAID.No_1.name] = res[0:-1, np.fabs(res[-1, :] - 1) < 0.001]
        sca[SCAID.No_2.name] = res[0:-1, np.fabs(res[-1, :] - 2) < 0.001]
        sca[SCAID.No_3.name] = res[0:-1, np.fabs(res[-1, :] - 3) < 0.001]

        return sca


class GetInstrument_L1B(GetInstrument_L1A):

    def __init__(self, mission: Mission = Mission.GRACE_FO, date: str = '2019-08-06'):
        super().__init__(mission, date)

    def configDir(self, DataDir: str):
        self._dataDir = os.path.join(DataDir, self.date.strftime("%Y-%m"),
                                     'gracefo_1B_%s_RL04.ascii.noLRI.tgz_files' % (self.date.strftime("%Y-%m-%d")))
        return self

    def getTIM(self, sat: SatID):
        """
        NOTICE! For a complete interpolation later on, the IMU on the day before and after
        the given date are needed as well.
        :param sat:
        :return:
        """

        file = 'TIM1B_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res0 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3, 4, 7), unpack=True)

        '''the day before'''
        i = 1
        while 1:
            date = self.date + datetime.timedelta(days=-i)
            dataDir = self._dataDir.replace(self.date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
            dataDir = dataDir.replace(self.date.strftime("%Y-%m"), date.strftime("%Y-%m"))
            file = 'TIM1B_%s_%s_04.txt' % (date.strftime("%Y-%m-%d"), sat.name)
            file = os.path.join(dataDir, file)
            skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
            res1 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3, 4, 7), unpack=True)
            if res1[3, -2] < 1000:
                res = np.hstack((res1[:, -2:], res0))
                break
            else:
                quality = res1[3, :]
                if np.all(quality == 1000):
                    i += 1
                    continue
                else:
                    index = np.where(quality < 1000)[0][-1]
                    res = np.hstack((res1[:, index][:, None], res0))
                    break

        '''the day after'''
        i = 1
        while 1:
            date = self.date + datetime.timedelta(days=i)
            dataDir = self._dataDir.replace(self.date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
            dataDir = dataDir.replace(self.date.strftime("%Y-%m"), date.strftime("%Y-%m"))
            file = 'TIM1B_%s_%s_04.txt' % (date.strftime("%Y-%m-%d"), sat.name)
            file = os.path.join(dataDir, file)
            skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
            res2 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3, 4, 7), unpack=True)
            if res2[3, 1] < 1000:
                res = np.hstack((res, res2[:, :2]))
                break
            else:
                quality = res2[3, :]
                if np.all(quality == 1000):
                    i += 1
                    continue
                else:
                    index = np.where(quality < 1000)[0][0] + 1
                    res = np.hstack((res, res2[:, :index]))
                    break

        return res

    def getCLK(self, sat: SatID):
        file = 'CLK1B_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res1 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3), unpack=True)
        res2 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(-1), unpack=True, dtype=str)
        flag = np.zeros(len(res2))
        '''can't extrapolate before the point '''
        flag[res2 == '00000010'] = 1
        '''can't extrapolate after the point'''
        flag[res2 == '00000001'] = 2
        res = np.vstack((res1, flag[None, :]))

        '''the day before'''
        i = 1
        while 1:
            date = self.date + datetime.timedelta(days=-i)
            dataDir = self._dataDir.replace(self.date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
            dataDir = dataDir.replace(self.date.strftime("%Y-%m"), date.strftime("%Y-%m"))
            file = 'CLK1B_%s_%s_04.txt' % (date.strftime("%Y-%m-%d"), sat.name)
            file = os.path.join(dataDir, file)
            skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
            res1 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3), unpack=True)
            res2 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(-1), unpack=True, dtype=str)
            flag = np.zeros(len(res2))
            '''can't extrapolate before the point '''
            flag[res2 == '00000010'] = 1
            '''can't extrapolate after the point'''
            flag[res2 == '00000001'] = 2
            res_pre = np.vstack((res1, flag[None, :]))
            if len(res_pre[0, :]) == 0:
                i += 1
                continue
            else:
                break

        '''the day after'''
        i = 1
        while 1:
            date = self.date + datetime.timedelta(days=i)
            dataDir = self._dataDir.replace(self.date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
            dataDir = dataDir.replace(self.date.strftime("%Y-%m"), date.strftime("%Y-%m"))
            file = 'CLK1B_%s_%s_04.txt' % (date.strftime("%Y-%m-%d"), sat.name)
            file = os.path.join(dataDir, file)
            skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
            res1 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3), unpack=True)
            res2 = np.loadtxt(file, comments='#', skiprows=skip, usecols=(-1), unpack=True, dtype=str)
            flag = np.zeros(len(res2))
            '''can't extrapolate before the point '''
            flag[res2 == '00000010'] = 1
            '''can't extrapolate after the point'''
            flag[res2 == '00000001'] = 2
            res_aft = np.vstack((res1, flag[None, :]))
            if len(res_aft[0, :]) == 0:
                i += 1
                continue
            else:
                break

        if len(res[0,:]) == 0:
            res = np.hstack((res_pre, res_aft))
        else:
            res = np.hstack((res_pre, res))
            res = np.hstack((res, res_aft))

        return res

    def getIMU(self, sat: SatID):
        file = 'IMU1B_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 1, 5, 4), unpack=True)

        imu = {}
        imu[IMUID.No_1.name] = res[0:3, np.fabs(res[-1, :] - 1) < 0.001]
        imu[IMUID.No_2.name] = res[0:3, np.fabs(res[-1, :] - 2) < 0.001]
        imu[IMUID.No_3.name] = res[0:3, np.fabs(res[-1, :] - 3) < 0.001]
        imu[IMUID.No_4.name] = res[0:3, np.fabs(res[-1, :] - 4) < 0.001]

        return imu

    def getSCA(self, sat: SatID):
        file = 'SCA1B_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3, 4, 5, 6), unpack=True)
        return res

    def getGNI(self, sat: SatID):
        file = 'GNI1B_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3, 4, 5, 9, 10, 11), unpack=True)
        return res

    def getACC(self, sat: SatID):
        file = 'ACT1B_%s_%s_04.txt' % (self.date.strftime("%Y-%m-%d"), sat.name)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'End of YAML header')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 2, 3, 4), unpack=True)
        return res


def demo():
    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date='2018-12-01').configDir('../data/GRACE_FO/RL04/L1A')
    # L1A.getIMU(SatID.C)
    # L1A.getSCA(SatID.C)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date='2019-01-01').configDir('../data/GRACE_FO/RL04/L1B')

    L1B.getTIM(SatID.C)
    # L1B.getCLK(SatID.C)
    # imu = L1B.getIMU(SatID.D)
    # L1B.getSCA(SatID.C)
    pass


if __name__ == '__main__':
    demo()
