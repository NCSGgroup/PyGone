"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/13
@Description:
"""

from Setting import SatID, Mission, IMUID
from GeoMathKit import GeoMathKit
import numpy as np
from enum import Enum
from scipy import interpolate
import os


class IMUinterpOption(Enum):
    OneHZ = 1
    TwoHZ = 2
    FourHZ = 3
    EightHZ = 4


class IMU1Bprocess:
    """
    A further process of HUST-IMU L1B data, as a preparation for the following attitude fusion (kalman filter or EKF)
    """

    design_matrix = {}

    def __init__(self, sat: SatID, mission: Mission, date: str, disable: IMUID = None):
        """

        :param sat:
        :param mission:
        :param date: e.g., '2008-02-06'
        """
        self.sat = sat
        self.date = date
        self.mission = mission
        self.__imu_pre = self.__extract_data()
        self.__imu_angular_rate = {}
        self.__interOption = None
        self.__disable = disable
        pass

    def __extract_data(self):
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, self.date)

        filename = 'IMU1B_%s_%s_%s.txt' % (self.date, self.sat.name, 'HUST01')

        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')

        res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 1, 5, 4), unpack=True)

        imu = {}
        imu[IMUID.No_1] = res[0:3, np.fabs(res[-1, :] - 1) < 0.001]
        imu[IMUID.No_2] = res[0:3, np.fabs(res[-1, :] - 2) < 0.001]
        imu[IMUID.No_3] = res[0:3, np.fabs(res[-1, :] - 3) < 0.001]
        imu[IMUID.No_4] = res[0:3, np.fabs(res[-1, :] - 4) < 0.001]

        return imu

    def angular_rate(self, interpOption: IMUinterpOption):

        self.__interOption = interpOption
        disable = self.__disable

        imu = self.__imu_pre
        interval = None
        if interpOption == IMUinterpOption.OneHZ:
            interval = 1
        elif interpOption == IMUinterpOption.TwoHZ:
            interval = 1 / 2
        elif interpOption == IMUinterpOption.FourHZ:
            interval = 1 / 4
        elif interpOption == IMUinterpOption.EightHZ:
            interval = 1 / 8

        time_out = np.arange(0, imu[IMUID.No_1][0][-1] - imu[IMUID.No_1][0][0] + 1, interval)

        for id in IMUID:
            if id == disable:
                continue
            print('Computing %s ...' % id.name)
            time_in = imu[id][0] - imu[IMUID.No_1][0][0] + imu[id][1] * 1e-6

            angular = imu[id][2]
            angular_diff = angular[1:] - angular[0:-1]
            time_diff = time_in[1:] - time_in[0:-1]

            '''delete jump > 4000'''
            jumps = np.where(np.fabs(angular_diff) > 4000)[0]
            if len(jumps) == 0:
                pass
            else:
                angular_diff = np.delete(angular_diff, jumps)
                time_in = np.delete(time_in, jumps)
                time_diff = np.delete(time_diff, jumps)

            '''compute angular rate [degree/sec]'''
            angular_rate = angular_diff / time_diff
            time_in = time_in[0:-1]

            '''interpolation'''
            f = interpolate.interp1d(x=time_in, y=angular_rate, kind='cubic', fill_value='extrapolate')

            '''truncation'''
            # time_out += imu[IMUID.No_1][0][0]
            # time_in +=
            condition1 = time_out >= time_in[0]
            condition2 = time_out <= time_in[-1]
            time_truncate = time_out[condition1 * condition2]

            angular_rate_new = f(time_truncate)
            time_truncate += imu[IMUID.No_1][0][0]

            '''output: [rad/sec]'''
            self.__imu_angular_rate[id] = [time_truncate, np.deg2rad(angular_rate_new, dtype=np.float)]

            pass

        le, re = [], []
        for x in IMUID:
            if x == disable:
                continue
            le.append(self.__imu_angular_rate[x][0][0])
            re.append(self.__imu_angular_rate[x][0][-1])
        leftEnd = max(le)
        rightEnd = min(re)
        # leftEnd = max([self.__imu_angular_rate[x][0][0] for x in IMUID])
        # rightEnd = min([self.__imu_angular_rate[x][0][-1] for x in IMUID])

        for id in IMUID:
            if id == disable:
                continue
            vv = self.__imu_angular_rate[id][1]
            tt = self.__imu_angular_rate[id][0]
            condition1 = tt >= leftEnd
            condition2 = tt <= rightEnd
            time_truncate = tt[condition1 * condition2]
            angular_rate = vv[condition1 * condition2]
            self.__imu_angular_rate[id] = [time_truncate, angular_rate]
            pass

        return self

    @staticmethod
    def ConfigDMforSF2Axis(disable: IMUID = None):
        """
        design matrix for SF to Axis: Sat C and Sat D both
        :return:
        """

        '''for sat C'''
        axis1 = [0.942826211, 0.001810853, 0.33327985]
        axis2 = [-0.472597539, 0.815825089, 0.333288147]
        axis3 = [-0.470092696, -0.817231401, 0.333385204]
        axis4 = [-0.000592072, -0.000382796, -0.999999751]

        axis = {
            IMUID.No_1: axis1,
            IMUID.No_2: axis2,
            IMUID.No_3: axis3,
            IMUID.No_4: axis4
        }

        ax = []
        for id in IMUID:
            if id == disable:
                continue
            ax.append(axis[id])

        DM = np.array(ax)

        # transformation matrix
        TM = np.array([
            [-0.501252164183920, 0.865301254687325, -0.000080854668175],
            [0.865300283426991, 0.501251737450002, 0.001454374272271],
            [0.001299000425484, 0.000659044684221, -0.999998939128437]
        ])

        matC = np.dot(DM, TM.T)

        IMU1Bprocess.design_matrix[SatID.C] = matC

        '''for sat D'''
        axis1 = [0.942687237, 0.000111944, 0.33367763]
        axis2 = [-0.471344745, 0.816437037, 0.333563632]
        axis3 = [-0.471100307, -0.816914065, 0.332740005]
        axis4 = [0.000753399, 0.000461058, -0.99999961]
        # DM = np.array([axis1, axis2, axis3, axis4])
        axis = {
            IMUID.No_1: axis1,
            IMUID.No_2: axis2,
            IMUID.No_3: axis3,
            IMUID.No_4: axis4
        }

        ax = []
        for id in IMUID:
            if id == disable:
                continue
            ax.append(axis[id])

        DM = np.array(ax)

        # transformation matrix
        TM = np.array([
            [-0.501005885615109, 0.865443870184570, 0.000100692470726],
            [0.865443802410115, 0.501005791604386, 0.000470795904316],
            [0.000356999918493, 0.000323015193725, -0.999999884106115]
        ])

        matD = np.dot(DM, TM.T)

        IMU1Bprocess.design_matrix[SatID.D] = matD

        pass

    def CoordinateTransform(self):
        rr = self.__imu_angular_rate
        # '''Four sensor strategy'''
        # fourS = [IMUID.No_1, IMUID.No_2, IMUID.No_3, IMUID.No_4]
        # '''Three sensor strategy'''
        # threeS1 = [IMUID.No_1, IMUID.No_2, IMUID.No_3]
        # threeS2 = [IMUID.No_1, IMUID.No_2, IMUID.No_4]
        # threeS3 = [IMUID.No_1, IMUID.No_3, IMUID.No_4]
        # threeS4 = [IMUID.No_2, IMUID.No_3, IMUID.No_4]
        #
        # stratgey = fourS
        stratgey = []
        for x in IMUID:
            if x == self.__disable:
                continue
            stratgey.append(x)

        newR = [rr[a][1] for a in stratgey]
        b = np.array(newR)
        c = np.linalg.lstsq(self.design_matrix[self.sat], b)[0]

        return {'Time': rr[IMUID.No_1][0], 'Velocity': c.T}

    def record(self, version='HUST01'):
        res = self.CoordinateTransform()
        t = res['Time']
        v = res['Velocity']

        mission = self.mission
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)

        print('Create IMU-L1B angular rate for date: %s' % self.date)

        filename = 'IMU1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__interOption.name, version)

        with open(fileDir + filename, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'IMU Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            for index in range(len(t)):
                vv = v[index]
                file.write('%16.5f %3s   % 20.15e   % 20.15e   % 20.15e \n' %
                           (t[index], self.sat.name, vv[0], vv[1], vv[2]))

        pass


def demo():
    date = '2018-12-02'
    IMU1Bprocess.ConfigDMforSF2Axis()

    imu = IMU1Bprocess(sat=SatID.C, date=date, mission=Mission.GRACE_FO)
    imu.angular_rate(interpOption=IMUinterpOption.FourHZ).record()
    pass


if __name__ == '__main__':
    demo()
