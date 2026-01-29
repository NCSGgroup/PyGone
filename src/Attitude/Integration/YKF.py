"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/12/17
@Description:
"""

from Setting import SCAID, SatID, Mission
from IMU1Bprocess import IMUinterpOption
from SCA1Ato1B import SCAinterOption
from GeoMathKit import GeoMathKit
import numpy as np
import os
import datetime
from tqdm import trange


class YKF_NoCov:
    """
    This is a new kalman filter, proposed by Yang Fan, which no longer requires the combination of SC beforehand.

    The traditional method treats the observations in the spacecraft's frame, so that a pre-combination of SC as well
    as the frame rotation from science frame into spacecraft's frame is required. Now we show that, the pre-combination
    might be not an optimal option, since it combines the SCs in a fixed weight that might be against the reality. To
    this end, we propose to involve the SC measurement directly in the kalman filter, and meanwhile we slightly modify
    the kalman filter to make it suitable for observations in science frame. In addition, such a modified Kalman filter
    also allows for introducing the a-priori of SC noise (boresight noise is 10 times bigger than the direction
    perpendicular to the boresight).
    """

    def __init__(self, date, sat: SatID, dir: str):
        self.date = date
        self.dir = dir
        self.sat = sat
        self.__IMUoption, self.__SCAoption = None, None
        self.IMU = None
        self.SCA = None
        self.SCA_combination = None
        self.__timelist = None
        pass

    def configInput(self, IMUoption: IMUinterpOption, SCAoption: SCAinterOption):
        self.__IMUoption = IMUoption
        self.__SCAoption = SCAoption
        return self

    def load(self):
        version = 'HUST01'
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (Mission.GRACE_FO.name, self.date)
        # fileDir = self.dir +'%s/' % self.date

        print('\nKalman filter loading')
        print('Loading IMU input ...')
        filename = 'IMU1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__IMUoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        IMU = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 2, 3, 4), unpack=False)

        print('\nLoading SCA input ...')
        filename = 'SCA1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__SCAoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 4, 5, 6, 7, 8, 9, 10, 3), unpack=True)
        SCA = {}
        SCA[SCAID.No_1] = res[0:-1, np.fabs(res[-1, :] - 1) < 0.001]
        SCA[SCAID.No_2] = res[0:-1, np.fabs(res[-1, :] - 2) < 0.001]
        SCA[SCAID.No_3] = res[0:-1, np.fabs(res[-1, :] - 3) < 0.001]

        self.IMU = IMU
        self.SCA = SCA
        t1 = SCA[SCAID.No_1][0]
        t2 = SCA[SCAID.No_2][0]
        t3 = SCA[SCAID.No_3][0]
        self.__timelist = [t1, t2, t3]

        '''read combination result'''
        filename = 'SCA1B_%s_%s_%s_combined_%s.txt' % (self.date, self.sat.name, self.__SCAoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        SCA_combination = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 3, 4, 5, 6), unpack=True)

        self.SCA_combination = SCA_combination
        return self

    def filter(self, iniP=None, iniQ=None, iniB=None, iniT=None, onlyPredict=False, isFirst=False,
               version='HUST01'):
        """
        Start filtering
        :param onlyPredict: True when there are no SCA measurements
        :param iniQ: initial value of X -- state vector-quaternion
        :param iniB: initial value of X -- state vector-bias
        :param iniP: initial value of P -- covariance of the state vector
        :param iniT: initial time epoch
        :param version: no actual meaning but a flag.
        :return:
        """
        fileDir = self.dir + '%s/' % self.date
        # fileDir = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, self.date)

        today = datetime.datetime.strptime(self.date, "%Y-%m-%d")
        previous_day = today + datetime.timedelta(days=-1)
        fileDir_previous = self.dir + '%s/' % previous_day.strftime("%Y-%m-%d")
        # fileDir_previous = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, previous_day.strftime(
        # "%Y-%m-%d"))

        if iniP is None:
            iniP = np.load(fileDir_previous + 'P_%s.npy' % self.sat.name)

        if iniB is None:
            iniB = np.load(fileDir_previous + 'Bias_%s.npy' % self.sat.name)

        IMU, SCA = self.IMU, self.SCA
        Ts = IMU[:, 0]

        SCA_combination = self.SCA_combination

        iniX = None
        if iniT is None:
            i = 0
            while 1:

                iniT = SCA_combination[0, i]
                iniX = np.array(list(SCA_combination[1:5, i]) + list(iniB))

                if iniT >= Ts[0]:
                    break
                else:
                    i += 1
        else:
            ii = np.fabs(SCA_combination[0] - iniT) < 0.01
            iniX = np.array(list(SCA_combination[1:5, ii].flatten()) + list(iniB))
            pass

        '''JPL convention'''
        X_update, P_update = iniX[[1, 2, 3, 0, 4, 5, 6]], iniP

        if iniT < Ts[0]:
            raise Exception('Kalman input error: no sufficient data')

        startT = np.where(np.fabs(Ts - iniT) < 0.0001)[0][0]
        if startT is None:
            raise Exception('Kalman input error: time mismatch')

        lastTime = {'index': {
            SCAID.No_1: 0,
            SCAID.No_2: 0,
            SCAID.No_3: 0
        }}

        '''record the value in Hamilton convention'''
        fusion = [X_update[[3, 0, 1, 2]]]
        fusionTime = [Ts[startT]]
        bias = []

        print('\nKalman fusion starts ...')
        for i in trange(startT + 1, len(Ts)):
            ept = Ts[i]
            if ept > self.__timelist[0][-1] and ept > self.__timelist[1][-1] and ept > self.__timelist[2][-1]:
                print('\nKalman filter finished')
                break
            '''get angular velocity from IMU, [radian/sec]'''
            w_meas = IMU[i - 1, 1:]

            '''remove the bias from the angular velocity'''
            w = w_meas - X_update[4:]

            '''get attitude at current step in JPL convention'''
            q = X_update[0:4]

            '''get the time increment'''
            deltaT = ept - Ts[i - 1]

            '''make a prediction'''
            DeltaTheta = w * deltaT
            AbsDeltaTheta = np.linalg.norm(DeltaTheta)
            OmegaDeltaTheta = np.array([
                [0, DeltaTheta[2], -DeltaTheta[1], DeltaTheta[0]],
                [-DeltaTheta[2], 0, DeltaTheta[0], DeltaTheta[1]],
                [DeltaTheta[1], -DeltaTheta[0], 0, DeltaTheta[2]],
                [-DeltaTheta[0], -DeltaTheta[1], -DeltaTheta[2], 0]
            ])

            MdeltaTheta = np.eye(4) * np.cos(AbsDeltaTheta / 2.0) + \
                          OmegaDeltaTheta * np.sin(AbsDeltaTheta / 2.0) / AbsDeltaTheta
            q_predict = MdeltaTheta @ q
            b_predict = X_update[4:].copy()
            X_predict = np.array([q_predict[0], q_predict[1], q_predict[2], q_predict[3],
                                  b_predict[0], b_predict[1], b_predict[2]])

            '''get transition matrix phi'''
            phi = np.eye(7)
            phi[0:4, 0:4] = MdeltaTheta.copy()
            boxI = np.array([
                [q[3], -q[2], q[1]],
                [q[2], q[3], -q[0]],
                [-q[1], q[0], q[3]],
                [-q[0], -q[1], -q[2]]
            ])
            phi[0:4, 4:] = MdeltaTheta.copy() @ boxI * (-1.0 / 2)

            '''get Gt '''
            Gt = np.zeros((7, 6))
            Gt[0:4, 0:3] = boxI * (-1.0 / 2)
            Gt[4:, 3:] = np.eye(3)

            '''covariance prediction'''
            if isFirst:
                sigma1 = (1e-11) ** 2  # Gaussian noise
                sigma2 = (1e-11) ** 2  # random walk noise
            else:
                sigma1 = (5e-13) ** 2  # Gaussian noise
                sigma2 = (5e-13) ** 2  # random walk noise
                # sigma1 = (1e-11) ** 2  # Gaussian noise
                # sigma2 = (1e-11) ** 2  # random walk noise

            Qt = np.diagflat([sigma1, sigma1, sigma1, sigma2, sigma2, sigma2]) * deltaT ** 2
            # Qt = np.eye(6) * deltaT ** 2 * sigma
            # Qt[3:, 3:] = np.zeros((3, 3))
            P_predict = phi @ P_update @ phi.T + phi @ Gt @ Qt @ Gt.T @ phi.T

            '''judge whether there is a SCA observation'''
            thisTime = self.__isSCAexist(ept, lastTime=lastTime)
            thisTime = self.__isSCAvalid(thisTime=thisTime)

            SCAexist = thisTime['isExist']

            '''update the prediction with new measurements'''
            Y = self.__getY(t=ept, SCA_exist=SCAexist, thisTime=thisTime, X=X_predict)

            # if i >= len(Ts)-2:
            #     SCAexist = False
            H = self.__getH(SCAexist)
            R = self.__getR(t=ept, SCA_exist=SCAexist, X=X_predict, thisTime=thisTime)

            if len(Y) == 0 or onlyPredict:
                X_update, P_update = X_predict, P_predict
            else:
                X_update, P_update = self.__update(X=X_predict, Y=Y, H=H, P=P_predict, R=R)

            '''Normalization'''
            X_update[0:4] = X_update[0:4] / np.linalg.norm(X_update[0:4])

            lastTime = thisTime
            fusion.append(X_update[[3, 0, 1, 2]].copy())
            fusionTime.append(ept)
            bias.append(X_update[4:])
        # print(X_update[4:])
        '''record the result'''
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'SCA1B_%s_%s_Fusion_%s.txt' % (self.date, self.sat.name, version)

        with open(fileDir + filename, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            for index in range(len(fusion)):
                vv = fusion[index]
                vt = fusionTime[index]
                file.write('%16.5f %3s   % 20.16f   % 20.16f   % 20.16f  % 20.16f\n' %
                           (vt, self.sat.name, vv[0], vv[1], vv[2], vv[3]))

        '''record the P matrix and bias, as the initial value for the day after this day'''
        np.save(fileDir + 'P_%s.npy' % self.sat.name, np.array(P_update))
        np.save(fileDir + 'Bias_%s.npy' % self.sat.name, np.array(X_update[4:]))
        np.save(fileDir + 'BiasAll_%s.npy' % self.sat.name, np.array(bias))

        pass

    def __isSCAexist(self, ept, lastTime: dict):
        """
        judge if there is a SCA measurement at given epoch, in an efficient way
        :return:
        """

        n1 = lastTime['index'][SCAID.No_1]
        n2 = lastTime['index'][SCAID.No_2]
        n3 = lastTime['index'][SCAID.No_3]

        [t1, t2, t3] = self.__timelist

        thisTime = {'isExist': {},
                    'index': {}}

        i, j, k = 0, 0, 0
        while 1:
            if np.fabs(t1[n1 + i] - ept) < 0.001:
                thisTime['isExist'][SCAID.No_1] = True
                thisTime['index'][SCAID.No_1] = n1 + i
                break
            elif t1[n1 + i] > ept + 0.001:
                thisTime['isExist'][SCAID.No_1] = False
                thisTime['index'][SCAID.No_1] = n1 + i
                break
            else:
                i += 1

        while 1:
            if np.fabs(t2[n2 + j] - ept) < 0.001:
                thisTime['isExist'][SCAID.No_2] = True
                thisTime['index'][SCAID.No_2] = n2 + j
                break
            elif t2[n2 + j] > ept + 0.001:
                thisTime['isExist'][SCAID.No_2] = False
                thisTime['index'][SCAID.No_2] = n2 + j
                break
            else:
                j += 1

        while 1:
            if np.fabs(t3[n3 + k] - ept) < 0.001:
                thisTime['isExist'][SCAID.No_3] = True
                thisTime['index'][SCAID.No_3] = n3 + k
                break
            elif t3[n3 + k] > ept + 0.001:
                thisTime['isExist'][SCAID.No_3] = False
                thisTime['index'][SCAID.No_3] = n3 + k
                break
            else:
                k += 1

        return thisTime

    def __isSCAvalid(self, thisTime):
        SCA = self.SCA

        for id in SCAID:
            index = thisTime['index'][id]
            if not thisTime['isExist'][id]:
                continue
            if SCA[id][5, index] > 6:
                # todo: > or >= ??, please check
                thisTime['isExist'][id] = False
        return thisTime

    def __update(self, X, Y, H, P, R):
        K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
        return X + np.dot(K, Y - np.dot(H, X)), np.dot(np.identity(7) - np.dot(K, H), P)

    def __getH(self, SCA_exist: dict):
        H1 = H2 = H3 = np.eye(N=4, M=7, k=0)

        Hmat = np.zeros(shape=(0, 7))
        if SCA_exist[SCAID.No_1]:
            Hmat = np.vstack((Hmat, H1))

        if SCA_exist[SCAID.No_2]:
            Hmat = np.vstack((Hmat, H2))

        if SCA_exist[SCAID.No_3]:
            Hmat = np.vstack((Hmat, H3))

        return Hmat

    def __getY(self, t, SCA_exist: dict, thisTime: dict, X):
        """
        get y in JPL convention
        :param t:
        :param SCA_exist:
        :param thisTime:
        :param X:
        :return:
        """
        SCA = self.SCA

        Y = []
        for id in SCAID:
            if SCA_exist[id]:
                '''flip the sign'''
                y = SCA[id][1:5, thisTime['index'][id]][[1, 2, 3, 0]]
                if np.dot(X[0:3], y[0:3]) < 0:
                    Y.append(-y)
                else:
                    Y.append(y)

        Y = np.array(Y).flatten()

        return Y

    def __getR(self, t, SCA_exist: dict, X, thisTime: dict):
        """
        constant R
        :param t:
        :return:
        """

        R = []

        ratio = []
        SCA = self.SCA
        for id in SCAID:
            rr = 1
            # rr = 0.5 * (SCA[id][5, thisTime['index'][id]] + 1)
            #
            # '''flip the sign'''
            # y = SCA[id][1:5, thisTime['index'][id]]
            # theta = np.dot(X[0:3], y[1:4])
            # if theta < 0:
            #     m = 180
            # else:
            #     m = 0
            # diff_deg = np.rad2deg(np.arccos(theta))
            # if np.fabs(diff_deg - m) < 5:
            #     ratio.append(1)
            # else:
            #     ratio.append(1e3)

            ratio.append(rr)

        q1 = q2 = q3 = q4 = 1e-10

        if SCA_exist[SCAID.No_1]:
            R += [q1 * ratio[0], q2 * ratio[0], q3 * ratio[0], q4 * ratio[0]]

        if SCA_exist[SCAID.No_2]:
            R += [q1 * ratio[1], q2 * ratio[1], q3 * ratio[1], q4 * ratio[1]]

        if SCA_exist[SCAID.No_3]:
            R += [q1 * ratio[2], q2 * ratio[2], q3 * ratio[2], q4 * ratio[2]]

        return np.diagflat(R)


class YKF_Cov:
    """
    This is a new kalman filter, proposed by Yang Fan, which no longer requires the combination of SC beforehand.

    The traditional method treats the observations in the spacecraft's frame, so that a pre-combination of SC as well
    as the frame rotation from science frame into spacecraft's frame is required. Now we show that, the pre-combination
    might be not an optimal option, since it combines the SCs in a fixed weight that might be against the reality. To
    this end, we propose to involve the SC measurement directly in the kalman filter, and meanwhile we slightly modify
    the kalman filter to make it suitable for observations in science frame. In addition, such a modified Kalman filter
    also allows for introducing the a-priori of SC noise (boresight noise is 10 times bigger than the direction
    perpendicular to the boresight).
    """

    def __init__(self, date, sat: SatID, dir: str):
        self.date = date
        self.dir = dir
        self.sat = sat
        self.__IMUoption, self.__SCAoption = None, None
        self.IMU = None
        self.SCA = None
        self.SCA_combination = None
        self.__timelist = None
        self.QSA = None
        pass

    def configInput(self, IMUoption: IMUinterpOption, SCAoption: SCAinterOption):
        self.__IMUoption = IMUoption
        self.__SCAoption = SCAoption
        return self

    def load(self):
        version = 'HUST01'
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (Mission.GRACE_FO.name, self.date)
        # fileDir = self.dir +'%s/' % self.date

        print('\nKalman filter loading')
        print('Loading IMU input ...')
        filename = 'IMU1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__IMUoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        IMU = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 2, 3, 4), unpack=False)

        print('\nLoading SCA input ...')
        filename = 'SCA1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__SCAoption.name, '02')
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 4, 5, 6, 7, 8, 9, 10, 3), unpack=True)
        SCA = {}
        SCA[SCAID.No_1] = res[0:-1, np.fabs(res[-1, :] - 1) < 0.001]
        SCA[SCAID.No_2] = res[0:-1, np.fabs(res[-1, :] - 2) < 0.001]
        SCA[SCAID.No_3] = res[0:-1, np.fabs(res[-1, :] - 3) < 0.001]

        self.IMU = IMU
        self.SCA = SCA
        t1 = SCA[SCAID.No_1][0]
        t2 = SCA[SCAID.No_2][0]
        t3 = SCA[SCAID.No_3][0]
        self.__timelist = [t1, t2, t3]

        '''read combination result'''
        filename = 'SCA1B_%s_%s_%s_combined_%s.txt' % (self.date, self.sat.name, self.__SCAoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        SCA_combination = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 3, 4, 5, 6), unpack=True)

        self.SCA_combination = SCA_combination
        return self

    def filter(self, iniP=None, iniQ=None, iniB=None, iniT=None, onlyPredict=False, isFirst=False,
               version='HUST01'):
        """
        Start filtering
        :param onlyPredict: True when there are no SCA measurements
        :param iniQ: initial value of X -- state vector-quaternion
        :param iniB: initial value of X -- state vector-bias
        :param iniP: initial value of P -- covariance of the state vector
        :param iniT: initial time epoch
        :param version: no actual meaning but a flag.
        :return:
        """
        fileDir = self.dir + '%s/' % self.date
        # fileDir = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, self.date)

        today = datetime.datetime.strptime(self.date, "%Y-%m-%d")
        previous_day = today + datetime.timedelta(days=-1)
        fileDir_previous = self.dir + '%s/' % previous_day.strftime("%Y-%m-%d")
        # fileDir_previous = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, previous_day.strftime(
        # "%Y-%m-%d"))

        if iniP is None:
            iniP = np.load(fileDir_previous + 'P_%s.npy' % self.sat.name)

        if iniB is None:
            iniB = np.load(fileDir_previous + 'Bias_%s.npy' % self.sat.name)

        IMU, SCA = self.IMU, self.SCA
        Ts = IMU[:, 0]

        SCA_combination = self.SCA_combination

        iniX = None
        if iniT is None:
            i = 0
            while 1:

                iniT = SCA_combination[0, i]
                iniX = np.array(list(SCA_combination[1:5, i]) + list(iniB))

                if iniT >= Ts[0]:
                    break
                else:
                    i += 1
        else:
            ii = np.fabs(SCA_combination[0] - iniT) < 0.01
            iniX = np.array(list(SCA_combination[1:5, ii].flatten()) + list(iniB))
            pass

        '''JPL convention'''
        X_update, P_update = iniX[[1, 2, 3, 0, 4, 5, 6]], iniP

        if iniT < Ts[0]:
            raise Exception('Kalman input error: no sufficient data')

        startT = np.where(np.fabs(Ts - iniT) < 0.0001)[0][0]
        if startT is None:
            raise Exception('Kalman input error: time mismatch')

        lastTime = {'index': {
            SCAID.No_1: 0,
            SCAID.No_2: 0,
            SCAID.No_3: 0
        }}

        '''record the value in Hamilton convention'''
        fusion = [X_update[[3, 0, 1, 2]]]
        fusionTime = [Ts[startT]]
        bias = []

        print('\nKalman fusion starts ...')
        for i in trange(startT + 1, len(Ts)):
            ept = Ts[i]
            if ept > self.__timelist[0][-1] and ept > self.__timelist[1][-1] and ept > self.__timelist[2][-1]:
                print('\nKalman filter finished')
                break
            '''get angular velocity from IMU, [radian/sec]'''
            w_meas = IMU[i - 1, 1:]

            '''remove the bias from the angular velocity'''
            w = w_meas - X_update[4:]

            '''get attitude at current step in JPL convention'''
            q = X_update[0:4]

            '''get the time increment'''
            deltaT = ept - Ts[i - 1]

            '''make a prediction'''
            DeltaTheta = w * deltaT
            AbsDeltaTheta = np.linalg.norm(DeltaTheta)
            OmegaDeltaTheta = np.array([
                [0, DeltaTheta[2], -DeltaTheta[1], DeltaTheta[0]],
                [-DeltaTheta[2], 0, DeltaTheta[0], DeltaTheta[1]],
                [DeltaTheta[1], -DeltaTheta[0], 0, DeltaTheta[2]],
                [-DeltaTheta[0], -DeltaTheta[1], -DeltaTheta[2], 0]
            ])

            MdeltaTheta = np.eye(4) * np.cos(AbsDeltaTheta / 2.0) + \
                          OmegaDeltaTheta * np.sin(AbsDeltaTheta / 2.0) / AbsDeltaTheta
            q_predict = MdeltaTheta @ q
            b_predict = X_update[4:].copy()
            X_predict = np.array([q_predict[0], q_predict[1], q_predict[2], q_predict[3],
                                  b_predict[0], b_predict[1], b_predict[2]])

            '''get transition matrix phi'''
            phi = np.eye(7)
            phi[0:4, 0:4] = MdeltaTheta.copy()
            boxI = np.array([
                [q[3], -q[2], q[1]],
                [q[2], q[3], -q[0]],
                [-q[1], q[0], q[3]],
                [-q[0], -q[1], -q[2]]
            ])
            phi[0:4, 4:] = MdeltaTheta.copy() @ boxI * (-1.0 / 2)

            '''get Gt '''
            Gt = np.zeros((7, 6))
            Gt[0:4, 0:3] = boxI * (-1.0 / 2)
            Gt[4:, 3:] = np.eye(3)

            '''covariance prediction'''
            if isFirst:
                sigma1 = (1e-11) ** 2  # Gaussian noise
                sigma2 = (1e-11) ** 2  # random walk noise
            else:
                sigma1 = (5e-13) ** 2  # Gaussian noise
                sigma2 = (5e-13) ** 2  # random walk noise
                # sigma1 = (1e-11) ** 2  # Gaussian noise
                # sigma2 = (1e-11) ** 2  # random walk noise

            Qt = np.diagflat([sigma1, sigma1, sigma1, sigma2, sigma2, sigma2]) * deltaT ** 2
            # Qt = np.eye(6) * deltaT ** 2 * sigma
            # Qt[3:, 3:] = np.zeros((3, 3))
            P_predict = phi @ P_update @ phi.T + phi @ Gt @ Qt @ Gt.T @ phi.T

            '''judge whether there is a SCA observation'''
            thisTime = self.__isSCAexist(ept, lastTime=lastTime)
            thisTime = self.__isSCAvalid(thisTime=thisTime)

            SCAexist = thisTime['isExist']

            '''update the prediction with new measurements'''
            # if i >= len(Ts)-2:
            #     SCAexist = False
            H = self.__getH(SCAexist)
            Y = self.__getY(t=ept, SCA_exist=SCAexist, thisTime=thisTime, X=H@X_predict)
            R = self.__getR(t=ept, SCA_exist=SCAexist, X=X_predict, thisTime=thisTime)

            if len(Y) == 0 or onlyPredict:
                X_update, P_update = X_predict, P_predict
            else:
                X_update, P_update = self.__update(X=X_predict, Y=Y, H=H, P=P_predict, R=R)

            '''Normalization'''
            X_update[0:4] = X_update[0:4] / np.linalg.norm(X_update[0:4])

            lastTime = thisTime
            fusion.append(X_update[[3, 0, 1, 2]].copy())
            fusionTime.append(ept)
            bias.append(X_update[4:])
        # print(X_update[4:])
        '''record the result'''
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'SCA1B_%s_%s_Fusion_%s.txt' % (self.date, self.sat.name, version)

        with open(fileDir + filename, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            for index in range(len(fusion)):
                vv = fusion[index]
                vt = fusionTime[index]
                file.write('%16.5f %3s   % 20.16f   % 20.16f   % 20.16f  % 20.16f\n' %
                           (vt, self.sat.name, vv[0], vv[1], vv[2], vv[3]))

        '''record the P matrix and bias, as the initial value for the day after this day'''
        np.save(fileDir + 'P_%s.npy' % self.sat.name, np.array(P_update))
        np.save(fileDir + 'Bias_%s.npy' % self.sat.name, np.array(X_update[4:]))
        np.save(fileDir + 'BiasAll_%s.npy' % self.sat.name, np.array(bias))

        pass

    def __isSCAexist(self, ept, lastTime: dict):
        """
        judge if there is a SCA measurement at given epoch, in an efficient way
        :return:
        """

        n1 = lastTime['index'][SCAID.No_1]
        n2 = lastTime['index'][SCAID.No_2]
        n3 = lastTime['index'][SCAID.No_3]

        [t1, t2, t3] = self.__timelist

        thisTime = {'isExist': {},
                    'index': {}}

        i, j, k = 0, 0, 0
        while 1:
            if np.fabs(t1[n1 + i] - ept) < 0.001:
                thisTime['isExist'][SCAID.No_1] = True
                thisTime['index'][SCAID.No_1] = n1 + i
                break
            elif t1[n1 + i] > ept + 0.001:
                thisTime['isExist'][SCAID.No_1] = False
                thisTime['index'][SCAID.No_1] = n1 + i
                break
            else:
                i += 1

        while 1:
            if np.fabs(t2[n2 + j] - ept) < 0.001:
                thisTime['isExist'][SCAID.No_2] = True
                thisTime['index'][SCAID.No_2] = n2 + j
                break
            elif t2[n2 + j] > ept + 0.001:
                thisTime['isExist'][SCAID.No_2] = False
                thisTime['index'][SCAID.No_2] = n2 + j
                break
            else:
                j += 1

        while 1:
            if np.fabs(t3[n3 + k] - ept) < 0.001:
                thisTime['isExist'][SCAID.No_3] = True
                thisTime['index'][SCAID.No_3] = n3 + k
                break
            elif t3[n3 + k] > ept + 0.001:
                thisTime['isExist'][SCAID.No_3] = False
                thisTime['index'][SCAID.No_3] = n3 + k
                break
            else:
                k += 1

        return thisTime

    def __isSCAvalid(self, thisTime):
        SCA = self.SCA

        for id in SCAID:
            index = thisTime['index'][id]
            if not thisTime['isExist'][id]:
                continue
            if SCA[id][5, index] > 6:
                # todo: > or >= ??, please check
                thisTime['isExist'][id] = False
        return thisTime

    def __update(self, X, Y, H, P, R):
        K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
        return X + np.dot(K, Y - np.dot(H, X)), np.dot(np.identity(7) - np.dot(K, H), P)

    def __getH(self, SCA_exist: dict):
        H1  = np.eye(N=4, M=7, k=0)
        H2 = np.eye(N=4, M=7, k=0)
        H3 = np.eye(N=4, M=7, k=0)
        H1[0:4,0:4] = self.QSA[0]
        H2[0:4, 0:4] = self.QSA[1]
        H3[0:4, 0:4] = self.QSA[2]

        Hmat = np.zeros(shape=(0, 7))
        if SCA_exist[SCAID.No_1]:
            Hmat = np.vstack((Hmat, H1))

        if SCA_exist[SCAID.No_2]:
            Hmat = np.vstack((Hmat, H2))

        if SCA_exist[SCAID.No_3]:
            Hmat = np.vstack((Hmat, H3))

        return Hmat

    def __getY(self, t, SCA_exist: dict, thisTime: dict, X):
        """
        get y in JPL convention
        :param t:
        :param SCA_exist:
        :param thisTime:
        :param X:
        :return:
        """
        SCA = self.SCA

        Y = []
        for id in SCAID:
            if SCA_exist[id]:
                '''flip the sign'''
                y = SCA[id][1:5, thisTime['index'][id]][[1, 2, 3, 0]]
                if np.dot(X[0:3], y[0:3]) < 0:
                    Y.append(-y)
                else:
                    Y.append(y)

        Y = np.array(Y).flatten()

        return Y

    def __getR(self, t, SCA_exist: dict, X, thisTime: dict):
        """
        constant R
        :param t:
        :return:
        """

        R = []

        ratio = []
        SCA = self.SCA
        for id in SCAID:
            rr = 1
            # rr = 0.5 * (SCA[id][5, thisTime['index'][id]] + 1)
            #
            # '''flip the sign'''
            # y = SCA[id][1:5, thisTime['index'][id]]
            # theta = np.dot(X[0:3], y[1:4])
            # if theta < 0:
            #     m = 180
            # else:
            #     m = 0
            # diff_deg = np.rad2deg(np.arccos(theta))
            # if np.fabs(diff_deg - m) < 5:
            #     ratio.append(1)
            # else:
            #     ratio.append(1e3)

            ratio.append(rr)

        q1 = 1e-8
        q2 = 1e-8
        q3 = 100e-8
        q4 = 1e-10


        if SCA_exist[SCAID.No_1]:
            R += [q1 * ratio[0], q2 * ratio[0], q3 * ratio[0], q4 * ratio[0]]

        if SCA_exist[SCAID.No_2]:
            R += [q1 * ratio[1], q2 * ratio[1], q3 * ratio[1], q4 * ratio[1]]

        if SCA_exist[SCAID.No_3]:
            R += [q1 * ratio[2], q2 * ratio[2], q3 * ratio[2], q4 * ratio[2]]

        return np.diagflat(R)

    def setQSA(self, QSA):
        self.QSA = self.__Q2TM(QSA=QSA)
        return self

    def __Q2TM(self, QSA):
        QSA_M = []
        for i in range(3):
            QSA_M.append(self.__Qmul_JPL(q=QSA[i]).T)
        return QSA_M

    def __Qmul(self, q):
        a, b, c, d = q.w, q.x, q.y, q.z
        M = np.array([
            [a, -b, -c, -d],
            [b, a, d, -c],
            [c, -d, a, b],
            [d, c, -b, a]
        ])
        return M

    def __Qmul_JPL(self, q):
        a, b, c, d = q.w, q.x, q.y, q.z
        M = np.array([
            [a, d, -c, b],
            [-d, a, b, c],
            [c, -b, a, d],
            [-b, -c, -d, a]
        ])
        return M
