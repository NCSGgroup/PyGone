"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/21
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


class NormalKalmanFilter:
    """
    Implement the kalman filter by day.
    Such a filter treat the problem as a linear system, and in this way the error is too big.
    """

    def __init__(self, date, Mission: Mission, sat: SatID):
        self.date = date
        self.mission = Mission
        self.sat = sat
        self.__IMUoption, self.__SCAoption = None, None
        self.IMU = None
        self.SCA = None
        self.__timelist = None
        pass

    def configInput(self, IMUoption: IMUinterpOption, SCAoption: SCAinterOption):
        self.__IMUoption = IMUoption
        self.__SCAoption = SCAoption
        return self

    def load(self):
        version = 'HUST01'
        mission = self.mission
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        print('\nLoading IMU input ...')
        filename = 'IMU1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__IMUoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        IMU = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 2, 3, 4), unpack=False)

        print('\nLoading SCA input ...')
        filename = 'SCA1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__SCAoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 1, 5, 6, 7, 8, 9, 10, 11, 4), unpack=True)
        SCA = {}
        SCA[SCAID.No_1] = res[0:-1, np.fabs(res[-1, :] - 1) < 0.001]
        SCA[SCAID.No_2] = res[0:-1, np.fabs(res[-1, :] - 2) < 0.001]
        SCA[SCAID.No_3] = res[0:-1, np.fabs(res[-1, :] - 3) < 0.001]

        self.IMU = IMU
        self.SCA = SCA
        t1 = SCA[SCAID.No_1][0] + SCA[SCAID.No_1][1] * 1e-6
        t2 = SCA[SCAID.No_2][0] + SCA[SCAID.No_2][1] * 1e-6
        t3 = SCA[SCAID.No_3][0] + SCA[SCAID.No_3][1] * 1e-6
        self.__timelist = [t1, t2, t3]
        return self

    def filter(self, iniX, iniP, iniT, onlyPredict=False, version='HUST01'):
        """
        Start filtering
        :param onlyPredict: True when there are no SCA measurements
        :param iniX: initial value of X -- state vector
        :param iniP: initial value of P -- covariance of the state vector
        :param iniT: initial time epoch
        :param version: no actual meaning but a flag.
        :return:
        """

        X_update, P_update = iniX, iniP

        IMU, SCA = self.IMU, self.SCA

        Ts = IMU[:, 0]

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

        fusion = [X_update[:4]]
        fusionTime = [Ts[startT]]
        print('\nKalman fusion starts ...')
        for i in trange(startT + 1, len(Ts)):
            ept = Ts[i]
            if ept > SCA[SCAID.No_1][0][-1] + SCA[SCAID.No_1][1][-1] * 1e-6:
                print('\nKalman filter finished')
                break
            # angular velocity in radian/sec
            w_meas = IMU[i, 1:]
            w = w_meas - X_update[4:]
            q = X_update[:4]
            deltaT = ept - Ts[i - 1]
            A = self.__getA(w=w, q=q, deltaT=deltaT)
            Q = self.__getQ(ept)
            # make a prediction
            X_predict, P_predict = self.__predict(X=X_update.copy(), P=P_update.copy(), A=A, Q=Q)
            thisTime = self.__isSCAexist(ept, lastTime=lastTime)
            thisTime = self.__isSCAvalid(thisTime=thisTime)

            SCAexist = thisTime['isExist']
            # update the prediction with new measurements
            Y = self.__getY(t=ept, SCA_exist=SCAexist, thisTime=thisTime, X=X_predict)
            H = self.__getH(SCAexist)
            R = self.__getR(t=ept, SCA_exist=SCAexist, X=X_predict, thisTime=thisTime)

            if len(Y) == 0 or onlyPredict:
                X_update, P_update = X_predict, P_predict
            else:
                X_update, P_update = self.__update(X=X_predict, Y=Y, H=H, P=P_predict, R=R)

            lastTime = thisTime
            fusion.append(X_update[:4].copy())
            fusionTime.append(ept)

        '''record the result'''
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, self.date)
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
            if SCA[id][6, index] >= 6:
                thisTime['isExist'][id] = False
        return thisTime

    def __getA(self, w, q, deltaT):
        w_star = w * deltaT
        q_star = q * deltaT
        F1 = [1, -w_star[0] / 2, -w_star[1] / 2, -w_star[2] / 2, q_star[1] / 2, q_star[2] / 2, q_star[3] / 2]
        F2 = [w_star[0] / 2, 1, w_star[2] / 2, -w_star[1] / 2, -q_star[0] / 2, q_star[3] / 2, -q_star[2] / 2]
        F3 = [w_star[1] / 2, -w_star[2] / 2, 1, w_star[0] / 2, -q_star[3] / 2, -q_star[0] / 2, q_star[1] / 2]
        F4 = [w_star[2] / 2, w_star[1] / 2, -w_star[0] / 2, 1, q_star[2] / 2, -q_star[1] / 2, -q_star[0] / 2]
        F5 = [0, 0, 0, 0, 1, 0, 0]

        F6 = [0, 0, 0, 0, 0, 1, 0]
        F7 = [0, 0, 0, 0, 0, 0, 1]

        return np.array([F1, F2, F3, F4, F5, F6, F7])

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
        SCA = self.SCA

        Y = []
        for id in SCAID:
            if SCA_exist[id]:
                '''flip the sign'''
                y = SCA[id][2:6, thisTime['index'][id]]
                if np.dot(X[1:4], y[1:4]) < 0:
                    Y.append(-y)
                else:
                    Y.append(y)

        Y = np.array(Y).flatten()

        return Y

    def __predict(self, X, P, A, Q):
        return np.dot(A, X), np.dot(np.dot(A, P), A.T) + Q

    def __update(self, X, Y, H, P, R):
        K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
        return X + np.dot(K, Y - np.dot(H, X)), np.dot(np.identity(7) - np.dot(K, H), P)

    def __getQ(self, t):
        """
        constant Q
        :param t:
        :return:
        """

        Q = np.identity(7) * 1e-6

        return Q

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
            '''flip the sign'''
            y = SCA[id][2:6, thisTime['index'][id]]
            diff_deg = np.rad2deg(np.fabs(np.arccos(np.dot(X[1:4], y[1:4]))))
            if diff_deg < 5:
                ratio.append(1)
                continue
            else:
                ratio.append(1e3)

        if SCA_exist[SCAID.No_1]:
            R += [ratio[0], ratio[0], ratio[0], ratio[0]]

        if SCA_exist[SCAID.No_2]:
            R += [ratio[1], ratio[1], ratio[1], ratio[1]]

        if SCA_exist[SCAID.No_3]:
            R += [ratio[2], ratio[2], ratio[2], ratio[2]]

        return np.diagflat(R) * 5e-5

@DeprecationWarning
class MEKF_7Vars_JPL:
    """
    Multiplication extended kalman filtering with 7 variables as the state vectors:
    four-element quaternion + 3 gyro bias

    Quaternion in convention of JPL

    Zeroth order

    Reference:
    1. Kalman filtering for spacecraft attitude estimation
    2. Indirect Kalman filtering for 3D attitude estimation
    """

    def __init__(self, date, Mission: Mission, sat: SatID):
        self.date = date
        self.mission = Mission
        self.sat = sat
        self.__IMUoption, self.__SCAoption = None, None
        self.IMU = None
        self.SCA = None
        self.__timelist = None
        pass

    def configInput(self, IMUoption: IMUinterpOption, SCAoption: SCAinterOption):
        self.__IMUoption = IMUoption
        self.__SCAoption = SCAoption
        return self

    def load(self):
        version = 'HUST01'
        mission = self.mission
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        print('\nLoading IMU input ...')
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
        return self

    def filter(self, iniP, iniX=None, iniT=None, onlyPredict=False, version='HUST01'):
        """
        Start filtering
        :param onlyPredict: True when there are no SCA measurements
        :param iniX: initial value of X -- state vector
        :param iniP: initial value of P -- covariance of the state vector
        :param iniT: initial time epoch
        :param version: no actual meaning but a flag.
        :return:
        """

        IMU, SCA = self.IMU, self.SCA
        Ts = IMU[:, 0]

        if iniX is None and iniT is None:
            i = 0
            while 1:
                whichX = [int(SCA[d][5, i]) for d in SCAID]
                if min(whichX) >= 6:
                    i += 1
                    continue

                ini = SCAID(whichX.index(min(whichX)) + 1)
                iniT = SCA[ini][0, i]
                iniX = np.array(list(SCA[ini][1:5, i]) + [0, 0, 0])

                if iniT >= Ts[0]:
                    break
                else:
                    i += 1
        else:
            assert not None == iniT, 'Input error in Kalman filtering!'

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

        # w_list = []
        print('\nKalman fusion starts ...')
        for i in trange(startT + 1, len(Ts)):
            ept = Ts[i]
            if ept > SCA[SCAID.No_1][0][-1] + SCA[SCAID.No_1][1][-1] * 1e-6:
                print('\nKalman filter finished')
                break
            '''get angular velocity from IMU, [radian/sec]'''
            w_meas = IMU[i, 1:]

            '''remove the bias from the angular velocity'''
            w = w_meas - X_update[4:]
            # w_list.append(w)
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
            MdeltaTheta = np.eye(4) * np.cos(AbsDeltaTheta) + \
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
            sigma1 = (1e-9) ** 2  # Gaussian noise
            sigma2 = (3e-9) ** 2  # random walk noise
            Qt = np.diagflat([sigma1, sigma1, sigma1, sigma2, sigma2, sigma2]) * deltaT ** 2
            # Qt = np.eye(6) * deltaT ** 2 * sigma
            # Qt[3:, 3:] = np.zeros((3, 3))
            P_predict = phi @ P_update @ phi.T + phi @ Gt @ Qt @ Gt.T @ phi.T

            '''judge whether there is a SCA observation'''
            thisTime = self.__isSCAexist(ept, lastTime=lastTime)
            thisTime = self.__isSCAvalid(thisTime=thisTime)

            SCAexist = thisTime['isExist']

            # update the prediction with new measurements
            Y = self.__getY(t=ept, SCA_exist=SCAexist, thisTime=thisTime, X=X_predict)
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

        '''record the result'''
        # np.save('w.npy', np.array(w_list))
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, self.date)
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
            if SCA[id][5, index] >= 6:
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
            rr = 0.5 * (SCA[id][5, thisTime['index'][id]] + 1)

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

        q1 = q2 = q3 = q4 = 1e-7

        if SCA_exist[SCAID.No_1]:
            R += [q1 * ratio[0], q2 * ratio[0], q3 * ratio[0], q4 * ratio[0]]

        if SCA_exist[SCAID.No_2]:
            R += [q1 * ratio[1], q2 * ratio[1], q3 * ratio[1], q4 * ratio[1]]

        if SCA_exist[SCAID.No_3]:
            R += [q1 * ratio[2], q2 * ratio[2], q3 * ratio[2], q4 * ratio[2]]

        return np.diagflat(R)


class MEKF_7Vars_JPL_combination:
    """
    Multiplication extended kalman filtering with 7 variables as the state vectors:
    four-element quaternion + 3 gyro bias

    Quaternion in convention of JPL

    Zeroth order

    SCA has been combined already before the kalman filtering.

    Reference:
    1. Kalman filtering for spacecraft attitude estimation
    2. Indirect Kalman filtering for 3D attitude estimation
    """

    def __init__(self, date, Mission: Mission, sat: SatID):
        self.date = date
        self.mission = Mission
        self.sat = sat
        self.__IMUoption, self.__SCAoption = None, None
        self.IMU = None
        self.SCA = None
        self.__timelist = None
        pass

    def configInput(self, IMUoption: IMUinterpOption, SCAoption: SCAinterOption):
        self.__IMUoption = IMUoption
        self.__SCAoption = SCAoption
        return self

    def load(self):
        version = 'HUST01'
        mission = self.mission
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        print('\nKalman filter loading')
        print('Loading IMU input ...')
        filename = 'IMU1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__IMUoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        IMU = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 2, 3, 4), unpack=False)

        print('Loading SCA input ...')
        filename = 'SCA1B_%s_%s_%s_combined_%s.txt' % (self.date, self.sat.name, self.__SCAoption.name, version)
        skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
        SCA = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 3, 4, 5, 6), unpack=True)

        self.IMU = IMU
        self.SCA = SCA
        self.__timelist = SCA[0, :]
        return self

    def filter(self, iniP=None, iniQ=None, iniB = None, iniT=None, onlyPredict=False, isFirst=False,
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

        fileDir = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, self.date)
        today = datetime.datetime.strptime(self.date, "%Y-%m-%d")
        previous_day = today + datetime.timedelta(days=-1)
        fileDir_previous = '../result/product/%s/RL04/L1B/%s/' % (self.mission.name, previous_day.strftime("%Y-%m-%d"))
        if iniP is None:
            iniP = np.load(fileDir_previous +'P_%s.npy' % self.sat.name)

        if iniB is None:
            iniB = np.load(fileDir_previous+'Bias_%s.npy'% self.sat.name)

        IMU, SCA = self.IMU, self.SCA
        Ts = IMU[:, 0]

        iniX = None
        if iniT is None:
            i = 0
            while 1:

                iniT = SCA[0, i]
                iniX = np.array(list(SCA[1:5, i]) + list(iniB))

                if iniT >= Ts[0]:
                    break
                else:
                    i += 1
        else:
            ii = np.fabs(SCA[0] - iniT)<0.01
            iniX = np.array(list(SCA[1:5, ii].flatten()) + list(iniB))
            pass

        '''JPL convention'''
        X_update, P_update = iniX[[1, 2, 3, 0, 4, 5, 6]], iniP

        if iniT < Ts[0]:
            raise Exception('Kalman input error: no sufficient data')

        startT = np.where(np.fabs(Ts - iniT) < 0.0001)[0][0]
        if startT is None:
            raise Exception('Kalman input error: time mismatch')

        lastTime = {'index': 0}

        '''record the value in Hamilton convention'''
        fusion = [X_update[[3, 0, 1, 2]]]
        fusionTime = [Ts[startT]]
        bias = []

        print('\nKalman fusion starts ...')
        for i in trange(startT + 1, len(Ts)):
            ept = Ts[i]
            if ept > self.__timelist[-1]:
                print('\nKalman filter finished')
                break
            '''get angular velocity from IMU, [radian/sec]'''
            w_meas = IMU[i-1, 1:]

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

            MdeltaTheta = np.eye(4) * np.cos(AbsDeltaTheta/2.0) + \
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

            SCAexist = thisTime['isExist']

            '''update the prediction with new measurements'''
            Y = self.__getY(t=ept, SCA_exist=SCAexist, thisTime=thisTime, X=X_predict)
            if SCAexist and np.any(np.isnan(Y)):
                SCAexist = False
            # if i >= len(Ts)-2:
            #     SCAexist = False
            H = self.__getH()
            R = self.__getR(t=ept, SCA_exist=SCAexist, X=X_predict, thisTime=thisTime)

            if (not SCAexist) or onlyPredict:
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
        np.save(fileDir+'P_%s.npy' % self.sat.name, np.array(P_update))
        np.save(fileDir+'Bias_%s.npy'% self.sat.name, np.array(X_update[4:]))
        np.save(fileDir + 'BiasAll_%s.npy' % self.sat.name, np.array(bias))

        pass

    def __isSCAexist(self, ept, lastTime: dict):
        """
        judge if there is a SCA measurement at given epoch, in an efficient way
        :return:
        """

        n1 = lastTime['index']

        t1 = self.__timelist

        thisTime = {'isExist': None,
                    'index': None}

        i = 0
        while 1:
            if np.fabs(t1[n1 + i] - ept) < 0.001:
                thisTime['isExist'] = True
                thisTime['index'] = n1 + i
                break
            elif t1[n1 + i] > ept + 0.001:
                thisTime['isExist'] = False
                thisTime['index'] = n1 + i
                break
            else:
                i += 1

        return thisTime

    def __update(self, X, Y, H, P, R):
        K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
        return X + np.dot(K, Y - np.dot(H, X)), np.dot(np.identity(7) - np.dot(K, H), P)

    def __getH(self):

        return np.eye(N=4, M=7, k=0)

    def __getY(self, t, SCA_exist: bool, thisTime: dict, X):
        """
        get y in JPL convention
        :param t:
        :param SCA_exist:
        :param thisTime:
        :param X:
        :return:
        """
        SCA = self.SCA
        y = None
        if SCA_exist:
            '''flip the sign'''
            y = SCA[1:5, thisTime['index']][[1, 2, 3, 0]]
            if np.dot(X[0:3], y[0:3]) < 0:
                y = -y

        return y

    def __getR(self, t, SCA_exist: bool, X, thisTime: dict):
        """
        constant R
        :param t:
        :return:
        """
        if not SCA_exist:
            return None

        SCA = self.SCA

        ratio = 1
        '''flip the sign'''
        y = SCA[1:5, thisTime['index']]
        theta = np.dot(X[0:3], y[1:4]) / np.linalg.norm(X[0:3]) / np.linalg.norm(y[1:4])
        if np.arccos(np.fabs(theta))<0.05:
            ratio = 1
        else:
            ratio = 1e3
        # if theta < 0:
        #     m = 180
        # else:
        #     m = 0
        # diff_deg = np.rad2deg(np.arccos(theta))
        # if np.fabs(diff_deg - m) < 5:
        #     ratio = 1
        # else:
        #     ratio = 1e3

        q1 = q2 = q3 = q4 = 1e-10

        '''notice, q4 is q.w'''

        R = [q1 * ratio, q2 * ratio, q3 * ratio, q4 * ratio]

        return np.diagflat(R)


class GetFusionRes:
    def __init__(self, mission: Mission = Mission.GRACE_FO, date: str = '2019-08-06'):
        self.mission = mission
        self.date = datetime.datetime.strptime(date, "%Y-%m-%d")
        self._dataDir = None
        pass

    def configDir(self, DataDir: str):
        self._dataDir = os.path.join(DataDir, self.date.strftime("%Y-%m-%d"))
        return self

    def getFusion(self, sat=SatID.C, version='HUST01'):
        # file = 'SCA1B_%s_%s_Fusion_%s.txt' % (self.date.strftime("%Y-%m-%d"), sat.name, version)
        file = 'SCA1B_%s_%s_Fusion_Resample_%s.txt' % (self.date.strftime("%Y-%m-%d"), sat.name, version)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'END OF HEADER')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 2, 3, 4, 5), unpack=True)
        return res

    def getSCAcombine(self, sat=SatID.C, version='HUST01'):
        file = 'SCA1B_%s_%s_OneHZ_combined_%s.txt' % (self.date.strftime("%Y-%m-%d"), sat.name, version)
        file = os.path.join(self._dataDir, file)
        skip = GeoMathKit.ReadEndOfHeader(file, 'END OF HEADER')
        res = np.loadtxt(file, comments='#', skiprows=skip, usecols=(0, 3, 4, 5, 6), unpack=True)
        return res

    pass


def demo():
    date = '2018-12-02'
    mission = Mission.GRACE_FO
    sat = SatID.D

    kf = NormalKalmanFilter(date=date, Mission=mission, sat=sat)

    kf.configInput(IMUoption=IMUinterpOption.TwoHZ, SCAoption=SCAinterOption.TwoHZ)

    kf.load()

    # iniX = np.array([-0.9588564537920072, -0.007615164614332016, -0.006070889054580652, -0.2837242578757104,
    #                  0, 0, 0])
    # iniT = 596980800

    iniX = np.array([0.2836102973924611, - 0.006213228214132989,
                     0.008131200168201111, - 0.9588850184242815, 0, 0, 0])
    iniT = 596980800

    iniP = np.diagflat([1e-4, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])
    kf.filter(iniX, iniP, iniT, onlyPredict=False, version='HUST01')

    pass


if __name__ == '__main__':
    demo()
