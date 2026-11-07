from pysrc.core.settings import SatID, SCAID, Mission
# from pysrc.fusion.multi_sca import SCAweighting
import numpy as np
import os
import quaternion
from pysrc.core.geometry import GeoMathKit
from enum import Enum
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B

import statistics

import datetime
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class SCAinterOption(Enum):
    NoInterp = 0
    OneHZ = 1
    TwoHZ = 2


class Demtwo:
    SCF2SRF = None

    def __init__(self, satID: SatID, date, SCA1A, SCA1B, TIM1B, CLK1B):
        self.sat = satID
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self.__SCA1B = SCA1B

        self.__interpOption = SCAinterOption.NoInterp
        pass

    def configInterp(self, interpOption: SCAinterOption):
        """

        :param interpOption: 0--> no interpolation,
        1--> interp at integer seconds, 2---> interp at every half sec
        :return:
        """
        self.__interpOption = interpOption
        pass

    @staticmethod
    def ConfigSCF2SRF():
        QSA = {}
        file = '../data/GRACE_FO/Auxiliary/QSA1B_2018-05-22_C_04.txt'
        skip = GeoMathKit.ReadEndOfHeader(file_name=file, end_flag='End of YAML header')
        q = np.loadtxt(file, skiprows=skip, usecols=[3, 4, 5, 6])
        QSA[SatID.C] = quaternion.as_quat_array(q)

        file = '../data/GRACE_FO/Auxiliary/QSA1B_2018-05-22_D_04.txt'
        skip = GeoMathKit.ReadEndOfHeader(file_name=file, end_flag='End of YAML header')
        q = np.loadtxt(file, skiprows=skip, usecols=[3, 4, 5, 6])
        QSA[SatID.D] = quaternion.as_quat_array(q)

        Demtwo.SCF2SRF = QSA

        pass

    def TimeTagTransformation(self):
        """
        Step 1: time tag transformation: OBC--->RCV, RCV--->GPS
        :return:
        """

        '''Step 1: OBC--->RCV'''
        sca = self.__SCA1A

        tim = self.__TIM1B

        for key in SCAID:
            scaValue = sca[key.name]
            # remove the first epoch to avoid too big number
            x = scaValue[0] - tim[0][0] + scaValue[1] * 1e-6
            xp = tim[0] - tim[0][0]
            yp = tim[1] - tim[0][0] + tim[2] * 1e-9
            # quality check for TIM1B
            valid = tim[3] < 1000
            print(valid)
            xp = xp[valid]
            yp = yp[valid]
            print(xp)
            V = np.interp(x=x, xp=xp, fp=yp)

            V_int = np.trunc(V) + tim[0][0]
            V_frac = V - np.trunc(V)
            scaValue[0] = V_int
            scaValue[1] = V_frac * 1e6
            sca[key.name] = scaValue

        '''Step 2: RCV ---> GPS'''
        clk = self.__CLK1B
        # make a judgement on the quality
        index = np.where(clk[2] > 0)[0]  # 返回的是数据位置信息
        # TODO: this should be refined by dividing the arcs
        if index[0] == 0 and index[-1] == (len(clk[2]) - 1):
            pass
        else:
            raise Warning('Take care of CLK1B file')

        clk = clk[:, np.where(clk[2] < 0.1)[0]]
        for key in SCAID:
            scaValue = sca[key.name]
            x = scaValue[0] - clk[0][0] + scaValue[1] * 1e-6
            xp = clk[0] - clk[0][0]
            yp = clk[1]
            V = np.interp(x=x, xp=xp, fp=yp)
            V_int = np.trunc(V)
            V_frac = V - np.trunc(V)

            # V_int[V_frac < 0] -= 1
            # V_frac[V_frac < 0] += 1

            scaValue[0] += V_int
            scaValue[1] += V_frac * 1e6

            scaValue[0][scaValue[1] < 0] -= 1
            scaValue[1][scaValue[1] < 0] += 1e6

            sca[key.name] = scaValue
            pass

        return sca

    def InterpSCA(self):
        # Unify the time series of Sca1A and Sca1B
        sat = SatID.C
        sca1B = self.__SCA1B
        sca1A = self.TimeTagTransformation()
        Sca1B = {}
        Sca1A = {}

        for key in SCAID:
            scaValue = sca1B
            SCA1A_ID = sca1A[key.name]
            starttime = SCA1A_ID[0][0]
            time1 = SCA1A_ID[0] - starttime + SCA1A_ID[1] * 1e-6
            time2 = scaValue[0] - starttime
            Demtwo.ConfigSCF2SRF()
            QSA = Demtwo.SCF2SRF
            qb = quaternion.as_quat_array(SCA1A_ID[2:6, :].T)
            if key.name == 'No_1':
                qa = qb * QSA[SatID.C][0]
            elif key.name == 'No_2':
                qa = qb * QSA[SatID.C][1]
            elif key.name == 'No_3':
                qa = qb * QSA[SatID.C][2]
            else:
                pass

            qc = quaternion.as_float_array(qa).T
            SCA1A_ID[2:6, :] = qc

            # interpolation
            new = quaternion.squad(R_in=qa, t_in=time1, t_out=time2)
            qc = quaternion.as_float_array(new).T
            sca = [[], [], [], [], []]
            sca[0] = scaValue[0]
            sca[1:5] = -qc
            Sca1A[key.name] = np.array(sca)
            Sca1B[key.name] = scaValue

        return Sca1A, Sca1B, sat

    def res(self):
        sca1A, sac1B, sat = self.InterpSCA()
        res = {}

        res_1 = sca1A['No_1']
        res_2 = sca1A['No_2']
        res_3 = sca1A['No_3']
        q1 = quaternion.as_quat_array(res_1[1:5, :].T)
        q2 = quaternion.as_quat_array(res_2[1:5, :].T)
        q3 = quaternion.as_quat_array(res_3[1:5, :].T)
        number = len(res_1[0, :])

        '''the residual of head#1 & head#2 '''
        for i in range(number):
            q1[i] = q1[i].inverse()
            pass
        res_12 = q1 * q2
        res_12 = quaternion.as_float_array(res_12).T
        res['No_12'] = np.vstack((res_1[0], res_12))

        '''the residual of head#1 & head#3'''
        res_13 = q1 * q3
        res_13 = quaternion.as_float_array(res_13).T
        res['No_13'] = np.vstack((res_1[0], res_13))

        '''the residual of head#2 & head#3'''
        for i in range(number):
            q2[i] = q2[i].inverse()
            pass
        res_23 = q2 * q3
        res_23 = quaternion.as_float_array(res_23).T
        res['No_23'] = np.vstack((res_2[0], res_23))

        return res

    pass

    @staticmethod
    def getEular(res):
        keys = ['No_12', 'No_13', 'No_23']
        Tx = []
        Ty = []
        Tz = []
        SCF_x = []
        SCF_y = []
        SCF_z = []
        SD = {}
        eular={}
        for key in keys:
            number = len(res[key][0, :])
            for i in range(number):
                if res[key][1, i] > 0.99:

                    Tx.append(res[key][0, i])
                    Ty.append(res[key][0, i])
                    Tz.append(res[key][0, i])
                    SCF_x.append(float(res[key][2, i]) * 2 * 3600)
                    SCF_y.append(float(res[key][3, i]) * 2 * 3600)
                    SCF_z.append(float(res[key][4, i]) * 2 * 3600)

                else:
                    pass
            idx = key + '_x'
            idy = key + '_y'
            idz = key + '_z'
            SD[idx] = statistics.stdev(SCF_x)
            SD[idy] = statistics.stdev(SCF_y)
            SD[idz] = statistics.stdev(SCF_z)
            eular[idx]=np.vstack((Tx,SCF_x))
            eular[idy]=np.vstack((Ty,SCF_y))
            eular[idz] = np.vstack((Tz, SCF_z))

        return eular,SD




if __name__ == '__main__':
    day = '2019-01-15'
    date = day
    sat = SatID.C

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    SCA1B = L1B.getSCA(sat)
    a = Demtwo(satID=sat, date=date, SCA1A=SCA1A, SCA1B=SCA1B, TIM1B=TIM1B, CLK1B=CLK1B)
    # Sca1A,Sca1B,sat=a.InterpSCA()
    res = a.res()
    eular,SD=a.getEular(res)
    datex=eular['No_12_x'][0:2,np.abs(eular['No_12_x'][-1,:])<10]

    datey=eular['No_12_y'][0:2,np.abs(eular['No_12_y'][-1,:])<15]

    datez=eular['No_12_z'][0:2,np.abs(eular['No_12_y'][-1,:])<10]


    plt.scatter(datey[0], datey[1], color='blue', label='No_12_y',s=0.5)
    plt.scatter(datez[0], datez[1], color='green', label='No_12_z',s=0.5,alpha=0.5)
    plt.scatter(datex[0], datex[1],color='red',label='No_12_x',s=0.5,alpha=0.2)
    # plt.plot(datex[0], datex[1],color='red',label='No_12_x',alpha=0.5)
    # plt.plot(datey[0], datey[1], color='blue', label='No_12_y',alpha=0.5)
    # plt.plot(datez[0], datez[1], color='green', label='No_12_z',alpha=0.5)

    plt.legend()


    plt.xlabel('time(s)')
    plt.ylabel('angle(arcsec)')

    plt.show()



    pass
