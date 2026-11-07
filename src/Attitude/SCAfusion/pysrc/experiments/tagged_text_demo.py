#!/user/bin/env python3
# -*- coding: utf-8 -*-
from pysrc.core.settings import SatID, SCAID, Mission, IMUID
from pysrc.fusion.multi_sca import SCAweighting
import numpy as np
import os
import quaternion
from pysrc.core.geometry import GeoMathKit
from enum import Enum
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
import matplotlib.pyplot as plt

class text():
    def __init__(self,satID: SatID, date, TIM1B, CLK1B,  IMU1A, IMU1B, SCA1A=None, SCA1B=None):
        self.sat = satID
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self.__SCA1B = SCA1B
        self.__IMU1A = IMU1A
        self.__IMU1B = IMU1B
        self._QSA = self.ConfigSCF2SRF()


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

        text.SCF2SRF = QSA

        return QSA

        pass

    def TimeTagTransformation(self):
        imu=self.__IMU1A
        '''RCV ---> GPS'''
        clk = self.__CLK1B
        # make a judgement on the quality
        index = np.where(clk[2] > 0)[0] #返回的是数据位置信息
        #  this should be refined by dividing the arcs
        if index[0] == 0 and index[-1] == (len(clk[2]) - 1):
            pass
        else:
            raise Warning('Take care of CLK1B file')

        clk = clk[:, np.where(clk[2] < 0.1)[0]]
        for key in IMUID:
            imuValue = imu[key.name]
            x = imuValue[0] - clk[0][0] + imuValue[1] * 1e-6
            xp = clk[0] - clk[0][0]
            yp = clk[1]
            V = np.interp(x=x, xp=xp, fp=yp)
            V_int = np.trunc(V)
            V_frac = V - np.trunc(V)


            imuValue[0] += V_int
            imuValue[1] += V_frac * 1e6

            imuValue[0][imuValue[1] < 0] -= 1
            imuValue[1][imuValue[1] < 0] += 1e6

            imu[key.name] = imuValue
            pass

        return imu

    def timetagText(self):
        imu1A=self.TimeTagTransformation()
        imu1B=self.__IMU1B
        delta={}
        for key in IMUID:
            T1 = imu1A[key.name][0] - imu1A[key.name][0][0] + imu1A[key.name][1] * 1e-6
            T2 = imu1B[key.name][0] - imu1B[key.name][0][0] + imu1B[key.name][1] * 1e-6
            deltaT=T2-T1
            delta[key.name]=np.vstack((np.arange(0,len(deltaT)),deltaT))
            span0=delta[key.name][:,np.abs(delta[key.name][1,:])<=0.3*1e-6]
            span1=delta[key.name][:,(np.abs(delta[key.name][1,:])<=0.6*1e-6) & (np.abs(delta[key.name][1,:])>0.3*1e-6)]
            span2=delta[key.name][:,(np.abs(delta[key.name][1,:])>0.6*1e-6) & (np.abs(delta[key.name][1,:])<1e-6)]
            span3=delta[key.name][:,np.abs(delta[key.name][1,:])>=1e-6]
            P0 = len(span0[0]) / len(deltaT)
            P1 = len(span1[0]) / len(deltaT)
            P2 = len(span2[0]) / len(deltaT)
            P3 = len(span3[0]) / len(deltaT)

            print('%s:%s,%s,%s,%s'%(key.name,P0,P1,P2,P3))
            print('mean:%s,std:%s'%(np.mean(deltaT),np.std(deltaT)))

            self.plot(span0,span1,span2,span3)
            pass


    def plot(self,span0,span1,span2,span3):
        fig, ax = plt.subplots(figsize=(12.8, 4.8))

        # color_x = (31 / 255, 119 / 255, 180 / 255)
        # color_y = (229 / 255, 114 / 255, 12 / 255)
        # color_z = (35 / 255, 142 / 255, 35 / 255)

        ax.scatter(span0[0], span0[1], s=0.1, color='green', label='span0', alpha=0.5)
        ax.scatter(span1[0], span1[1], s=0.1, color='green', label='span1', alpha=0.5)
        ax.scatter(span2[0], span2[1], s=0.1, color='blue', label='span2', alpha=0.5)
        ax.scatter(span3[0], span3[1], s=0.1, color='red', label='span3', alpha=0.5)

        ax.axhline(y=1e-6, color='blue', linestyle='--')
        ax.axhline(y=-1e-6, color='blue', linestyle='--')
        ax.axhline(y=2*1e-6, color='blue', linestyle='--')
        ax.axhline(y=-2*1e-6, color='blue', linestyle='--')

        ax.legend()
        plt.show()

        pass





def demo():
    date='2019-01-01'
    sat=SatID.C
    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1A')
    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1B')
    TIM1B=L1B.getTIM(sat=sat)
    CLK1B=L1B.getCLK(sat=sat)
    IMU1A=L1A.getIMU(sat=sat)
    IMU1B=L1B.getIMU(sat=sat)

    a=text(satID=sat,date=date,TIM1B=TIM1B,CLK1B=CLK1B,IMU1A=IMU1A,IMU1B=IMU1B)
    # imu=a.TimeTagTransformation()
    a.timetagText()
    pass

if __name__ == '__main__':
    demo()