from pysrc.core.settings import SatID, SCAID, Mission
# from pysrc.fusion.multi_sca import SCAweighting
import numpy as np
import os
import quaternion
from pysrc.core.geometry import GeoMathKit
from enum import Enum
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
from scipy.spatial.transform import Rotation

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

    def __init__(self, satID: SatID, date, SCA1A, SCA1B, TIM1B, CLK1B, GNI1B):
        self.sat = satID
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self.__SCA1B = SCA1B
        self.__GNI1B = GNI1B

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

    def quaternion_to_angular_velocity(self):
        sca1A = self.TimeTagTransformation()
        ang_vel = {}
        for key in SCAID:
            SCA1A_ID = sca1A[key.name]

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

            quaternions=qa
            starttime = sca1A[key.name][0,0]
            timestamps=sca1A[key.name][0]-starttime+sca1A[key.name][1] * 1e-6
            angular_velocities = []

            for i in range(len(quaternions) - 1):
                q1 = quaternions[i]
                q2 = quaternions[i + 1]
                t1 = timestamps[i]
                t2 = timestamps[i + 1]

                delta_q = q2 * q1.conjugate()
                delta_t = t2 - t1

                # 将四元数差异转换为角轴表示法
                rotation = Rotation.from_quat([delta_q.x, delta_q.y, delta_q.z, delta_q.w])
                angle_axis = rotation.as_rotvec()

                # 计算角速度
                angular_velocity = angle_axis / delta_t

                angular_velocities.append(angular_velocity)

            angular_velocities=np.array(angular_velocities)

            angular_velocities=np.vstack((sca1A[key.name][0:2,:-1],angular_velocities.T))
            ang_vel[key.name] = angular_velocities
        return ang_vel


    def get_vel(self,Ang_vel:dict):
        """
        get the SCA of velocity and the GNI of velocity
        """
        ang_vel=self.quaternion_to_angular_velocity()
        GNI1B=self.__GNI1B
        position=GNI1B[1:4]
        velocity=GNI1B[4:7]
        velocity=np.vstack((GNI1B[0],velocity))
        SCA_V={}
        for key in SCAID:
            time1=ang_vel[key.name][0]+ang_vel[key.name][1] * 1e-6
            w1 = ang_vel[key.name][2]
            w2 = ang_vel[key.name][3]
            w3 = ang_vel[key.name][4]
            time2=GNI1B[0]
            angular_velocity_w1= np.interp(time2, time1, w1)
            angular_velocity_w2 = np.interp(time2, time1, w2)
            angular_velocity_w3 = np.interp(time2, time1, w3)
            ang_vel[key.name]=np.vstack((time2,angular_velocity_w1,angular_velocity_w2,angular_velocity_w3))

            for i in range(len(GNI1B[0])):
                if i==0:
                    R=np.zeros((3,3))
                    w=np.zeros((3,1))
                    R[0,1]=float(position[2,i])
                    R[0,2]=-float(position[1,i])
                    R[1,0]=-float(position[2,i])
                    R[1,2]=float(position[0,i])
                    R[2,0]=float(position[1,i])
                    R[2,1]=-float(position[0,i])
                    w[0,0]=ang_vel[key.name][1,i]
                    w[1, 0] = ang_vel[key.name][2, i]
                    w[2, 0] = ang_vel[key.name][3, i]
                    v=np.dot(R,w)
                    SCA_V[key.name]=v
                else:
                    R=np.zeros((3,3))
                    w=np.zeros((3,1))
                    R[0,1]=float(position[2,i])
                    R[0,2]=-float(position[1,i])
                    R[1,0]=-float(position[2,i])
                    R[1,2]=float(position[0,i])
                    R[2,0]=float(position[1,i])
                    R[2,1]=-float(position[0,i])
                    w[0,0]=ang_vel[key.name][1,i]
                    w[1, 0] = ang_vel[key.name][2, i]
                    w[2, 0] = ang_vel[key.name][3, i]
                    v=np.dot(R,w)
                    SCA_V[key.name]=np.hstack((SCA_V[key.name],v))

            SCA_V[key.name]=np.vstack((GNI1B[0],SCA_V[key.name]))

            pass


        return velocity,SCA_V


        pass



    def GNI_to_velocity(self):
        GNI1B=self.__GNI1B



        pass



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
    GNI1B = L1B.getGNI(sat)
    a = Demtwo(satID=sat, date=date, SCA1A=SCA1A, SCA1B=SCA1B, TIM1B=TIM1B, CLK1B=CLK1B, GNI1B=GNI1B)

    ang_vel=a.quaternion_to_angular_velocity()
    velocity,SCA_V=a.get_vel(ang_vel)
    Res={}
    mean_std={}
    ID='No_3'
    for key in SCAID:

        Res[key.name]=velocity[1:4,:]-SCA_V[key.name][1:4,:]
        Res[key.name]=np.vstack((velocity[0],Res[key.name]))
        mean_std[key.name]=[[np.mean(Res[key.name][1]),np.mean(Res[key.name][2]),np.mean(Res[key.name][3])],
                      [np.std(Res[key.name][1]),np.std(Res[key.name][2]),np.std(Res[key.name][3])]]



    plt.scatter(velocity[0], Res[ID][2], color='green', label=ID+'_y',s=0.5,alpha=0.5)
    plt.scatter(velocity[0], Res[ID][1], color='blue', label=ID + '_x', s=0.5)
    plt.scatter(velocity[0], Res[ID][3],color='red',label=ID+'_z',s=0.5,alpha=0.2)
    # plt.plot([velocity[0,0], mean_std[ID][0][0]], [velocity[0,-1], mean_std[ID][0][0]], color='blue', linewidth=2)
    # plt.plot([velocity[0, 0], mean_std[ID][0][ 1]], [velocity[0, -1], mean_std[ID][0][ 1]], color='green', linewidth=2)
    # plt.plot([velocity[0, 0], mean_std[ID][0][2]], [velocity[0, -1], mean_std[ID][0][2]], color='red', linewidth=2)



    plt.legend()


    plt.xlabel('time(s)')
    plt.ylabel('Res(m/s)')

    plt.show()




    pass