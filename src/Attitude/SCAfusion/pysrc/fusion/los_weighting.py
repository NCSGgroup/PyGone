import plotly.express

from pysrc.core.settings import SatID, SCAID, Mission
from pysrc.fusion.multi_sca import SCAweighting
import numpy as np
from scipy import signal
from tqdm import trange
import os
import quaternion
from pysrc.core.geometry import GeoMathKit
from enum import Enum
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
import copy
from pysrc.attitude import los as LOS
import datetime

import matplotlib.pyplot as plt
import scipy.spatial.transform as st




class LOSweight():
    def __init__(self,sat: SatID, date,SCA1A, TIM1B, CLK1B):
        self.sat = sat
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self._QSA=self.ConfigSCF2SRF()
        self._LOS=self.ConfigIF2LOS(date)


    def ConfigIF2LOS(self,date):

        file='../result/product/GRACE_FO/RL04/L1B/%s/IF2LOS_%s_HUST01_%s.npy'%(date,date,self.sat.name)
        LOS=np.load(file)

        return LOS


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

        return QSA

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
            # print(valid)
            xp = xp[valid]
            yp = yp[valid]
            # print(xp)
            V = np.interp(x=x, xp=xp, fp=yp)

            V_int = np.trunc(V) + tim[0][0]
            V_frac = V - np.trunc(V)
            scaValue[0] = V_int
            scaValue[1] = V_frac * 1e6
            sca[key.name] = scaValue

        '''Step 2: RCV ---> GPS'''
        clk = self.__CLK1B
        # make a judgement on the quality
        index = np.where(clk[2] > 0)[0] #返回的是数据位置信息
        #  this should be refined by dividing the arcs
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


            scaValue[0] += V_int
            scaValue[1] += V_frac * 1e6

            scaValue[0][scaValue[1] < 0] -= 1
            scaValue[1][scaValue[1] < 0] += 1e6

            sca[key.name] = scaValue
            pass


        '''Step 3: Flip signs'''
        for id in SCAID:
            '''judge by the vector of quaternion'''
            a = sca[id.name][2:6, :].T
            quality = sca[id.name][6, :]

            m = [a[0, :]]

            for i in range(len(a[:, 0]) - 1):
                if np.dot(a[i + 1, 1:], m[-1][1:]) < 0 and quality[i] <= 6:
                    m.append(-a[i + 1, :])
                else:
                    m.append(a[i + 1, :])

            a = np.array(m)

            sca[id.name][2:6, :] = a.T


        '''Step 4: Unified time series'''
        SCA_scf={}
        for key in SCAID:
            SCA1A=sca[key.name][0:,sca[key.name][6,:]<=6]
            starttime = SCA1A[0][0]
            time1 = SCA1A[0] - starttime + SCA1A[1] * 1e-6
            time2=self._LOS[:,0].T-starttime
            qa=quaternion.as_quat_array(SCA1A[2:6].T)
            new = quaternion.squad(R_in=qa, t_in=time1, t_out=time2)
            #The first data of interpolation has a significant error, which needs to be noted
            qc = quaternion.as_float_array(new).T
            SCA_scf[key.name]=np.vstack((time2+starttime,qc))

            pass

        return SCA_scf


    def getSCA_LOS(self):
        QSA=self._QSA

        LOS=self._LOS
        SCA_LOS={}
        for key in SCAID:
            #Convert QSA data from JPL format to Hamilton format
            # qsa=quaternion.as_float_array(QSA[self.sat][key.value-1])
            # qsa[1:]=-qsa[1:]

            R_T=quaternion.as_rotation_matrix(QSA[self.sat][key.value-1]).T

            print('Calculating the %s_%s_%s camera from LOS frame'%(self.date,self.sat.name,key.name))

            for i in trange(len(LOS[:,0])):

                res=LOS[i,1:].reshape(3,3).T @ R_T

                if i==0:
                    sca=quaternion.as_float_array(quaternion.from_rotation_matrix(res))
                else:
                    SCA=quaternion.as_float_array(quaternion.from_rotation_matrix(res))
                    sca=np.vstack((sca,SCA))

            # print(len(LOS[:,0]),np.shape(sca))

            SCA_LOS[key.name]=np.vstack((LOS[:,0],sca.T))

        return SCA_LOS

        pass


    @staticmethod
    def getAngle(satID=SatID.C,date='2020-04-01',Dir='E:/data/GRACE_FO/RL04'):
        L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir(Dir+'/L1A')
        L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir(Dir+'/L1B')
        SCA1A = L1A.getSCA(satID)
        TIM1B=L1B.getTIM(satID)
        CLK1B=L1B.getCLK(satID)

        a=LOSweight(sat=satID,date=date,SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

        SCA_scf=a.TimeTagTransformation()
        SCA_LOS=a.getSCA_LOS()
        angle={}
        for key in SCAID:
            '''sign flip'''
            if np.dot(SCA_scf[key.name][1:,5], SCA_LOS[key.name][1:,5]) < 0:
                SCA_LOS[key.name][1:, :]=-SCA_LOS[key.name][1:, :]
            else:
                pass


            qa=quaternion.as_quat_array(SCA_scf[key.name][1:,:].T)
            qb=quaternion.as_quat_array(SCA_LOS[key.name][1:,:].T)
            qb_inv=np.array([quat.inverse() for quat in qb])
            res=qb_inv*qa
            angle[key.name]=np.vstack((SCA_LOS['No_1'][0],quaternion.as_float_array(res).T[1:,:]*2*3600*180/np.pi))


        return angle

        pass



def EulerFile(satID=SatID.C,begindate='2019-10-02',enddate='2019-01-02'):
    """
        saving Euler date:../Produce
    """
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    for day in daylist:
        # try:
            day = day.strftime('%Y-%m-%d')
            date = day
            sat = satID
            los = LOS.LOS(mission=Mission.GRACE_FO, date=day, sat=satID).configDir('E:/data/GRACE_FO/RL04/L1B')
            los.getLOS('../result/product/GRACE_FO/RL04/L1B/')
            dir='E:/data/GRACE_FO/RL04'
            Euler=LOSweight.getAngle(satID=sat,date=date,Dir=dir)
            file_name = day+'-'+sat.name+'.csv'
            fname=datetime.datetime.strptime(day, '%Y-%m-%d')
            fname=fname.strftime('%Y-%m')
            path=os.path.join('../Produce',fname)
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, file_name)
            #input the Euler Date
            if os.path.exists(file_path):
                print("%s文件存在"%(file_path))
            else:
                for key in SCAID:
                    #Notice:we replaced the first of epoch of data with the STD of data because of interpolation error
                    suffix_array = np.full((1, len(Euler[key.name][0])), key.name, dtype=object)
                    # Euler[key.name][0, 0]=0
                    Euler[key.name][1, 0]=np.std(Euler[key.name][1, 1:])
                    Euler[key.name][2, 0] = np.std(Euler[key.name][2, 1:])
                    Euler[key.name][3, 0] = np.std(Euler[key.name][3, 1:])
                    merged_euler = np.vstack((Euler[key.name], suffix_array))
                    merged_euler = merged_euler.T
                    with open(file_path, 'a') as f:
                        np.savetxt(f, merged_euler, fmt='%s', delimiter=' ', newline='\n')

                print("Saving %s-%s Euler Date completed"%(day, sat.name))
        # except:
            print('Missing data on %s'%day)
            continue




if __name__ == '__main__':
    # plotdemo()
    EulerFile(satID=SatID.C, begindate='2020-04-02', enddate='2020-04-02')

    pass



