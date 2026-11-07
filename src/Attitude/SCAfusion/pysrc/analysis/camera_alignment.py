import numpy as np
from pysrc.core.geometry import GeoMathKit
from pysrc.core.settings import SatID, SCAID, Mission
import os
import datetime
from pysrc.analysis import hfn_analysis as HFNAnalysis
from pysrc.fusion.multi_sca import SCAweighting
from tqdm import trange
import quaternion
import copy
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B

class CameraAlignment():
    """
    directly compare the alignment of two cameras over the course of a day
    """
    def __init__(self,sat:SatID,date,SCA1A,TIM1B,CLK1B):
        self.sat = sat
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self._QSA=self.ConfigSCF2SRF()


        pass


    @staticmethod
    def ConfigSCF2SRF():
        QSA={}
        file='../data/GRACE_FO/Auxiliary/QSA1B_2018-05-22_C_04.txt'
        skip=GeoMathKit.ReadEndOfHeader(file_name=file,end_flag='End of YAML header')
        q=np.loadtxt(file,skiprows=skip,usecols=[3,4,5,6])
        QSA[SatID.C]=quaternion.as_quat_array(q)

        file='../data/GRACE_FO/Auxiliary/QSA1B_2018-05-22_D_04.txt'
        skip=GeoMathKit.ReadEndOfHeader(file_name=file,end_flag='End of YAML header')
        q=np.loadtxt(file,skiprows=skip,usecols=[3,4,5,6])
        QSA[SatID.D]=quaternion.as_quat_array(q)

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


        return sca




    def SCA_SCF2SRF(self,SCA):
        """
        SCF--->SRF
        """

        satID=self.sat
        for key in SCAID:
            QSA=self.ConfigSCF2SRF()
            qb = quaternion.as_quat_array(SCA[key.name][2:6, :].T)
            if key.name=='No_1':
                qa = qb * QSA[satID][0]
            elif key.name=='No_2':
                qa = qb * QSA[satID][1]
            elif key.name=='No_3':
                qa = qb * QSA[satID][2]
            else:
                pass

            qc = quaternion.as_float_array(qa).T
            SCA[key.name][2:6,:] = -qc

        return SCA

    def CA12(self):
        sca=self.TimeTagTransformation()
        SCA1A=self.SCA_SCF2SRF(sca)
        sca1=SCA1A[SCAID.No_1.name]
        sca2=SCA1A[SCAID.No_2.name]
        quality=np.maximum(sca1[6], sca2[6])

        q1=quaternion.as_quat_array(sca1[2:6].T)
        q2=quaternion.as_quat_array(sca2[2:6].T)
        num_quaternions = q1.size
        for i in range(num_quaternions):
            q1[i] = q1[i].inverse()

        q = q1 * q2
        detal = quaternion.as_float_array(q)
        # 化角秒
        euler = detal[:, 1:] * 2 * 3600 * 180/np.pi

        Euler=np.vstack((sca1[0:2,:],euler.T,quality))

        fileDir = os.path.join('../CA','%s/%s/' % (self.date.strftime('%Y-%m'),self.date.strftime('%Y-%m-%d')))
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'CA12_%s_%s.npy' % (self.date.strftime('%Y-%m-%d'),self.sat.name)
        np.save(str(fileDir) + filename, Euler)

        pass

    def CA13(self):
        sca = self.TimeTagTransformation()
        SCA1A=self.SCA_SCF2SRF(sca)
        sca1=SCA1A[SCAID.No_1.name]
        sca2=SCA1A[SCAID.No_3.name]
        quality=np.maximum(sca1[6], sca2[6])

        q1 = quaternion.as_quat_array(sca1[2:6].T)
        q2 = quaternion.as_quat_array(sca2[2:6].T)
        num_quaternions = q1.size
        for i in range(num_quaternions):
            q1[i] = q1[i].inverse()

        q = q1 * q2
        detal = quaternion.as_float_array(q)
        # 化角秒
        euler = detal[:, 1:] * 2 * 3600 * 180 / np.pi

        Euler = np.vstack((sca1[0:2, :], euler.T, quality))

        fileDir = os.path.join('../CA', '%s/%s/' % (self.date.strftime('%Y-%m'),self.date.strftime('%Y-%m-%d')))
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'CA13_%s_%s.npy' % (self.date.strftime('%Y-%m-%d'),self.sat.name)
        np.save(str(fileDir) + filename, Euler)

        pass


    def CA23(self):
        sca = self.TimeTagTransformation()
        SCA1A=self.SCA_SCF2SRF(copy.deepcopy(sca))
        sca1=SCA1A[SCAID.No_2.name]
        sca2=SCA1A[SCAID.No_3.name]
        quality=np.maximum(sca1[6], sca2[6])

        q1 = quaternion.as_quat_array(sca1[2:6].T)
        q2 = quaternion.as_quat_array(sca2[2:6].T)
        num_quaternions = q1.size
        for i in range(num_quaternions):
            q1[i] = q1[i].inverse()

        q = q1 * q2
        detal = quaternion.as_float_array(q)
        # 化角秒
        euler = detal[:, 1:] * 2 * 3600 * 180 / np.pi

        Euler = np.vstack((sca1[0:2, :], euler.T, quality))

        fileDir = os.path.join('../CA','%s/%s/' % (self.date.strftime('%Y-%m'),self.date.strftime('%Y-%m-%d')))
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'CA23_%s_%s.npy' % (self.date.strftime('%Y-%m-%d'),self.sat.name)
        np.save(str(fileDir) + filename, Euler)

        pass


def Eulerfile(begindate='2020-02-01',enddate='2020-02-28',satID=SatID.C):
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    Dir = 'E:/data/GRACE_FO/RL04'

    for day in daylist:
        try:
            date=day.strftime('%Y-%m-%d')
            L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir(Dir+'/L1A')
            L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir(Dir+'/L1B')
            SCA1A = L1A.getSCA(satID)
            TIM1B=L1B.getTIM(satID)
            CLK1B=L1B.getCLK(satID)

            a=CameraAlignment(sat=satID,date=day,SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)
            a.CA12()
            a.CA13()
            a.CA23()
            print('the data of Camera Alignment has be completed on %s'%date)
        except:
            print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))

    pass

if __name__ == '__main__':

    Eulerfile(begindate='2020-01-30',enddate='2020-12-31',satID=SatID.D)

    pass






