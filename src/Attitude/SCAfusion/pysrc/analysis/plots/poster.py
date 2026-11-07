import matplotlib.dates as mdates
import numpy as np
import os
from pysrc.core.geometry import GeoMathKit
from pysrc.core.settings import SatID, SCAID, Mission
from enum import Enum
import datetime
from scipy import signal
from pysrc.fusion.multi_sca import SCAweighting
from tqdm import trange
import os
import quaternion
import copy
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
import matplotlib.pyplot as plt
from pysrc.attitude import los as LOS

from math import atan2, asin

class poster():
    def __init__(self,sat: SatID, date,SCA1A, TIM1B, CLK1B):
        self.sat = sat
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self._QSA=self.ConfigSCF2SRF()
        self._LOS=self.ConfigIF2LOS(date)



    @staticmethod
    def ConfigIF2LOS(date):

        file='../result/product/GRACE_FO/RL04/L1B/%s/IF2LOS_%s_HUST01.npy'%(date,date)
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
            qual=np.ones((1,len(qc[0])))
            SCA_scf[key.name]=np.vstack((time2+starttime,qc,qual))

            pass

        return SCA_scf



    def SCA_SCF2SRF(self,SCA):
        """
        SCF--->SRF
        """
        data = SCA.copy()
        satID=self.sat
        for key in SCAID:
            QSA=poster.ConfigSCF2SRF()
            qb = quaternion.as_quat_array(SCA[key.name][1:5, :].T)
            if key.name=='No_1':
                qa = qb * QSA[satID][0]
            elif key.name=='No_2':
                qa = qb * QSA[satID][1]
            elif key.name=='No_3':
                qa = qb * QSA[satID][2]
            else:
                pass

            qc = quaternion.as_float_array(qa).T
            data[key.name][1:5,:] = -qc

        return data


    @staticmethod
    def calculate_std_with_outliers(data):
        """
        After removing outliers, calculate the standard deviation
        """
        res1 = data
        res2 = res1[:, np.abs(res1[-1]) < 3 * np.std(res1[-1])]
        while len(res1[0]) != len(res2[0]):
            res1 = res2[:, np.abs(res2[-1]) < 3 * np.std(res2[-1])]
            res2 = res1[:, np.abs(res1[-1]) < 3 * np.std(res1[-1])]

        res_std = np.std(res2[-1])



        return res2,res_std



    def q_combine(self,A):

        sca_C=self.TimeTagTransformation()

        sca_R = self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))

        QSA = self._QSA
        a=SCAweighting(sca_R,QSA[self.sat],A)
        sca_new=a.combine_1t()
        sca_new=np.array(sca_new)

        return sca_new


    def Get_angle(self,A=np.array([[3,3,30],[3,3,30],[3,3,30]])):
        LOS=self._LOS
        sca_new=self.q_combine(A)
        Res=[]

        for i in trange(len(LOS[:, 0])):
            q=quaternion.as_quat_array(sca_new[:,1:5])
            R_T=quaternion.as_rotation_matrix(q[i]).T
            res = LOS[i, 1:].reshape(3, 3).T @ R_T
            a=quaternion.from_rotation_matrix(res)
            Res.append(a)

        Res=quaternion.as_float_array(Res)
        angle=np.vstack((LOS[np.abs(Res[:,0].T)>0.99,0].T,Res[np.abs(Res[:,0].T)>0.99,1:4].T*3600*2*180/np.pi))


        return angle


    def get_angle(self,A=np.array([[3,3,30],[3,3,30],[3,3,30]])):

        sca_new=self.q_combine(A)

        roll1=[]
        pitch1=[]
        yaw1=[]


        for i in trange(len(sca_new[:, 0])):
            q=quaternion.as_quat_array(sca_new[:,1:5])
            R=quaternion.as_rotation_matrix(q[i])
            roll1.append(atan2(R[2][1], R[2][2]))
            pitch1.append(-asin(R[2][0]))
            yaw1.append( atan2(R[1][0], R[0][0]))



        Res1 = np.vstack((sca_new[:,0].T,roll1,pitch1,yaw1))



        return Res1





    def plotPSD(self,data1,data2):


        time1 = data1[0]

        # angle1=data1[1]

        d1,std1=self.calculate_std_with_outliers(data1)
        angle1=np.interp(time1,d1[0],d1[1])


        # wn = 2 * 0.02 / 1
        # order = 4  # filter order
        # b, a = signal.butter(order, wn, btype='high', fs=1)
        # angle1 = signal.lfilter(b, a, angle1)

        time2 = data2[0]

        # angle2=data2[1]

        d2,std2=self.calculate_std_with_outliers(data2)
        angle2=np.interp(time2,d2[0],d2[1])

        # wn = 2 * 0.02 / 1
        # order = 4  # filter order
        # b, a = signal.butter(order, wn, btype='high', fs=1)
        # angle2 = signal.lfilter(b, a, angle2)

        fig = plt.figure()
        fig2 = plt.figure()

        axx = fig2.add_subplot(1, 1, 1)

        psd1, freq1 = plt.psd(angle1, Fs=1, NFFT=len(angle1))
        psd2, freq2 = plt.psd(angle2, Fs=1, NFFT=len(angle2))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')

        psd1 = np.sqrt(2 * psd1)
        psd2 = np.sqrt(2 * psd2)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.plot(freq1, psd1, label='OLS', lw=0.5,
                alpha=0.8)
        ax.plot(freq2, psd2, label='ILS', lw=0.5,
                alpha=0.9)

        ax.legend(ncol=3)
        ax.grid(True, linestyle='dashed')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')

        fig.align_ylabels()
        fig.align_xlabels()

        plt.show()

        pass


    def PlotEuler(self,res1,res2):
        # res1x,res1x_std=self.calculate_std_with_outliers(res1[[0,1],:])
        # res1y,res1y_std=self.calculate_std_with_outliers(res1[[0,2],:])
        # res1z,res1z_std=self.calculate_std_with_outliers(res1[[0,3],:])
        #
        # print([res1x_std,res1y_std,res1z_std])
        #
        # res2x,res2x_std=self.calculate_std_with_outliers(res2[[0,1],:])
        # res2y,res2y_std=self.calculate_std_with_outliers(res2[[0,2],:])
        # res2z,res2z_std=self.calculate_std_with_outliers(res2[[0,3],:])
        #
        # print([res2x_std, res2y_std, res2z_std])

        res1x=res1[[0,1],:]
        res1y=res1[[0,2],:]
        res1z=res1[[0,3],:]


        res2x=res2[[0,1],:]
        res2y=res2[[0,2],:]
        res2z=res2[[0,3],:]


        seconds_x1=res1x[0,:]
        x1=res1x[1,:]
        seconds_x2=res2x[0,:]
        x2=res2x[1,:]

        label_x1='roll-pre'
        label_x2 = 'roll-aft'

        seconds_y1=res1y[0,:]
        y1=res1y[1,:]
        seconds_y2=res2y[0,:]
        y2=res2y[1,:]
        label_y1='pitch-pre'
        label_y2 = 'pitch-aft'

        seconds_z1=res1z[0,:]
        z1=res1z[1,:]
        seconds_z2=res2z[0,:]
        z2=res2z[1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z1='yaw-pre'
        label_z2 = 'yaw-aft'

        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time_x1 = []
        time_y1 = []
        time_z1 = []
        time_x2 = []
        time_y2 = []
        time_z2 = []
        for sec_x in seconds_x1:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x1.append(dt)

        for sec_y in seconds_y1:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y1.append(dt)

        for sec_z in seconds_z1:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z1.append(dt)

        for sec_x in seconds_x2:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x2.append(dt)

        for sec_y in seconds_y2:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y2.append(dt)

        for sec_z in seconds_z2:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z2.append(dt)

        fig, ax = plt.subplots()

        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)
        ax.scatter(time_x1, x1, s=0.1, color='red', label=label_x1, alpha=0.5)
        ax.scatter(time_x2, x2, s=0.1, color='blue', label=label_x2, alpha=0.5)

        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        plt.ylim(-5000, 5000)
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.grid(True, axis='y', linestyle='dashed')
        plt.legend()
        plt.show()


        fig, ax = plt.subplots()

        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)
        ax.scatter(time_y1, y1, s=0.1, color='red', label=label_y1, alpha=0.5)
        ax.scatter(time_y2, y2, s=0.1, color='blue', label=label_y2, alpha=0.5)

        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        plt.ylim(-5000, 5000)
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.grid(True, axis='y', linestyle='dashed')
        plt.legend()


        fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)
        ax.scatter(time_z1, z1, s=0.1, color='red', label=label_z1, alpha=0.5)
        ax.scatter(time_z2, z2, s=0.1, color='blue', label=label_z2, alpha=0.5)

        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        plt.ylim(-5000, 5000)
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.grid(True, axis='y', linestyle='dashed')
        plt.legend()






        plt.show()
        pass



if __name__ == '__main__':
    day='2020-02-12'
    date = day
    sat=SatID.C

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    sca1B = L1B.getSCA(sat)
    a = poster(sat=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

    r1=np.array([[3,3,30],[3,3,30],[3,3,30]])
    sca_new1 = a.q_combine(r1)
    res1=a.Get_angle(r1)
    r2=np.array([[2.52,1.42,25.19],[2.85,2.09,29.72],[2.18,1.89,21.67]])
    sca_new2 = a.q_combine(r2)
    res2=a.Get_angle(r2)

    data1 = res1[[0,1]]
    data2 = res2[[0,1]]


    # r1 = [3, 3, 30]
    # r2 = [5, 20, 50]
    #
    # res1=a.get_angle(r1)
    # res2=a.get_angle(r2)
    #
    # data1=res1[0:2]
    # data2 = res2[0:2]
    a.PlotEuler(res1,res2)
    # a.plotPSD(res1[[0,1]],res2[[0,1]])
    # a.plotPSD(res1[[0, 2]], res2[[0, 2]])
    # a.plotPSD(res1[[0, 3]], res2[[0, 3]])



    pass