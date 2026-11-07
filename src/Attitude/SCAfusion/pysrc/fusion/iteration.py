from pysrc.core.settings import SatID, SCAID, Mission
from pysrc.fusion.multi_sca import SCAweighting
import numpy as np
import os
import quaternion
from pysrc.core.geometry import GeoMathKit
from enum import Enum
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
import copy
from scipy import signal

import datetime
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import signal
from pysrc.analysis.hfn_analysis import configReadEuler


class Iteration():
    def __init__(self,satID: SatID, date, SCA1A, TIM1B, CLK1B,Lamnda=None):
        self.sat = satID
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self.sca_C = None
        self.__Lamnda=Lamnda
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

        Iteration.SCF2SRF = QSA

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

    # @staticmethod
    # def interpolation(sca):
    #
    #     '''Unified time series'''
    #     SCA_scf={}
    #     for key in SCAID:
    #         SCA1A=sca[key.name][0:,sca[key.name][6,:]<=6]
    #         starttime = SCA1A[0][0]
    #         time1 = SCA1A[0] - starttime + SCA1A[1] * 1e-6
    #         time2=self._LOS[:,0].T-starttime
    #         qa=quaternion.as_quat_array(SCA1A[2:6].T)
    #         new = quaternion.squad(R_in=qa, t_in=time1, t_out=time2)
    #         #The first data of interpolation has a significant error, which needs to be noted
    #         qc = quaternion.as_float_array(new).T
    #         SCA_scf[key.name]=np.vstack((time2+starttime,qc))
    #
    #         pass
    #
    #     return SCA_scf


    def SCA_SCF2SRF(self,SCA):
        """
        SCF--->SRF
        """
        data = SCA.copy()
        satID=self.sat
        for key in SCAID:
            QSA=Iteration.ConfigSCF2SRF()
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
            data[key.name][2:6,:] = -qc

        return data



    @staticmethod
    def calculate_std_with_outliers(data):
        """
        After removing outliers, calculate the standard deviation
        """
        res1 = data
        res2 = res1[:, np.abs(res1[-1]) < 3 * np.std(res1[-1])]
        while len(res1[0]) != len(res2[0]):
            res1 = res2[:, np.abs(res2[1]) < 3 * np.std(res2[1])]
            res2 = res1[:, np.abs(res1[1]) < 3 * np.std(res1[1])]

        res_std = np.std(res2[-1])



        return res2,res_std


    def getweighting(self,date='2020-02-20'):
        b=3
        c=3
        d=30
        A=np.array([[b,c,d],[b,c,d],[b,c,d]])
        sca_C=self.TimeTagTransformation()



        # sca_R=self.SCA_SCF2SRF(SCA=sca_temp.copy())
        sca_R=self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))

        sca={}
        time={}
        # Resx, Resy, Resz, Res_std, days = configReadEuler(begindate=date, enddate=date)


        for key in SCAID:

            sca[key.name]=np.vstack((sca_R[key.name][0:6],sca_R[key.name][6]))
            time[key.name]=sca_R[key.name][0:2]
        QSA = self._QSA
        a=SCAweighting(sca,QSA[self.sat],A)
        sca_new=a.combine_2t()
        sca_new=np.array(sca_new)
        delta={}
        q_new={}
        angle={}
        x={}
        y={}
        z={}
        for key in SCAID:
            q=quaternion.as_quat_array(sca_new[:,2:6])
            QSA_inv=QSA[self.sat][int(key.name[-1])-1].inverse()
            q_new[key.name]=q*QSA_inv
            ppp=quaternion.as_float_array(q_new[key.name]).T


            for i in range(len(q_new[key.name])):
                q_new[key.name][i]=q_new[key.name][i].inverse()

            de=q_new[key.name]*quaternion.as_quat_array(sca_C[key.name][2:6,:].T)
            delta[key.name]=quaternion.as_float_array(de)
            del_index=np.where(np.abs(delta[key.name][:,0].T)>0.99)[0]
            angle[key.name]=np.vstack((time[key.name],delta[key.name][:,1:4].T*3600*2*180/np.pi))
            angle[key.name]=angle[key.name][:,del_index]
            if key.name=='No_1':
                x[key.name]=np.vstack((angle[key.name][0:2],angle[key.name][2]))
                y[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][3]))
                z[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][4]))
                x[key.name],aa = self.calculate_std_with_outliers(x[key.name])
                y[key.name],bb = self.calculate_std_with_outliers(y[key.name])
                z[key.name],cc = self.calculate_std_with_outliers(z[key.name])

                x_std=[aa]
                y_std=[bb]
                z_std=[cc]
            else:

                x[key.name]=np.vstack((angle[key.name][0:2],angle[key.name][2]))
                y[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][3]))
                z[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][4]))
                x[key.name],aa = self.calculate_std_with_outliers(x[key.name])
                y[key.name],bb = self.calculate_std_with_outliers(y[key.name])
                z[key.name],cc = self.calculate_std_with_outliers(z[key.name])

                x_std.append(aa)
                y_std.append(bb)
                z_std.append(cc)

            pass

        New_Lamnda=np.vstack((x_std,y_std,z_std))
        print(New_Lamnda)
        # Lamnda=np.full((3, 3), [float(b), float(c), float(d)]).T

        std1_weight = 1 / (New_Lamnda[0, 1]) * b
        std2_weight = 1 / (New_Lamnda[1, 1]) * c
        std3_weight = 1 / (New_Lamnda[2, 1]) * d
        print(std1_weight / std2_weight, std3_weight / std2_weight)
        iteration_i=0

        xdata1=x['No_1']

        # xtime=x['No_1'][0]+x['No_1'][1] * 1e-6-x['No_1'][0,0]
        # intertime=np.arange(0,np.max(x['No_1'][0]-x['No_1'][0,0]))
        # x_interp=np.interp(intertime,xtime,x['No_1'][2])
        # x_interp=np.array(x_interp)
        #
        # fig = plt.figure()
        # fig2 = plt.figure()
        #
        # axx = fig2.add_subplot(1, 1, 1)
        #
        # psd, freq = plt.psd(x_interp, Fs=1, NFFT=len(x_interp))
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('PSD')
        #
        # psd = np.sqrt(2 * psd)
        # ax = fig.add_subplot(1, 1, 1)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        #
        # ax.plot(freq, psd, label='2020-02-20_resNo_1_x', lw=0.5,
        #         alpha=0.8)
        #
        # ax.legend(ncol=3)
        # ax.grid(True, linestyle='dashed')
        # ax.set_xlabel('Frequency [Hz]')
        # ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
        #
        # fig.align_ylabels()
        # fig.align_xlabels()
        #
        # plt.show()










        while abs(std1_weight/std2_weight-1)>0.01 or abs(std1_weight/std3_weight-1)>0.01 :
            Lamnda=New_Lamnda
            iteration_i=iteration_i+1
            print('The %s-th iteration'%iteration_i)
            o = Lamnda[0,1]
            p = Lamnda[1,1]
            s = Lamnda[2,1]

            sca_C = self.TimeTagTransformation()

            sca_R = self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))

            sca = {}
            time = {}
            # Resx, Resy, Resz, Res_std, days = configReadEuler(begindate=date, enddate=date)

            for key in SCAID:
                sca[key.name] = np.vstack((sca_R[key.name][0:6], sca_R[key.name][6]))
                time[key.name] = sca_R[key.name][0:2]
            QSA = self._QSA
            a = SCAweighting(sca, QSA[self.sat], o, p, s)
            sca_new = a.combine_2t()
            sca_new = np.array(sca_new)
            delta = {}
            q_new = {}
            angle = {}
            x = {}
            y = {}
            z = {}
            for key in SCAID:
                q = quaternion.as_quat_array(sca_new[:, 2:6])
                QSA_inv = QSA[self.sat][int(key.name[-1]) - 1].inverse()
                q_new[key.name] = q * QSA_inv
                ppp = quaternion.as_float_array(q_new[key.name]).T

                for i in range(len(q_new[key.name])):
                    q_new[key.name][i] = q_new[key.name][i].inverse()

                de = q_new[key.name] * quaternion.as_quat_array(sca_C[key.name][2:6, :].T)
                delta[key.name] = quaternion.as_float_array(de)
                del_index = np.where(np.abs(delta[key.name][:, 0].T) > 0.99)[0]
                angle[key.name] = np.vstack((time[key.name], delta[key.name][:, 1:4].T * 3600 * 2 * 180 / np.pi))
                angle[key.name] = angle[key.name][:, del_index]
                if key.name == 'No_1':
                    x[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][2]))
                    y[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][3]))
                    z[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][4]))
                    x[key.name], aa = self.calculate_std_with_outliers(x[key.name])
                    y[key.name], bb = self.calculate_std_with_outliers(y[key.name])
                    z[key.name], cc = self.calculate_std_with_outliers(z[key.name])

                    x_std = [aa]
                    y_std = [bb]
                    z_std = [cc]
                else:

                    x[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][2]))
                    y[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][3]))
                    z[key.name] = np.vstack((angle[key.name][0:2], angle[key.name][4]))
                    x[key.name], aa = self.calculate_std_with_outliers(x[key.name])
                    y[key.name], bb = self.calculate_std_with_outliers(y[key.name])
                    z[key.name], cc = self.calculate_std_with_outliers(z[key.name])

                    x_std.append(aa)
                    y_std.append(bb)
                    z_std.append(cc)

                pass

            New_Lamnda = np.vstack((x_std, y_std, z_std))
            print(New_Lamnda)

            std1_weight = 1/(New_Lamnda[0, 1]) * Lamnda[0,1]
            std2_weight = 1/(New_Lamnda[1, 1]) * Lamnda[1,1]
            std3_weight = 1/(New_Lamnda[2, 1]) * Lamnda[2,1]
            print(std1_weight/std2_weight,std3_weight/std2_weight)

            # if iteration_i==8:
            #
            #     ILS_sca=sca_new
            #
            #     break
                # xtime = x['No_1'][0] + x['No_1'][1] * 1e-6 - x['No_1'][0, 0]
                # intertime = np.arange(0, np.max(x['No_1'][0] - x['No_1'][0, 0]))
                # x_interp = np.interp(intertime, xtime, x['No_1'][2])
                # x_interp = np.array(x_interp)
                #
                # psd1, freq1 = axx.psd(x_interp, Fs=1, NFFT=len(x_interp))
                # plt.xlabel('Frequency (Hz)')
                # plt.ylabel('PSD')
                # psd1 = np.sqrt(2 * psd1)
                #
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                #
                # ax.plot(freq1, psd1, label='2020-02-18_resNo_1_x', lw=0.5,
                #         alpha=0.8)
                #
                # ax.legend(ncol=3)
                # ax.grid(True, linestyle='dashed')
                # ax.set_xlabel('Frequency [Hz]')
                # ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
                #
                # fig.align_ylabels()
                # fig.align_xlabels()
                #
                # plt.show()








        data_x=np.array([angle['No_2'][0],angle['No_2'][1],angle['No_2'][2]])
        data_x_out=data_x[0:3,np.abs(data_x[2,:])<100]
        T_x=data_x_out[0]+data_x_out[1]* 10**(-6)-data_x_out[0,0]
        xx=data_x_out[2]
        print(len(xx))

        data_y=np.array([angle['No_2'][0],angle['No_2'][1],angle['No_2'][3]])
        data_y_out=data_y[0:3,np.abs(data_y[2,:])<100]
        T_y=data_y_out[0]+data_y_out[1]* 10**(-6)-data_y_out[0,0]
        yy=data_y_out[2]
        print(len(yy))

        data_z=np.array([angle['No_2'][0],angle['No_2'][1],angle['No_2'][4]])
        data_z_out=data_z[0:3,np.abs(data_z[2,:])<100]
        T_z=data_z_out[0]+data_z_out[1]* 10**(-6)-data_z_out[0,0]
        zz=data_z_out[2]
        print(len(zz))

        plt.scatter(T_x,xx, color='red', label='C_x',s=0.5,alpha=0.5)
        plt.scatter(T_y, yy, color='blue', label='C_y', s=0.5)
        plt.scatter(T_z, zz, color='green', label='C_z', s=0.5, alpha=0.5)

        plt.legend()

        plt.xlabel('time(s)')
        plt.ylabel('Res(arcsec)')

        plt.show()

        plt.show()



        return sca_new,self.__Lamnda


if __name__ == '__main__':
    day='2019-01-24'
    date = day
    sat=SatID.C

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    sca1B = L1B.getSCA(sat)
    a = Iteration(satID=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

    sca_new,l=a.getweighting()

    # from scipy import signal
    #
    # xtime1 = xdata1[0] + xdata1[1] * 1e-6 - xdata1[0, 0]
    # intertime1 = np.arange(0, np.max(xdata1[0] - xdata1[0, 0]))
    # x_interp1 = np.interp(intertime1, xtime1, xdata1[2])
    # x_interp1 = np.array(x_interp1)
    # wn = 2 * 0.02 / 1
    # order = 4  # filter order
    # b, a = signal.butter(order, wn, btype='high', fs=1)
    # x_interp1 = signal.lfilter(b, a, x_interp1)
    #
    # xtime2 = xdata2[0] + xdata2[1] * 1e-6 - xdata2[0, 0]
    # intertime2 = np.arange(0, np.max(xdata2[0] - xdata2[0, 0]))
    # x_interp2 = np.interp(intertime2, xtime2, xdata2[2])
    # x_interp2 = np.array(x_interp2)
    # wn = 2 * 0.02 / 1
    # order = 4  # filter order
    # b, a = signal.butter(order, wn, btype='high', fs=1)
    # x_interp2 = signal.lfilter(b, a, x_interp2)
    #
    # fig = plt.figure()
    # fig2 = plt.figure()
    #
    # axx = fig2.add_subplot(1, 1, 1)
    #
    # psd1, freq1 = plt.psd(x_interp1, Fs=1, NFFT=len(x_interp1))
    # psd2, freq2 = plt.psd(x_interp2, Fs=1, NFFT=len(x_interp2))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('PSD')
    #
    # psd1 = np.sqrt(2 * psd1)
    # psd2 = np.sqrt(2 * psd2)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    #
    # ax.plot(freq1, psd1, label='2020-02-18_noIteration_No_1_x', lw=0.5,
    #         alpha=0.8)
    # ax.plot(freq2, psd2, label='2020-02-18_Iteration_No_1_x', lw=0.5,
    #         alpha=0.9)
    #
    # ax.legend(ncol=3)
    # ax.grid(True, linestyle='dashed')
    # ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    #
    # fig.align_ylabels()
    # fig.align_xlabels()
    #
    # plt.show()

    pass