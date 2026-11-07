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
from matplotlib.offsetbox import AnchoredText

import matplotlib.pyplot as plt
import scipy.spatial.transform as st

class SCandGNSSpsd():
    def __init__(self,sat: SatID, date,SCA1A, TIM1B, CLK1B):
        self.sat = sat
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self._QSA=self.ConfigSCF2SRF()
        self._LOS=self.ConfigIF2LOS()


    def ConfigIF2LOS(self):

        file='../result/product/GRACE_FO/RL04/L1B/%s/IF2LOS_%s_HUST01_%s.npy'%(self.date,self.date,self.sat.name)
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
    def getattitude(satID=SatID.C,date='2020-04-01',Dir='G:/data/GRACE_FO/RL04'):
        L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir(Dir+'/L1A')
        L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir(Dir+'/L1B')
        SCA1A = L1A.getSCA(satID)
        TIM1B=L1B.getTIM(satID)
        CLK1B=L1B.getCLK(satID)

        a=SCandGNSSpsd(sat=satID,date=date,SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

        SCA_scf=a.TimeTagTransformation()
        SCA_LOS=a.getSCA_LOS()
        '''Unified positive and negative signs'''
        for id in SCAID:
            if np.dot(SCA_scf[id.name][1:,5],SCA_LOS[id.name][1:,5])<0:
                if SCA_scf[id.name][1,5]>0:
                    SCA_LOS[id.name][1:,:]=-SCA_LOS[id.name][1:,:]
                else:
                    SCA_scf[id.name][1:, :] = -SCA_scf[id.name][1:, :]
            else:
                pass

        return SCA_scf,SCA_LOS

        pass


def plot(satID=SatID.C,date='2020-04-01',Dir='G:/data/GRACE_FO/RL04'):
    q_sc,q_gnss=SCandGNSSpsd.getattitude(satID=SatID.C,date='2020-04-01',Dir='E:/data/GRACE_FO/RL04')

    # plt.style.use(['science', 'grid', 'ieee'])
    # plt.style.use(['ggplot', 'grid', 'seaborn'])
    fig = plt.figure()
    fig2 = plt.figure()
    axx = fig2.add_subplot(1, 1, 1)
    ax = fig.add_subplot(2, 1, 1)

    sc = q_sc[SCAID.No_1.name][-1, 1:]
    gnss = q_gnss[SCAID.No_1.name][-1, 1:]
    lab = 'SC1-x'
    lab1='diff(g-s)'

    psd1, freq1 = axx.psd(sc, NFFT=int(len(sc) / 2),
                          pad_to=len(sc), Fs=1, label='SC-3', lw=0.5, c='green')
    psd2, freq2 = axx.psd(gnss, NFFT=int(len(gnss) / 2),
                          pad_to=len(gnss), Fs=1, label='GNSS', lw=0.5, alpha=0.8, c='blue')
    ax.legend(loc=1)
    psd1 = np.sqrt(2 * psd1)
    psd2 = np.sqrt(2 * psd2)
    ax.plot(freq1, psd1, label=lab, lw=0.5, c='green')
    ax.plot(freq2, psd2, label='GNSS', lw=0.5, c='blue')
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.set_xscale('log', base=10)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(ncol=1)
    ax.grid(True, linestyle='dashed')
    # ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    # ax.set_yscale('log',base=10)
    # text = AnchoredText('(a)', loc=4, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    # ax.add_artist(text)
    fig.align_ylabels()

    ax = fig.add_subplot(2, 1, 2)
    psd = psd2 - psd1
    ax.plot(freq1, psd, label=lab1, lw=0.5, c='grey')
    ax.legend(loc=1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(ncol=1)
    ax.grid(True, linestyle='dashed')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    # text = AnchoredText('(b)', loc=4, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    # ax.add_artist(text)
    fig.align_ylabels()

    fig.savefig('../SCandGNSS/figure1.png', dpi=300)
    plt.show()

def plot2():
    q_sc,q_gnss=SCandGNSSpsd.getattitude(satID=SatID.C,date='2020-04-18',Dir='E:/data/GRACE_FO/RL04')
    fig = plt.figure(figsize=(8, 7))
    fig2 = plt.figure()
    axx = fig2.add_subplot(1, 1, 1)
    sc = q_sc[SCAID.No_2.name][-2, 1:]
    gnss = q_gnss[SCAID.No_2.name][-2, 1:]
    lab = 'SC2-y'
    lab1 = 'diff'
    psd1, freq1 = axx.psd(sc, NFFT=int(len(sc) / 2),
                          pad_to=len(sc), Fs=1, label='SC-3', lw=0.5, c='green')
    psd2, freq2 = axx.psd(gnss, NFFT=int(len(gnss) / 2),
                          pad_to=len(gnss), Fs=1, label='GNSS', lw=0.5, alpha=0.8, c='blue')

    psd1 = np.sqrt(2 * psd1)
    psd2 = np.sqrt(2 * psd2)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(freq1, psd1, label=lab, lw=0.5, c='green')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(ncol=1)
    ax.grid(True, linestyle='dashed')
    # ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    # ax.set_yscale('log',base=10)
    # text = AnchoredText('(a)', loc=4, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    # ax.add_artist(text)
    fig.align_ylabels()
    ax = fig.add_subplot(3, 1, 2)
    ax.plot(freq2, psd2, label='GNSS', lw=0.5, c='blue')
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.set_xscale('log', base=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(ncol=1)
    ax.grid(True, linestyle='dashed')
    # ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    # ax.set_yscale('log',base=10)
    # text = AnchoredText('(a)', loc=4, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    # ax.add_artist(text)
    fig.align_ylabels()
    ax = fig.add_subplot(3, 1, 3)
    psd = psd1 - psd2
    ax.plot(freq1, psd, label=lab1, lw=0.5, c='grey')
    ax.legend(loc=1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(None, 200)
    ax.legend(ncol=1)
    ax.grid(True, linestyle='dashed')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    fig.align_ylabels()
    fig.savefig('../SCandGNSS/figure2.png', dpi=300)
    plt.show()



if __name__=='__main__':
    plot2()
    pass

















