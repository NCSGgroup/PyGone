#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pysrc.core.settings import Mission, SatID,Coord,SCAID
import numpy as np
import datetime
import h5py
import sys
import time
from matplotlib.ticker import MultipleLocator
from pysrc.core.geometry import GeoMathKit
from pysrc.data.instruments import GetInstrument_L1B
from scipy.stats import linregress
from pysrc.analysis.hfn_analysis import configReadEuler, ConfigAnglefile, configReadEuler3
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from scipy import signal


class SunModel:
    def __init__(self, date, sat: SatID):
        self.__date = date
        self.__sat = sat
        self.__LOS = self.ConfigIF2LOS()
        self.__time = self.__LOS[:, 0].T
        pass



    @staticmethod
    def print_slowly(text):
        for char in text:
            if char == ' ':
                input()  # 等待用户按下回车键继续
            time.sleep(0.1)  # 调整延迟时间以控制打印速度

    def ConfigHFNangle(self,coord:Coord):
        Date = datetime.datetime.strptime(self.__date, '%Y-%m-%d')
        file='../HFN_Euler/%s/%s/HFNeuler-%s_%s_%s.h5'\
             %(Date.strftime('%Y-%m'),Date.strftime('%Y-%m-%d'),Date.strftime('%Y-%m-%d'),coord.name,self.__sat.name)
        data={}
        with h5py.File(file, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    # 对于组，创建一个嵌套字典
                    nested_dict = {}
                    for nested_key in f[key].keys():
                        nested_dict[nested_key] = f[key][nested_key][()]
                    data[key] = nested_dict
                else:
                    # 对于数据集，直接读取值
                    data[key] = f[key][()]


        return data
        pass


    def ConfigIF2LOS(self):

        file='../result/product/GRACE_FO/RL04/L1B/%s/IF2LOS_%s_HUST01_%s.npy'%(self.__date,self.__date,self.__sat.name)
        LOS=np.load(file)

        return LOS

    def ConfigsunAnglefile(self):
        Date = datetime.datetime.strptime(self.__date, '%Y-%m-%d')
        file = 'E:/data/SunAngle/%s/%s/Angle_Sun-Moon_%s_%s.h5' \
               % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d'), Date.strftime('%Y-%m-%d'), self.__sat.name)
        data = {}
        with h5py.File(file, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    # 对于组，创建一个嵌套字典
                    nested_dict = {}
                    for nested_key in f[key].keys():
                        nested_dict[nested_key] = f[key][nested_key][()]
                    data[key] = nested_dict
                else:
                    # 对于数据集，直接读取值
                    data[key] = f[key][()]

        return data

    def ConfigsunAnglefile1(self,day):
        Date = datetime.datetime.strptime(day, '%Y-%m-%d')
        file = 'E:/data/SunAngle/%s/%s/Angle_Sun-Moon_%s_%s.h5' \
               % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d'), Date.strftime('%Y-%m-%d'), self.__sat.name)
        data = {}
        with h5py.File(file, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    # 对于组，创建一个嵌套字典
                    nested_dict = {}
                    for nested_key in f[key].keys():
                        nested_dict[nested_key] = f[key][nested_key][()]
                    data[key] = nested_dict
                else:
                    # 对于数据集，直接读取值
                    data[key] = f[key][()]

        return data

    def liner_function(self, scaID=SCAID.No_2, win_size=2, number_hour=1):
        """
        the sun angle data from SunAngle file
        """
        HFNangle1=self.ConfigHFNangle(coord=Coord.x)
        HFNangle2=self.ConfigHFNangle(coord=Coord.y)
        HFNangle3=self.ConfigHFNangle(coord=Coord.z)
        a=HFNangle1[scaID.name]
        b=HFNangle2[scaID.name]
        c=HFNangle3[scaID.name]

        sunangle=self.ConfigsunAnglefile()
        sunangle=sunangle[scaID.name][0:2]

        window_size=win_size*60*60
        start_time = sunangle[0, 0]+number_hour*window_size
        end_time=start_time+window_size

        index1 = np.where((a[0] >= start_time) & (a[0] <= end_time))[0]
        index2 = np.where((b[0] >= start_time) & (b[0] <= end_time))[0]
        index3 = np.where((c[0] >= start_time) & (c[0] <= end_time))[0]
        index0 = np.where((sunangle[0] >= start_time) & (sunangle[0] <= end_time))[0]

        Sunangle = sunangle[:, index0]
        x = a[:, index1]
        y = b[:, index2]
        z = c[:, index3]



        idx_a = np.in1d(Sunangle[0], x[0])
        idx_b = np.in1d(Sunangle[0], y[0])
        idx_c = np.in1d(Sunangle[0], z[0])

        sunAngle1 = Sunangle[:, idx_a]
        sunAngle2 = Sunangle[:, idx_b]
        sunAngle3 = Sunangle[:, idx_c]

        Angle1 = np.vstack((x, sunAngle1[1]))
        Angle2 = np.vstack((y, sunAngle2[1]))
        Angle3 = np.vstack((z, sunAngle3[1]))

        std_x = np.std(a[1])
        std_y = np.std(b[1])
        std_z = np.std(c[1])


        # slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(Angle1[2], Angle1[1])
        # slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(Angle2[2], Angle2[1])
        # slope_z, intercept_z, r_value_x, p_value_z, std_err_z = linregress(Angle3[2], Angle3[1])
        #
        # print(slope_x, intercept_x, r_value_x, p_value_x, std_err_x)
        # print(slope_y, intercept_y, r_value_y, p_value_y, std_err_y)
        # print(slope_z, intercept_z, r_value_x, p_value_z, std_err_z)

        f1= np.polyfit(Angle1[2], abs(Angle1[1]), 2)/std_x
        f2= np.polyfit(Angle2[2], abs(Angle2[1]), 2)/std_y
        f3= np.polyfit(Angle3[2], abs(Angle3[1]), 2)/std_z

        yvals1 = np.polyval(f1, Angle1[2])
        yvals2 = np.polyval(f2, Angle2[2])
        yvals3 = np.polyval(f3, Angle3[2])

        # fig=plt.figure()
        # ax=fig.add_axes([0.1,0.1,0.8,0.8])
        # ax.plot(Angle1[0],yvals1,label='x')
        # ax.plot(Angle2[0],yvals2,label='y')
        # ax.plot(Angle3[0],yvals3,label='z')
        #
        # ax.legend()
        # plt.show()

        print(number_hour+1)
        np.set_printoptions(precision=5, suppress=True)
        print(f1)
        print(f2)
        print(f3)


        return Angle1, Angle2, Angle3, yvals1, yvals2, yvals3
        pass


    def model_function(self, data1, data2, data3, sdata, i):


        # slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(Angle1[2], Angle1[1])
        # slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(Angle2[2], Angle2[1])
        # slope_z, intercept_z, r_value_x, p_value_z, std_err_z = linregress(Angle3[2], Angle3[1])
        #
        # print(slope_x, intercept_x, r_value_x, p_value_x, std_err_x)
        # print(slope_y, intercept_y, r_value_y, p_value_y, std_err_y)
        # print(slope_z, intercept_z, r_value_x, p_value_z, std_err_z)

        f1= np.polyfit(data1, sdata, 2)
        f2= np.polyfit(data2, sdata, 2)
        f3= np.polyfit(data3, sdata, 2)

        yvals1 = np.polyval(f1, data1)
        yvals2 = np.polyval(f2, data1)
        yvals3 = np.polyval(f3, data1)

        # fig=plt.figure()
        # ax=fig.add_axes([0.1,0.1,0.8,0.8])
        # ax.plot(Angle1[0],yvals1,label='x')
        # ax.plot(Angle2[0],yvals2,label='y')
        # ax.plot(Angle3[0],yvals3,label='z')
        #
        # ax.legend()
        # plt.show()

        print(i)
        np.set_printoptions(precision=5, suppress=True)
        print(f1)
        print(f2)
        print(f3)


        return f1, f2, f3
        pass

    def fft_max(self, data, fs=5):
        mean=np.mean(data)
        data=data-mean
        Y = np.fft.fft(data)
        t = np.arange(0,len(data),1)

        N = len(data)
        frequencies = np.fft.fftfreq(N, 1 / fs)

        Y_half = Y[:N // 2]
        frequencies_half = frequencies[:N // 2]

        idx = np.argmax(np.abs(Y_half))

        filter = np.zeros_like(Y)
        if idx!=0:
            filter[N - idx - 1] = Y[N - idx - 1]
            filter[idx] = Y[idx]
        if idx==0:
            filter[0]=Y[0]
        extracted_signal = np.fft.ifft(filter)

        extracted_signal=extracted_signal+mean

        # plt.figure(figsize=(12, 8))
        #
        # plt.subplot(3, 1, 1)
        # plt.plot(t, data)
        # plt.title('row signal')
        # plt.xlabel('step')
        # plt.ylabel('amplitude')
        #
        # plt.subplot(3, 1, 2)
        # plt.plot(frequencies_half, np.abs(Y_half) / (N / 2))
        # plt.title('Normalized Fourier Transform Spectrum')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude')
        #
        # plt.subplot(3, 1, 3)
        # plt.plot(t, extracted_signal.real)  # 反向变换得到的信号并进行归一化
        # plt.title('Maximum frequency signal extracted (normalized)')
        # plt.xlabel('step')
        # plt.ylabel('Amplitude')
        #
        # plt.tight_layout()
        # plt.show()

        return extracted_signal.real
        pass

    def LPF(self, fs, cutoff_freq, data):
        """
        low pass filter
        """
        order = 4 #filter order
        b,a = signal.butter(order, cutoff_freq, btype='low', fs=fs)
        filtered_data = signal.lfilter(b,a,data)

        return filtered_data
        pass


    def plot_fft(self,data1=None,data2=None,data3=None,fs=None):
        data1=data1-np.mean(data1)
        # data2=data2-np.mean(data2)
        data3=data3-np.mean(data3)

        color_1 = (31 / 255, 119 / 255, 180 / 255)
        color_2 = (229 / 255, 114 / 255, 12 / 255)
        color_3 = (35 / 255, 142 / 255, 35 / 255)

        fig2 = plt.figure(figsize=(6, 3))
        ax2 = fig2.add_axes([0.15, 0.1, 0.8, 0.7])

        p1=ax2.magnitude_spectrum(data1,Fs=1/fs)
        # p2=ax2.magnitude_spectrum(data2,Fs=1/fs)
        p3=ax2.magnitude_spectrum(data3,Fs=1/fs)
        plt.close(fig2)

        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_axes([0.15, 0.1, 0.8, 0.7])
        axx=ax.twinx()
        ax.plot(p1[1], p1[0], color=color_1, label='roll')
        # ax.plot(p2[1], p2[0], color=color_2, label='pitch')
        axx.plot(p3[1], p3[0], color='grey', label='sun')
        ax.grid(True)
        ax.legend()

        # ax.set_xscale('log')
        plt.show()


        pass

    def segment_continuous(self,data):
        data=data[:,data[1]>0.1]

        samples=data[0]
        values=data[1]

        segments = []
        current_segment = [(samples[0], values[0])]

        for i in range(1, len(samples)):
            if samples[i] - samples[i - 1] == 1:
                current_segment.append((samples[i], values[i]))
            else:
                segments.append(current_segment)
                current_segment = [(samples[i], values[i])]

        # Add the last segment
        segments.append(current_segment)

        return segments

    def Plot_Noise(self, scaID=SCAID.No_1, win_size=2):
        step=60

        HFNangle1 = self.ConfigHFNangle(coord=Coord.x)
        HFNangle2 = self.ConfigHFNangle(coord=Coord.y)
        HFNangle3 = self.ConfigHFNangle(coord=Coord.z)
        angle1 = HFNangle1[scaID.name]
        angle2 = HFNangle2[scaID.name]
        angle3 = HFNangle3[scaID.name]
        std_x = np.std(angle1[1])
        mean_x = np.mean(angle1[1])
        std_y = np.std(angle2[1])
        mean_y = np.mean(angle2[1])
        std_z = np.std(angle3[1])
        mean_z = np.mean(angle3[1])
        sunangle=self.ConfigsunAnglefile()
        sunangle=sunangle[scaID.name][0:2]
        std_s = np.std(sunangle[1])
        mean_s = np.mean(sunangle[1])

        # ppppp=self.fft_max(data=sunangle[1],fs=1)

        for i in range(12):
            window_size = win_size * 60 * 60
            start_time = sunangle[0, 0] + i * window_size
            end_time = start_time + window_size
            index1 = np.where((angle1[0] >= start_time) & (angle1[0] <= end_time))[0]
            index2 = np.where((angle2[0] >= start_time) & (angle2[0] <= end_time))[0]
            index3 = np.where((angle3[0] >= start_time) & (angle3[0] <= end_time))[0]
            index4 = np.where((sunangle[0] >= start_time) & (sunangle[0] <= end_time))[0]

            roll=angle1[:,index1]
            pitch=angle2[:,index2]
            yaw=angle3[:,index3]
            sun=sunangle[:, index4]

            roll_std=[]
            pitch_std=[]
            yaw_std=[]
            s_mean=[]

            for j in range(int(window_size/step)):
                t1=start_time+j*step
                t2=start_time+(j+1)*step
                q1 = roll[:, np.where((roll[0] >= t1) & (roll[0] < t2))[0]]
                q2 = pitch[:, np.where((pitch[0] >= t1) & (pitch[0] < t2))[0]]
                q3 = yaw[:, np.where((yaw[0] >= t1) & (yaw[0] < t2))[0]]
                s = sun[:,np.where((sun[0] >= t1) & (sun[0] < t2))[0]]
                roll_std.append(np.std(q1[1]))
                pitch_std.append(np.std(q2[1]))
                yaw_std.append(np.std(q3[1]))
                s_mean.append(np.mean(s[1]))


            print(i+1)
            # correlation_matrix1 = pearsonr(np.array(roll_std),np.array(s_mean))
            # correlation_matrix2 = pearsonr(np.array(pitch_std),np.array(s_mean))
            # correlation_matrix3 = pearsonr(np.array(yaw_std),np.array(s_mean))
            #
            # print(correlation_matrix1)
            # print(correlation_matrix2)
            # print(correlation_matrix3)

            # correlation_matrix1 = np.corrcoef(np.array(roll_std),np.array(s_mean))
            # correlation_matrix2 = np.corrcoef(np.array(pitch_std),np.array(s_mean))
            # correlation_matrix3 = np.corrcoef(np.array(yaw_std),np.array(s_mean))
            #
            # print(correlation_matrix1)
            # print(correlation_matrix2)
            # print(correlation_matrix3)



            # m_roll=self.fft_max(data=roll_std).T
            # m_pitch=self.fft_max(data=pitch_std).T
            # m_yaw=self.fft_max(data=yaw_std).T
            # m_sun=self.fft_max(data=s_mean)


            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.7])

            ax2 = ax.twinx()
            color_1 = (31 / 255, 119 / 255, 180 / 255)
            color_2 = (229 / 255, 114 / 255, 12 / 255)
            color_3 = (35 / 255, 142 / 255, 35 / 255)

            continue_roll=self.segment_continuous(np.vstack((np.arange(0,len(roll_std),1),roll_std)))
            continue_pitch=self.segment_continuous(np.vstack((np.arange(0,len(pitch_std),1),pitch_std)))
            continue_yaw=self.segment_continuous(np.vstack((np.arange(0,len(yaw_std),1),yaw_std)))

            print('roll')
            for k1 in range(len(continue_roll)):
                data_roll=np.array([list(pair) for pair in continue_roll[k1]]).T
                filtered_roll=self.LPF(fs=1/step,cutoff_freq=0.0015,data=np.hstack((data_roll[1],data_roll[1],data_roll[1],data_roll[1])))
                filtered_roll=filtered_roll[0:int(len(filtered_roll)/4)]
                # self.plot_fft(data1=data_roll[1],data3=np.array(s_mean)[data_roll[0].astype(int).tolist()],fs=step)
                if len(data_roll[1])>2:
                    correlation_matrix1 = pearsonr(data_roll[1], np.array(s_mean)[data_roll[0].astype(int).tolist()])
                    print(correlation_matrix1)
                else:
                    pass

                # if k1==0:
                #     ax2.plot(data_roll[0],filtered_roll,color=color_1, label='fitered_roll',linestyle='--')
                # else:
                #     ax2.plot(data_roll[0],filtered_roll,color=color_1,linestyle='--')

            print('pitch')
            for k2 in range(len(continue_pitch)):
                data_pitch=np.array([list(pair) for pair in continue_pitch[k2]]).T
                # self.plot_fft(data1=data_roll[1],data3=np.array(s_mean)[data_roll[0].astype(int).tolist()],fs=step)
                if len(data_pitch[1])>2:
                    correlation_matrix1 = pearsonr(data_pitch[1], np.array(s_mean)[data_pitch[0].astype(int).tolist()])
                    print(correlation_matrix1)
                else:
                    pass

            print('yaw')
            for k3 in range(len(continue_yaw)):
                data_yaw=np.array([list(pair) for pair in continue_yaw[k3]]).T
                # self.plot_fft(data1=data_roll[1],data3=np.array(s_mean)[data_roll[0].astype(int).tolist()],fs=step)
                if len(data_yaw[1])>2:
                    correlation_matrix1 = pearsonr(data_yaw[1], np.array(s_mean)[data_yaw[0].astype(int).tolist()])
                    print(correlation_matrix1)
                else:
                    pass

            # filtered_roll=self.LPF(fs=1/step,cutoff_freq=0.0015,data=roll_std)
            # filtered_pitch=self.LPF(fs=1/step,cutoff_freq=0.0015,data=pitch_std)
            # filtered_yaw= self.LPF(fs=1/step, cutoff_freq=0.0015, data=s_mean)

            # correlation_matrix1 = pearsonr(np.array(filtered_roll),np.array(s_mean))
            # correlation_matrix2 = pearsonr(np.array(filtered_pitch),np.array(s_mean))
            # correlation_matrix3 = pearsonr(np.array(filtered_yaw),np.array(s_mean))

            # print(correlation_matrix1)
            # print(correlation_matrix2)
            # print(correlation_matrix3)


            ax2.plot(np.arange(0,len(roll_std),1), roll_std, color=color_1, label='roll')
            # ax2.plot(np.arange(0, len(roll_std), 1), filtered_roll, color=color_1, label='fitered_roll',linestyle='--')

            # ax2.plot(np.arange(0, len(roll_std), 1), m_roll, color=color_1, label='m_roll',linestyle='--')

            ax2.plot(np.arange(0,len(pitch_std),1), pitch_std, color=color_2, label='pitch')
            # ax2.plot(np.arange(0, len(pitch_std), 1), filtered_pitch, color=color_2, label='fitered_pitch', linestyle='--')

            # ax2.plot(np.arange(0, len(pitch_std), 1), m_pitch, color=color_2, label='m_pitch',linestyle='--')

            # ax.plot(np.arange(0,len(yaw_std),1), yaw_std, color=color_3, label='yaw')
            # ax2.plot(np.arange(0,len(yaw_std),1), m_yaw, color=color_3, label='m_yw',linestyle='--')

            ax.plot(np.arange(0, len(s_mean), 1), s_mean, color='grey', label='sun angle', linestyle='--')

            font = {'size': 20}
            font1 = {'size': 20, 'color': 'purple'}
            font2 = {'size': 20, 'color': 'red'}
            ax2.set_ylabel('Std. Deviation (arcsec)', fontdict=font2)
            # ax.set_ylabel('Std. Deviation (arcsec)', fontdict=font1)
            ax.set_ylabel('Sun Angle (deg)', fontdict=font1)
            ax.set_xlabel('Step', fontdict=font)
            ax.set_ylim(0,180)
            y_major_locator = MultipleLocator(30)
            ax.yaxis.set_major_locator(y_major_locator)
            # ax2.set_ylim(0.1)
            y2 = MultipleLocator(1)
            ax2.yaxis.set_major_locator(y2)
            x_major_locator = MultipleLocator(20)
            ax.xaxis.set_major_locator(x_major_locator)
            # y_major_locator1 = MultipleLocator(5)
            # ax.yaxis.set_major_locator(y_major_locator1)
            # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
            # ax.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False)
            ax.tick_params(direction='in', top=True, bottom=True, left=True, labelsize=20, labelcolor='purple')
            ax.tick_params(direction='in', axis='x', labelsize=20, labelcolor='black')
            ax.legend(loc='upper left', ncol=1, fontsize='x-large', framealpha=0.8)
            ax.grid(True, axis='y', linestyle='dashed')

            ax2.tick_params(direction='in', axis='y', left=False, labelsize=20, labelcolor='red')
            ax2.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)


            for spine in ax.spines.values():
                spine.set_linewidth(2)

            fig.savefig('../paper_figure/Fig99.png', dpi=300)
            plt.show()

            # self.plot_fft(roll_std,pitch_std,s_mean,fs=step)
            # plt.show()

            pass

            # roll_std = np.array(roll_std)
            # pitch_std = np.array(pitch_std)
            # yaw_std = np.array(yaw_std)
            #
            # roll_nor = (roll_std - mean_x)/std_x
            # pitch_nor = (pitch_std - mean_y) / std_y
            # yaw_nor = (yaw_std - mean_z) / std_z
            # sun_nor = (s_mean - mean_s) / std_s
            #
            # f1,f2,f3=self.model_function(data1=roll_nor, data2=pitch_nor, data3=yaw_nor, sdata=sun_nor, i=i)






    def sunangle_and_noise(self,win_size=2, number_hour=0):
        """
        Using Sunangle to Model Noise
        """
        fig=plt.figure(figsize=(13.6,6))
        for key in SCAID:
            if key.value == 1:
                ax=fig.add_axes([0.08+ 0.31 * (key.value-1), 0.2 , 0.22, 0.75])
                Angle1, Angle2, Angle3, yvals1, yvals2, yvals3=self.liner_function(scaID=key, win_size=win_size,
                                                                                   number_hour=number_hour)
                ax.scatter(Angle1[0], Angle1[1], s=0.1, alpha=0.8, label='x')
                ax.scatter(Angle2[0], Angle2[1], s=0.1, alpha=0.8, label='y')
                ax.scatter(Angle3[0], Angle3[1], s=0.1, alpha=0.8, label='z')
                ax2 = ax.twinx()
                ax2.plot(Angle1[0], yvals1, alpha=0.8, label='x')
                ax2.plot(Angle2[0], yvals2, alpha=0.8, label='y')
                ax2.plot(Angle3[0], yvals3, alpha=0.8, label='z')

                ax.set_ylim(-75,75)
                font = {'size': 20}
                ax.set_ylabel('Res_Angle (arcsec)', fontdict=font)
                ax.set_xlabel('Time (s)', fontdict=font)

                plt.grid(True, axis='y', linestyle='dashed')

                ax.legend(loc='upper left', fontsize=15)
                ax2.legend(loc='upper right', fontsize=15)

                ax.tick_params(direction='in', top=True, bottom=True, left=True)
                ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5)
                ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

                ax2.tick_params(direction='in', right=True)
                ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
                ax2.tick_params(axis='x', direction='in', right=True)
                y_major_locator = MultipleLocator(25)
                ax.yaxis.set_major_locator(y_major_locator)

                ax.tick_params(axis='both', labelsize=20)
                ax2.tick_params(axis='both', labelsize=20)

                for spine in ax.spines.values():
                    spine.set_linewidth(2)

                ax.text(0, -0.21, s='(g) Noise model for %s-%s'%(self.__sat.name, key.name[-1]),
                        fontdict={'size': 20}, transform=ax.transAxes)

            elif key.value == 2:
                ax = fig.add_axes([0.08 + 0.31 * (key.value - 1), 0.2, 0.22, 0.75])
                Angle1, Angle2, Angle3, yvals1, yvals2, yvals3 = self.liner_function(scaID=key, win_size=win_size,
                                                                                     number_hour=number_hour)
                ax.scatter(Angle1[0], Angle1[1], s=0.1, alpha=0.8, label='x')
                ax.scatter(Angle2[0], Angle2[1], s=0.1, alpha=0.8, label='y')
                ax.scatter(Angle3[0], Angle3[1], s=0.1, alpha=0.8, label='z')
                ax2 = ax.twinx()
                ax2.plot(Angle1[0], yvals1, alpha=0.8, label='x')
                ax2.plot(Angle2[0], yvals2, alpha=0.8, label='y')
                ax2.plot(Angle3[0], yvals3, alpha=0.8, label='z')

                ax.set_ylim(-75, 75)
                font = {'size': 20}
                ax.set_xlabel('Time (s)', fontdict=font)

                plt.grid(True, axis='y', linestyle='dashed')

                ax.legend(loc='upper left', fontsize=15)
                ax2.legend(loc='upper right', fontsize=15)

                ax.tick_params(direction='in', top=True, bottom=True, left=True)
                ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5)
                ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

                ax2.tick_params(direction='in', right=True)
                ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
                ax2.tick_params(axis='x', direction='in', right=True)
                y_major_locator = MultipleLocator(25)
                ax.yaxis.set_major_locator(y_major_locator)

                ax.tick_params(axis='both', labelsize=20)
                ax2.tick_params(axis='both', labelsize=20)

                for spine in ax.spines.values():
                    spine.set_linewidth(2)

                ax.text(0, -0.21, s='(h) Noise model for %s-%s'%(self.__sat.name, key.name[-1]),
                        fontdict={'size': 20}, transform=ax.transAxes)

            elif key.value == 3:
                ax = fig.add_axes([0.08 + 0.31 * (key.value - 1), 0.2, 0.22, 0.75])
                Angle1, Angle2, Angle3, yvals1, yvals2, yvals3 = self.liner_function(scaID=key, win_size=win_size,
                                                                                     number_hour=number_hour)
                ax.scatter(Angle1[0], Angle1[1], s=0.1, alpha=0.8, label='x')
                ax.scatter(Angle2[0], Angle2[1], s=0.1, alpha=0.8, label='y')
                ax.scatter(Angle3[0], Angle3[1], s=0.1, alpha=0.8, label='z')
                ax2 = ax.twinx()
                ax2.plot(Angle1[0], yvals1, alpha=0.8, label='x')
                ax2.plot(Angle2[0], yvals2, alpha=0.8, label='y')
                ax2.plot(Angle3[0], yvals3, alpha=0.8, label='z')

                ax.set_ylim(-75, 75)
                font = {'size': 20}
                ax2.set_ylabel('Standardized STD', fontdict=font)
                ax.set_xlabel('Time (s)', fontdict=font)

                plt.grid(True, axis='y', linestyle='dashed')

                ax.legend(loc='upper left',fontsize=15)
                ax2.legend(loc='upper right', fontsize=15)

                ax.tick_params(direction='in', top=True, bottom=True, left=True)
                ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5)
                ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

                ax2.tick_params(direction='in', right=True)
                ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
                ax2.tick_params(axis='x', direction='in', right=True)
                y_major_locator = MultipleLocator(25)
                ax.yaxis.set_major_locator(y_major_locator)

                ax.tick_params(axis='both', labelsize=20)
                ax2.tick_params(axis='both', labelsize=20)

                for spine in ax.spines.values():
                    spine.set_linewidth(2)

                ax.text(0, -0.21, s='(i) Noise model for %s-%s'%(self.__sat.name, key.name[-1]),
                        fontdict={'size': 20}, transform=ax.transAxes)



        fig.savefig('../paper_figure/Fig20.png', dpi=300)

        plt.show()


        pass


        pass


    def plot_sun(self, begindate='2020-05-01', enddate='2020-05-15', scaID=SCAID.No_1):
        """plot the sun angle from LOS"""
        daylist = GeoMathKit.dayListByDay(begindate,enddate)
        for day in daylist:
            try:
                if day.strftime('%Y-%m-%d')==begindate:
                    sun=ConfigAnglefile(date=day.strftime('%Y-%m-%d'),sat=self.__sat)
                else:
                    sundate = ConfigAnglefile(date=day.strftime('%Y-%m-%d'),sat=self.__sat)
                    for key in SCAID:
                        sun[key.name]=np.hstack((sun[key.name], sundate[key.name]))
            except:
                print('the day of data is empty on %s' % day.strftime('%Y-%m-%d'))


        fig = plt.figure(figsize=(13.6, 6.4))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])

        suntime1 = []
        suntime2 = []
        suntime3 = []

        base_time = datetime.datetime(2000, 1, 1, 12, 0 ,0)
        for sec in sun[SCAID.No_1.name][0]:
            dt = base_time + datetime.timedelta(seconds=sec)
            suntime1.append(dt)
            pass

        for sec in sun[SCAID.No_2.name][0]:
            dt = base_time + datetime.timedelta(seconds=sec)
            suntime2.append(dt)
            pass

        for sec in sun[SCAID.No_3.name][0]:
            dt = base_time + datetime.timedelta(seconds=sec)
            suntime3.append(dt)
            pass

        color_1 = (31 / 255, 119 / 255, 180 / 255)
        color_2 = (229 / 255, 114 / 255, 12 / 255)
        color_3 = (35 / 255, 142 / 255, 35 / 255)

        ax.plot(suntime1, sun[SCAID.No_1.name][1], color=color_1, label='No_1', linestyle='--')
        ax.plot(suntime2, sun[SCAID.No_2.name][1], color=color_2, label='No-2', linestyle='--')
        ax.plot(suntime3, sun[SCAID.No_3.name][1], color=color_3, label='No-3', linestyle='--')

        font = {'size': 20}
        ax.set_ylim(0, 180)
        ax.set_ylabel('Sun angle (deg)', fontdict=font)
        ax.set_xlabel('date', fontdict=font)
        y_major_locator = MultipleLocator(20)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        # ax.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False)
        ax.tick_params(direction='in', top=True, bottom=True, left=True, labelsize=20)

        ax.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
        ax.grid(True, axis='y', linestyle='dashed')

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.show()


        pass



if __name__=='__main__':
    a = SunModel(date='2020-04-01', sat=SatID.C)
    a.Plot_Noise(scaID=SCAID.No_1, win_size=2)
    pass