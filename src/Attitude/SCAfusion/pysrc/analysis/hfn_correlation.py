import os

import matplotlib.pyplot as plt

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


class CorrelationAnalysis():
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

    def Analysis1(self,win_size,scaID:SCAID):
        """
        Perform correlation analysis on the same time series
        obtained by interpolation of the three-axis residual angle
        """

        HFNangle1=self.ConfigHFNangle(coord=Coord.x)
        HFNangle2=self.ConfigHFNangle(coord=Coord.y)
        HFNangle3=self.ConfigHFNangle(coord=Coord.z)
        angle1=HFNangle1[scaID.name]
        angle2=HFNangle2[scaID.name]
        angle3=HFNangle3[scaID.name]

        a=np.interp(self.__time,angle1[0],angle1[1])
        b=np.interp(self.__time,angle2[0],angle2[1])
        c=np.interp(self.__time,angle3[0],angle3[1])

        Angle1=np.vstack((self.__time,a))
        Angle2=np.vstack((self.__time,b))
        Angle3=np.vstack((self.__time,c))

        window_size=win_size * 60 * 60
        num_windows = len(Angle1[0]) // window_size

        for i in range(num_windows):
            # 计算时间窗口的起始索引和结束索引
            start_index = i * window_size
            end_index = (i + 1) * window_size

            # 提取时间窗口内的数据
            window_data1 = Angle1[1, start_index:end_index]
            window_data2 = Angle2[1, start_index:end_index]
            window_data3 = Angle3[1, start_index:end_index]

            data = np.stack((window_data1, window_data2, window_data3), axis=0)

            covariance_matrix = np.cov(data)
            print("协方差矩阵:")
            print(covariance_matrix)

            correlation_matrix = np.corrcoef(data)
            print("\n皮尔逊相关系数:")
            print(correlation_matrix)




        pass

    def Analysis2(self,win_size,scaID:SCAID):
        """
        Correlation analysis of three-axis residual angle by searching for data at the same time point in the data set
        """

        HFNangle1=self.ConfigHFNangle(coord=Coord.x)
        HFNangle2=self.ConfigHFNangle(coord=Coord.y)
        HFNangle3=self.ConfigHFNangle(coord=Coord.z)
        a=HFNangle1[scaID.name]
        b=HFNangle2[scaID.name]
        c=HFNangle3[scaID.name]
        AA=[]

        # print(np.std(a[1]))
        # print(np.std(b[1]))
        # print(np.std(c[1]))

        idx_a = np.in1d(a[0], np.intersect1d(b[0], c[0]))
        idx_b = np.in1d(b[0], np.intersect1d(a[0], c[0]))
        idx_c = np.in1d(c[0], np.intersect1d(a[0], b[0]))

        Angle1 = a[:, idx_a]
        Angle2 = b[:, idx_b]
        Angle3 = c[:, idx_c]

        # print(np.std(Angle1[1]))
        # print(np.std(Angle2[1]))
        # print(np.std(Angle3[1]))

        window_size = win_size * 60 * 60
        num_windows = len(self.__time) // window_size


        for i in range(num_windows):
            start_index = i * window_size
            end_index = (i + 1) * window_size
            time=self.__time[start_index:end_index]
            indx_time=np.in1d(Angle1[0],time)
            window_data1=Angle1[:,indx_time]
            window_data2=Angle2[:,indx_time]
            window_data3=Angle3[:,indx_time]

            data = np.stack((window_data1[1], window_data2[1], window_data3[1]), axis=0)

            covariance_matrix = np.cov(data)
            print("协方差矩阵:")
            self.print_slowly(" ")
            print(covariance_matrix)
            ma = covariance_matrix.flatten()
            AA.append(ma)
            correlation_matrix = np.corrcoef(data)
            print("\n皮尔逊相关系数:")
            print(correlation_matrix)

        AA=np.array(AA)

        return AA
        pass


    def Analysis3(self, win_size=3, scaID=SCAID.No_1):
        HFNangle1=self.ConfigHFNangle(coord=Coord.x)
        HFNangle2=self.ConfigHFNangle(coord=Coord.y)
        HFNangle3=self.ConfigHFNangle(coord=Coord.z)
        a=HFNangle1[scaID.name]
        b=HFNangle2[scaID.name]
        c=HFNangle3[scaID.name]

        sunangle=self.ConfigsunAnglefile()
        sunangle=sunangle[scaID.name][0:2]

        idx_a = np.in1d(sunangle[0], a[0])
        idx_b = np.in1d(sunangle[0], b[0])
        idx_c = np.in1d(sunangle[0], c[0])

        sunAngle1 = sunangle[:, idx_a]
        sunAngle2 = sunangle[:, idx_b]
        sunAngle3 = sunangle[:, idx_c]

        Angle1 = np.vstack((a, sunAngle1[1]))
        Angle2 = np.vstack((b, sunAngle2[1]))
        Angle3 = np.vstack((c, sunAngle3[1]))

        window_size = win_size * 60 * 60
        num_windows = len(self.__time) // window_size

        for i in range(num_windows):
            start_index = i * window_size
            end_index = (i + 1) * window_size
            time=self.__time[start_index:end_index]
            indx_time1 = np.in1d(Angle1[0],time)
            indx_time2 = np.in1d(Angle2[0], time)
            indx_time3 = np.in1d(Angle3[0], time)
            window_data1=Angle1[:, indx_time1]
            window_data2=Angle2[:, indx_time2]
            window_data3=Angle3[:, indx_time3]

            window_data1[1]= window_data1[1]**2
            window_data2[1]= window_data2[1]**2
            window_data3[1]= window_data3[1]**2

            covariance_matrix1 = np.cov(np.abs(window_data1[1:, :]))
            covariance_matrix2 = np.cov(np.abs(window_data2[1:, :]))
            covariance_matrix3 = np.cov(np.abs(window_data3[1:, :]))
            print("The covariance matrix between the x-axis and sunangle:")
            # self.print_slowly(" ")
            print(covariance_matrix1)

            print("The covariance matrix between the y-axis and sunangle:")
            # self.print_slowly(" ")
            print(covariance_matrix2)

            print("The covariance matrix between the z-axis and sunangle:")
            # self.print_slowly(" ")
            print(covariance_matrix3)

            correlation_matrix1 = np.corrcoef(window_data1[1:, :])
            correlation_matrix2 = np.corrcoef(window_data2[1:, :])
            correlation_matrix3 = np.corrcoef(window_data3[1:, :])
            print("The Pearson correlation coefficient between the x-axis and sunangle:")
            # self.print_slowly(" ")
            print(correlation_matrix1)

            print("The Pearson correlation coefficient between the y-axis and sunangle:")
            # self.print_slowly(" ")
            print(correlation_matrix2)

            print("The Pearson correlation coefficient between the z-axis and sunangle:")
            # self.print_slowly(" ")
            print(correlation_matrix3)

        pass


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

    def quadratic_polynomials(self, scaID=SCAID.No_2, win_size=2, number_hour=1):
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

        f1= np.polyfit(Angle1[2], abs(Angle1[1]), 2)/std_x
        f2= np.polyfit(Angle2[2], abs(Angle2[1]), 2)/std_y
        f3= np.polyfit(Angle3[2], abs(Angle3[1]), 2)/std_z

        yvals1 = np.polyval(f1, Angle1[2])
        yvals2 = np.polyval(f2, Angle2[2])
        yvals3 = np.polyval(f3, Angle3[2])

        return f1, f2, f3
        pass


    def quadratic_polynomials1(self, scaID=SCAID.No_2, win_size=2, number_hour=1):
        """
        the sun angle data from SunMoonAngle file
        """
        HFNangle1=self.ConfigHFNangle(coord=Coord.x)
        HFNangle2=self.ConfigHFNangle(coord=Coord.y)
        HFNangle3=self.ConfigHFNangle(coord=Coord.z)
        a=HFNangle1[scaID.name]
        b=HFNangle2[scaID.name]
        c=HFNangle3[scaID.name]

        sunangle=ConfigAnglefile(date=self.__date, sat=self.__sat)
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

        ide_a = np.in1d(x[0], Sunangle[0])
        ide_b = np.in1d(y[0], Sunangle[0])
        ide_c = np.in1d(z[0], Sunangle[0])

        sunAngle1 = Sunangle[:, idx_a]
        sunAngle2 = Sunangle[:, idx_b]
        sunAngle3 = Sunangle[:, idx_c]

        X = x[:, ide_a]
        Y = y[:, ide_b]
        Z = z[:, ide_c]

        Angle1 = np.vstack((X, sunAngle1[1]))
        Angle2 = np.vstack((Y, sunAngle2[1]))
        Angle3 = np.vstack((Z, sunAngle3[1]))

        std_x = np.std(a[1])
        std_y = np.std(b[1])
        std_z = np.std(c[1])

        f1= np.polyfit(Angle1[2], abs(Angle1[1]), 2)/std_x
        f2= np.polyfit(Angle2[2], abs(Angle2[1]), 2)/std_y
        f3= np.polyfit(Angle3[2], abs(Angle3[1]), 2)/std_z


        return f1, f2, f3
        pass

    def liner_function1(self,scaID=SCAID.No_2, win_size=2, number_hour=0):
        """
        the sun angle data from SunMoonAngle file
        """
        HFNangle1=self.ConfigHFNangle(coord=Coord.x)
        HFNangle2=self.ConfigHFNangle(coord=Coord.y)
        HFNangle3=self.ConfigHFNangle(coord=Coord.z)
        a=HFNangle1[scaID.name]
        b=HFNangle2[scaID.name]
        c=HFNangle3[scaID.name]

        sunangle=ConfigAnglefile(date=self.__date, sat=self.__sat)
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

        ide_a = np.in1d(x[0], Sunangle[0])
        ide_b = np.in1d(y[0], Sunangle[0])
        ide_c = np.in1d(z[0], Sunangle[0])

        sunAngle1 = Sunangle[:, idx_a]
        sunAngle2 = Sunangle[:, idx_b]
        sunAngle3 = Sunangle[:, idx_c]

        X = x[:, ide_a]
        Y = y[:, ide_b]
        Z = z[:, ide_c]

        Angle1 = np.vstack((X, sunAngle1[1]))
        Angle2 = np.vstack((Y, sunAngle2[1]))
        Angle3 = np.vstack((Z, sunAngle3[1]))

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

        # fig=plt.figure(figsize=(13.6,6.4))
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

    def Plot_Noise(self, scaID=SCAID.No_1, win_size=2, number_hour=1):
        step=120

        HFNangle1 = self.ConfigHFNangle(coord=Coord.x)
        HFNangle2 = self.ConfigHFNangle(coord=Coord.y)
        HFNangle3 = self.ConfigHFNangle(coord=Coord.z)
        angle1 = HFNangle1[scaID.name]
        angle2 = HFNangle2[scaID.name]
        angle3 = HFNangle3[scaID.name]
        sunangle=self.ConfigsunAnglefile()
        sunangle=sunangle[scaID.name][0:2]

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

            # time1=[]
            # time2=[]
            # time3=[]

            # base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
            # for sec in roll[0]:
            #     dt = base_time + datetime.timedelta(seconds=sec)
            #     time1.append(dt)
            #     pass
            #
            # for sec in pitch[0]:
            #     dt = base_time + datetime.timedelta(seconds=sec)
            #     time2.append(dt)
            #     pass
            #
            # for sec in yaw[0]:
            #     dt = base_time + datetime.timedelta(seconds=sec)
            #     time3.append(dt)
            #     pass

            fig = plt.figure(figsize=(13.6, 6.4))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax2 = ax.twinx()
            color_1 = (31 / 255, 119 / 255, 180 / 255)
            color_2 = (229 / 255, 114 / 255, 12 / 255)
            color_3 = (35 / 255, 142 / 255, 35 / 255)

            # ax2.plot(np.arange(0,len(roll_std),1), roll_std, color=color_1, label='roll')
            # ax2.plot(np.arange(0,len(pitch_std),1), pitch_std, color=color_2, label='pitch')
            ax2.plot(np.arange(0,len(yaw_std),1), yaw_std, color=color_3, label='yaw')
            ax.plot(np.arange(0, len(s_mean), 1), s_mean, color=color_3, label='sun angle', linestyle='--')

            font = {'size': 20}
            ax2.set_ylabel('Standard Deviation (arcsec)', fontdict=font)
            ax.set_ylabel('Sun angle (deg)', fontdict=font)
            ax.set_xlabel('Step', fontdict=font)
            ax.set_ylim(0,180)
            y_major_locator = MultipleLocator(20)
            ax.yaxis.set_major_locator(y_major_locator)
            # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
            # ax.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False)
            ax.tick_params(direction='in', top=True, bottom=True, left=True, labelsize=20)

            ax.legend(loc='upper left', ncol=1, fontsize='x-large', framealpha=0.8)
            plt.grid(True, axis='y', linestyle='dashed')

            ax2.tick_params(direction='in', top=True, bottom=True, left=False, labelsize=20)
            ax2.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)


            for spine in ax.spines.values():
                spine.set_linewidth(2)

            plt.show()



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

    def model_coefficient(self):
        Roll={}
        Pitch={}
        Yaw={}

        for key in SCAID:
            for i in range(12):
                if i == 0:
                    C_roll, C_pitch, C_yaw = self.quadratic_polynomials(scaID=key, number_hour=i)
                else:
                    roll, pitch, yaw = self.quadratic_polynomials(scaID=key, number_hour=i)
                    C_roll = np.vstack((C_roll, roll))
                    C_pitch = np.vstack((C_pitch, pitch))
                    C_yaw = np.vstack((C_yaw, yaw))
                pass
            pass
            Roll[key.name] = np.array([np.mean(C_roll[:,0]), np.mean(C_roll[:,1]), np.mean(C_roll[:,2])])
            Pitch[key.name] = np.array([np.mean(C_pitch[:,0]), np.mean(C_pitch[:,1]), np.mean(C_pitch[:,2])])
            Yaw[key.name] = np.array([np.mean(C_yaw[:,0]), np.mean(C_yaw[:,1]), np.mean(C_yaw[:,1])])

        return Roll, Pitch, Yaw

    def lamda(self,time, sunangle, std):
        Roll, Pitch, Yaw = self.model_coefficient()
        r = {}
        for key in SCAID:
            sun = np.interp(time, sunangle[key.name][0], sunangle[key.name][1])
            r[key.name] = np.array([np.polyval(Roll[key.name], sun)*std[key.name][0],
                                    np.polyval(Pitch[key.name], sun)*std[key.name][1],
                                    np.polyval(Yaw[key.name], sun)*std[key.name][2]])
            pass
        R = np.vstack((r[SCAID.No_1.name], r[SCAID.No_2.name], r[SCAID.No_3.name]))

        return R




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

    def plot_model_noise(self, scaID:SCAID):
        """plot model noise as gif picture"""
        fig = plt.figure(figsize=(3,2.5))
        ax=fig.add_axes([0.15, 0.1, 0.8, 0.8])

        def update(frame):
            ax.clear()
            Angle1, Angle2, Angle3, yvals1, yvals2, yvals3=self.liner_function(scaID=scaID, number_hour=frame)
            roll = np.vstack((Angle1[0], yvals1))
            pitch = np.vstack((Angle2[0], yvals2))
            yaw = np.vstack((Angle3[0], yvals3))

            time1=[]
            time2=[]
            time3=[]
            base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
            for sec in roll[0]:
                dt = base_time + datetime.timedelta(seconds=sec)
                time1.append(dt)
                pass
            for sec in pitch[0]:
                dt = base_time + datetime.timedelta(seconds=sec)
                time2.append(dt)
                pass
            for sec in yaw[0]:
                dt = base_time + datetime.timedelta(seconds=sec)
                time3.append(dt)
                pass

            color_1 = (31 / 255, 119 / 255, 180 / 255)
            color_2 = (229 / 255, 114 / 255, 12 / 255)
            color_3 = (35 / 255, 142 / 255, 35 / 255)

            # ax.plot(time1, roll[1], color=color_1, label='roll')
            ax.plot(time2, pitch[1], color=color_2, label='pitch')
            # ax.plot(time3, yaw[1], color=color_3, label='yaw')

            font = {'size': 20}
            # ax.set_ylabel('Model Noise (arcsec)', fontdict=font)
            # ax.set_xlabel('Time (s)', fontdict=font)
            ax.set_ylim(0.5, 1)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            # ax.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False)
            ax.tick_params(direction='in', top=True, bottom=True, left=True, labelsize=20)

            ax.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
            ax.grid(True, axis='y', linestyle='dashed')

            for spine in ax.spines.values():
                spine.set_linewidth(2)

        ani = FuncAnimation(fig, update, frames=12, repeat=False)
        ani.save("model_noise.gif", writer="imagemagick", fps=2)


        pass

    def plot_model_noise1(self, scaID:SCAID):
        for i in range(12):
            Angle1, Angle2, Angle3, yvals1, yvals2, yvals3 = self.liner_function1(scaID=scaID, number_hour=i)
            roll = np.vstack((Angle1[0], yvals1))
            pitch = np.vstack((Angle2[0], yvals2))
            yaw = np.vstack((Angle3[0], yvals3))
            if i==0:
                Roll=roll
                Pitch=pitch
                Yaw=yaw
            else:
                Roll=np.hstack((Roll, roll))
                Pitch=np.hstack((Pitch,pitch))
                Yaw=np.hstack((Yaw,yaw))
            pass
        fig = plt.figure(figsize=(13.6, 2.5))
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])

        color_1 = (31 / 255, 119 / 255, 180 / 255)
        color_2 = (229 / 255, 114 / 255, 12 / 255)
        color_3 = (35 / 255, 142 / 255, 35 / 255)

        time1 = []
        time2 = []
        time3 = []
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        for sec in Roll[0]:
            dt = base_time + datetime.timedelta(seconds=sec)
            time1.append(dt)
            pass
        for sec in Pitch[0]:
            dt = base_time + datetime.timedelta(seconds=sec)
            time2.append(dt)
            pass
        for sec in Yaw[0]:
            dt = base_time + datetime.timedelta(seconds=sec)
            time3.append(dt)
            pass

        # ax.plot(time1, Roll[1], color=color_1, label='roll')
        ax.plot(time2, Pitch[1], color=color_2, label='pitch')
        # ax.plot(time3, Yaw[1], color=color_3, label='yaw')

        font = {'size': 20}
        # ax.set_ylabel('Model Noise (arcsec)', fontdict=font)
        # ax.set_xlabel('Time (s)', fontdict=font)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False, labelsize=20)
        # ax.tick_params(direction='in', top=True, bottom=True, left=True, labelsize=20)

        ax.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
        ax.grid(True, axis='y', linestyle='dashed')

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/figure99.png', dpi=300)

        plt.show()


    def plot_coefficient(self, begindate='2020-05-01', enddate='2020-05-05', scaID=SCAID.No_1):
        daylist=GeoMathKit.dayListByDay(begindate,enddate)
        for day in daylist:
            self.__date=day.strftime('%Y-%m-%d')
            for i in range(12):
                if day.strftime('%Y-%m-%d')==begindate and i==0:
                    C_roll, C_pitch, C_yaw = self.quadratic_polynomials(scaID=scaID, number_hour=i)
                else:
                    roll, pitch, yaw = self.quadratic_polynomials(scaID=scaID, number_hour=i)
                    C_roll=np.vstack((C_roll, roll))
                    C_pitch=np.vstack((C_pitch,pitch))
                    C_yaw=np.vstack((C_yaw,yaw))
                pass
            pass

        fig = plt.figure(figsize=(13.6, 2.5))
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])
        color_1 = (31 / 255, 119 / 255, 180 / 255)
        color_2 = (229 / 255, 114 / 255, 12 / 255)
        color_3 = (35 / 255, 142 / 255, 35 / 255)

        array=np.arange(0,len(C_roll[:,0]),1)
        ax.plot(array,C_roll[:,0],label='$C_{2}$',color=color_1)
        # ax.plot(array, C_roll[:, 1], label='$C_{1}$', color=color_2)
        # ax.plot(array, C_roll[:, 2], label='$C_{0}$', color=color_3)

        # array=np.arange(0,len(C_pitch[:,0]),1)
        # ax.plot(array,C_pitch[:,0],label='$C_{2}$',color=color_1)
        # ax.plot(array, C_pitch[:, 1], label='$C_{1}$', color=color_2)
        # ax.plot(array, C_pitch[:, 2], label='$C_{0}$', color=color_3)

        # array=np.arange(0,len(C_yaw[:,0]),1)
        # ax.plot(array,C_yaw[:,0],label='$C_{2}$',color=color_1)
        # ax.plot(array, C_yaw[:, 1], label='$C_{1}$', color=color_2)
        # ax.plot(array, C_yaw[:, 2], label='$C_{0}$', color=color_3)



        # ax.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False, labelsize=20)
        ax.tick_params(direction='in', top=True, bottom=True, left=True, labelsize=20)
        ax.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
        ax.grid(True, axis='y', linestyle='dashed')

        # x_major_locator = MultipleLocator(10)
        # ax.xaxis.set_major_locator(x_major_locator)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/figure99.png', dpi=300)

        plt.show()

        pass






if __name__=='__main__':
    a=CorrelationAnalysis(date='2020-05-01',sat=SatID.C)
    # AA = a.Analysis2(win_size=3,scaID=SCAID.No_1)
    # a.Analysis3(win_size=24,scaID=SCAID.No_1)
    # for i in range(12):
    #     a.liner_function1(scaID=SCAID.No_3, win_size=2, number_hour=i)
    #     a.liner_function(scaID=SCAID.No_3, win_size=2, number_hour=i)
    # pass
    # a.sunangle_and_noise( win_size=2, number_hour=2)

    # a.plot_sun(begindate='2020-05-01', enddate='2020-05-03')
    # a.plot_model_noise(scaID=SCAID.No_2)
    # a.plot_model_noise1(scaID=SCAID.No_2)
    # a.plot_coefficient(begindate='2020-05-03',enddate='2020-05-03',scaID=SCAID.No_3)
    # R=a.lamda(time=641583202.5)
    a.Plot_Noise(scaID=SCAID.No_1, win_size=2, number_hour=1)
    pass










