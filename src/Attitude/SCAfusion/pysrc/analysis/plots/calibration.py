import matplotlib.dates as mdates
import numpy as np
from pysrc.core.geometry import GeoMathKit
from pysrc.core.settings import SatID, SCAID, Mission, CA
from enum import Enum
import datetime
from pysrc.analysis import hfn_analysis as HFNAnalysis
from pysrc.fusion.multi_sca import SCAweighting
from tqdm import trange
import quaternion
import copy
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
import matplotlib.pyplot as plt


class CaPlot():
    def __init__(self, sat: SatID, begindate='2020-02-01', enddate='2020-02-28'):
        self._sat = sat
        self._begindate = begindate
        self._enddate = enddate

    def ConfigCA(self, data, date):
        day = datetime.datetime.strptime(date, '%Y-%m-%d')
        file = '../CA/%s/%s/%s_%s_%s.npy' % (
        day.strftime('%Y-%m'), day.strftime('%Y-%m-%d'), data, day.strftime('%Y-%m-%d'), self._sat.name)
        CA = np.load(file)

        return CA

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

        res_mean = np.mean(res2[-1])

        return res2, res_mean

    def GetAngle(self, ca: CA):
        daylist = GeoMathKit.dayListByDay(self._begindate, self._enddate)
        Resx = np.array([])
        Resy = np.array([])
        Resz = np.array([])
        for day in daylist:
            if day.strftime('%Y-%m-%d') == self._begindate:
                caAng = self.ConfigCA(data=ca.name, date=day.strftime('%Y-%m-%d'))
                CAng = caAng[0:, np.abs(caAng[5]) <= 6]
                resx = CAng[[0, 1, 2], :]
                Resx, Resx_mean = self.calculate_std_with_outliers(resx)
                resy = CAng[[0, 1, 3], :]
                Resy, Resy_mean = self.calculate_std_with_outliers(resy)
                resz = CAng[[0, 1, 4], :]
                Resz, Resz_mean = self.calculate_std_with_outliers(resz)

                print('The average Euler angles for camera calibration are:%s %s %s on %s'
                      % (Resx_mean, Resy_mean, Resz_mean, day.strftime('%Y-%m-%d')))

            else:
                caAng = self.ConfigCA(data=ca.name, date=day.strftime('%Y-%m-%d'))
                CAng = caAng[0:, np.abs(caAng[5]) <= 6]
                resx = CAng[[0, 1, 2], :]
                Resx_temp, Resx_mean = self.calculate_std_with_outliers(resx)
                resy = CAng[[0, 1, 3], :]
                Resy_temp, Resy_mean = self.calculate_std_with_outliers(resy)
                resz = CAng[[0, 1, 4], :]
                Resz_temp, Resz_mean = self.calculate_std_with_outliers(resz)

                print('The average Euler angles for camera calibration are:%s %s %s on %s'
                      % (Resx_mean, Resy_mean, Resz_mean, day.strftime('%Y-%m-%d')))

                Resx = np.hstack((Resx, Resx_temp))
                Resy = np.hstack((Resy, Resy_temp))
                Resz = np.hstack((Resz, Resz_temp))

        return Resx, Resy, Resz

        pass

    def plotCA(self, ca=CA.CA12):

        Resx, Resy, Resz = self.GetAngle(ca=ca)

        seconds_x = Resx[0, :] + Resx[1, :] * 1e-6
        x = Resx[2, :]/(3.6*180/np.pi)
        # avg_x=Resx[2,:]
        label_x = 'roll'

        seconds_y = Resy[0, :] + Resy[1, :] * 1e-6
        y = Resy[2, :]/(3.6*180/np.pi)
        # avg_y=Resy[2,:]
        label_y = 'pitch'

        seconds_z = Resz[0, :] + Resz[1, :] * 1e-6
        z = Resz[2, :]/(3.6*180/np.pi)
        # avg_z=Resz[2,:]
        label_z = 'yaw'

        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time_x = []
        time_y = []
        time_z = []
        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        # 设置图片宽度和长度
        # fig, ax = plt.subplots(figsize=(12.8, 4.8))
        fig, ax = plt.subplots()

        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)



        ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        ax.scatter(time_x, x, s=0.1, color=color_x, label=label_x, alpha=0.5)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.set_xlabel('Time(days)')
        ax.set_ylabel('res_angle(mrad)')

        plt.ylim(-1,1)
        # plt.ylim(-75, 75)
        # 1.设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%m-%d')
        # ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # 2.设置横坐标的日期格式为 "%H"
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_mean[:,1],label='x_mean')
        # plt.figure(days, Res_mean[:, 2], label='y_mean')
        # plt.figure(days, Res_mean[:, 3], label='z_mean')
        plt.grid(True, axis='y', linestyle='dashed')

        plt.legend(loc="upper right")
        plt.show()
        pass


if __name__ == '__main__':
    a = CaPlot(sat=SatID.D, begindate='2020-04-01', enddate='2020-04-01')
    a.plotCA(ca=CA.CA13)

    pass
