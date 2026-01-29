"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/12/15
@Description:
"""
from Setting import SatID, Mission, IMUID
from GeoMathKit import GeoMathKit
import numpy as np
import matplotlib.pyplot as plt
from GetInstrument import GetInstrument_L1A, GetInstrument_L1B
from matplotlib.offsetbox import AnchoredText
import os
from kalman import GetFusionRes
from PointingAnalysis import Pointing

'''This is designed for paper2----IMU peformance assessment'''


def BiasIMUoneDay(date='2019-10-01'):
    plt.style.use(['science', 'grid', 'ieee'])

    dataDir = '../result/product/GRACE_FO/RL04/L1B'
    fileDir = os.path.join(dataDir, '%s/' % date)
    fig = plt.figure()

    for sat in [SatID.C, SatID.D]:
        filename = 'BiasAll_%s.npy' % (sat.name)
        bias = np.load(str(fileDir) + filename)
        # bias= GeoMathKit.rad2as(bias)
        if sat == SatID.C:
            figindex = 1
        else:
            figindex = 2
        ax = fig.add_subplot(2, 1, figindex)
        drift_rate = (bias[-1] - bias[0]) / 3600 / 24
        bias = bias[0::8 * 240, :]
        t = np.arange(len(bias[:, 0])) / 3600 * 240
        ax.plot(t, bias[:, 0], label='Bias X')
        ax.plot(t, bias[:, 1], label='Bias Y')
        ax.plot(t, bias[:, 2], label='Bias Z')
        ax.legend()
        # text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
        # ax.add_artist(text)
        ax.set_ylabel('arcsec per sec')
        ax.set_xlabel('Hours')
        ax.set_title('Sat %s' % sat.name)

    fig.align_ylabels()
    fig.align_xlabels()

    plt.show()

    pass


def BiasIMU_DailyMean(begin='2019-10-01', end='2020-12-31', sat=SatID.C):
    dataDir = '../result/product/GRACE_FO/RL04/L1B'

    days = GeoMathKit.dayListByDay(begin, end)

    dailymean = []
    for day in days:
        date = day.strftime("%Y-%m-%d")
        print(date, sat)
        fileDir = os.path.join(dataDir, '%s/' % date)
        filename = 'BiasAll_%s.npy' % (sat.name)
        try:
            bias = np.load(str(fileDir) + filename)
        except FileNotFoundError as e:
            print(e)
            continue
        bias_mean = GeoMathKit.rad2as(np.mean(bias, 0))
        dailymean.append(bias_mean)

    np.save('../paper2/Bias_IMU/%s-%s-%s.npy' % (begin, end, sat.name), np.array(dailymean))
    pass


def plot_BiasIMU_DailyMean(begin='2019-10-01', end='2019-10-05', sat=SatID.C):
    dailymean = np.load('../paper2/Bias_IMU/%s-%s-%s.npy' % (begin, end, sat.name))
    # plt.style.use(['science', 'grid','ieee'])
    plt.style.use(['science', 'grid'])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(dailymean[:,0]))
    # ax.plot(dailymean[:,0], ls = '--', marker = 'o', label='X')
    # ax.plot(dailymean[:, 1], ls='--', marker='^', label='Y')
    # ax.plot(dailymean[:, 2], ls='--', marker='s', label='Z')

    ax.scatter(x,dailymean[:,0], marker = 'o', s = 1.5, label='X')
    ax.scatter(x,dailymean[:, 1],  marker='o', s =1.5,label='Y')
    ax.scatter(x,dailymean[:, 2], marker='o', s=1.5, label='Z')

    # ax.plot(dailymean[:,0], label='X')
    # ax.plot(dailymean[:, 1],label='Y')
    # ax.plot(dailymean[:, 2], label='Z')

    ax.legend()
    ax.set_ylabel('arcsec per sec')
    ax.set_xlabel('Days')

    plt.show()

    pass


def PointingError(date=None, sat=SatID.D):
    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    # date = '2019-01-06'
    # sat = SatID.D

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    SCA = L1B.getSCA(sat=sat)

    lr = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    fus = lr.getFusion(sat=sat)

    lr1 = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/Method/cov')
    fus1 = lr1.getFusion(sat=sat)

    lr2 = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    combine2 = lr.getSCAcombine(sat=sat)

    '''find the common time'''
    SCA_time, fus_time = SCA[0], fus[0]
    a, b = [], []

    k = 0
    for i, time1 in zip(range(len(SCA_time)), SCA_time):
        for j in range(k, len(fus_time)):
            if np.fabs(time1 - fus_time[j]) < 0.001:
                a.append(i)
                b.append(j)
                k = j
                break

    time = SCA_time[a]
    data1_q = SCA[1:, a]
    data2_q = fus[1:, b]

    po = Pointing(SCA_attitude=fus[:, b].T).loadLos(date=date, dataDir='../result/product/GRACE_FO/RL04/L1B')
    time, roll, pitch, yaw = po.getRPY(sat)

    po1 = Pointing(SCA_attitude=fus1[:, b].T).loadLos(date=date, dataDir='../result/product/GRACE_FO/RL04/L1B')
    time3, roll3, pitch3, yaw3 = po1.getRPY(sat)

    # ax = fig.add_subplot(3, 1, 1)
    # ax.plot(time, roll, label='roll')
    # ax.plot(time, pitch, label='pitch')
    # ax.plot(time, yaw, label='yaw')
    # ax.set_title('JPL-SCA pointing error')
    # ax.set_xlabel('GPS time [sec]')
    # ax.set_ylabel('[rad]')
    # ax.legend()
    #
    # ax = fig.add_subplot(3, 1, 2)
    # # ax.plot(time, roll, label='roll')
    # ax.plot(time, pitch, label='pitch')
    # ax.plot(time, yaw, label='yaw')
    # ax.set_title('JPL-SCA pointing error')
    # ax.set_xlabel('GPS time [sec]')
    # ax.set_ylabel('[rad]')
    # ax.legend()

    po = Pointing(SCA_attitude=SCA[:, a].T).loadLos(date=date, dataDir='../result/product/GRACE_FO/RL04/L1B')
    time1, roll1, pitch1, yaw1 = po.getRPY(sat)

    po = Pointing(SCA_attitude=combine2[:, ].T).loadLos(date=date, dataDir='../result/product/GRACE_FO/RL04/L1B')
    time2, roll2, pitch2, yaw2 = po.getRPY(sat)

    ax = fig.add_subplot(1, 1, 1)
    # ax.plot(time, roll, label='roll')
    ax.plot(time2, pitch2, label='SCA only')
    ax.plot(time1, pitch1, label='JPL')
    ax.plot(time, pitch, label='Fusion')
    ax.plot(time, pitch3, label='Free')
    ax.plot(time, pitch - pitch1, label='JPL-Fusion')
    ax.plot(time, pitch - pitch3, label='Free-Fusion')
    ax.plot(time, pitch3 - pitch1, label='JPL-Free')
    # ax.set_title('JPL-SCA pointing error')
    ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('[rad]')
    ax.legend()
    ax.set_ylim([-1e-3, 1e-3])
    #
    plt.show()
    return time1, roll1, pitch1, yaw1, time2, roll2, pitch2, yaw2


if __name__ == '__main__':
    # BiasIMUoneDay(date='2019-10-02')
    # BiasIMU_DailyMean(begin='2019-10-01', end='2020-12-31', sat=SatID.D)
    # plot_BiasIMU_DailyMean(begin='2019-10-01', end='2020-12-31', sat=SatID.D)
    PointingError(date='2019-01-01', sat=SatID.C)
