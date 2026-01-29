"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/20
@Description:
"""
import Quaternion

from GetInstrument import GetInstrument_L1A, GetInstrument_L1B, Mission, SatID
from kalman import GetFusionRes
import matplotlib.pyplot as plt
import numpy as np
from Quaternion import Quat
from PointingAnalysis import Pointing


def SCAgapAndFlip():
    from pysrc.SCA1Ato1B import SCA1Ato1B, SCAinterOption, SCAID

    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    date = '2018-12-02'
    id = SCAID.No_2

    SCA1Ato1B.ConfigSCF2SRF()

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
    # IMU1A = L1A.getIMU(SatID.C)
    SCA1A = L1A.getSCA(SatID.C)

    qw = SCA1A[id.name][2]
    qx = SCA1A[id.name][3]
    qy = SCA1A[id.name][4]
    qz = SCA1A[id.name][5]

    t = SCA1A[id.name][0] - SCA1A[id.name][0][0] + 1e-6 * SCA1A[id.name][1]

    q1 = qw
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, q1)
    ax.set(ylabel='q.w')
    ax.set_title('Eclipsed time since the first epoch [sec]')

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    # L1B.getIMU(SatID.C)
    TIM1B = L1B.getTIM(SatID.C)
    CLK1B = L1B.getCLK(SatID.C)

    SCA = SCA1Ato1B(satID=SatID.C, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)
    SCA.configInterp(interpOption=SCAinterOption.TwoHZ)
    sca = SCA.produce_combine_last(mission=Mission.GRACE_FO)

    q2 = sca[id.name][2]
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(t, q2)
    ax.set(ylabel='q.w')
    ax.set_title('Eclipsed time since the first epoch [sec]')

    plt.show()


def FusionResult1():
    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    date = '2018-12-02'
    sat = SatID.C

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    SCA = L1B.getSCA(sat=sat)

    lr = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    fus = lr.getFusion(sat=sat)

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(SCA[0], SCA[2])
    ax.set(ylabel='q.w')
    ax.set_title('Eclipsed time since the first epoch [sec]')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(fus[0], -fus[2])
    ax.set(ylabel='q.w')
    ax.set_title('Eclipsed time since the first epoch [sec]')

    plt.show()

    pass


def FusionResult2():
    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    date = '2019-01-01'
    sat = SatID.D

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    SCA = L1B.getSCA(sat=sat)

    lr = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    fus = lr.getFusion(sat=sat)

    # np.where()
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

    '''Quaternion'''
    data1_d = Quaternion.normalize(data1_q[[1, 2, 3, 0], :].T)
    data2_d = Quaternion.normalize(data2_q[[1, 2, 3, 0], :].T)

    data1_d = Quat(q=data1_d)
    data2_d = Quat(q=data2_d)

    diff_q = data1_d.dq(q2=data2_d)
    # diff_d = Quat(q=diff_q)

    ax = fig.add_subplot(4, 2, 1)
    ax.plot(time, diff_q.q[:, 3])
    ax.set(ylabel='Quaternion.w')

    ax = fig.add_subplot(4, 2, 3)
    ax.plot(time, diff_q.q[:, 0])
    ax.set(ylabel='Quaternion.x')

    ax = fig.add_subplot(4, 2, 5)
    ax.plot(time, diff_q.q[:, 1])
    ax.set(ylabel='Quaternion.y')

    ax = fig.add_subplot(4, 2, 7)
    ax.plot(time, diff_q.q[:, 2])
    ax.set(ylabel='Quaternion.z')
    ax.set(xlabel='Eclipsed time since the first epoch [sec]')

    '''Pitch, roll, yaw'''

    # pitch = data1_d.pitch
    # roll = data1_d.roll
    # yaw = data1_d.yaw

    # pitch = data1_d.pitch - data2_d.pitch
    # roll = data1_d.roll - data2_d.roll
    # yaw = data1_d.yaw - data2_d.yaw
    #
    pitch = diff_q.pitch
    roll = diff_q.roll
    yaw = diff_q.yaw

    pitch[pitch > 180] -= 360
    roll[roll > 180] -= 360
    yaw[yaw > 180] -= 360
    pitch[pitch < -180] += 360
    roll[roll < -180] += 360
    yaw[yaw < -180] += 360

    ax = fig.add_subplot(4, 2, 2)
    ax.plot(time, pitch, 'g')
    ax.set(ylabel='Pitch [deg]')

    ax = fig.add_subplot(4, 2, 4)
    ax.plot(time, roll, 'g')
    ax.set(ylabel='Roll [deg]')

    ax = fig.add_subplot(4, 2, 6)
    ax.plot(time, yaw, 'g')
    ax.set(ylabel='Yaw [deg]')
    ax.set(xlabel='Eclipsed time since the first epoch [sec]')

    plt.show()

    pass


def FusionResultPSD1():
    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    date = '2019-01-02'
    sat = SatID.D

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    SCA = L1B.getSCA(sat=sat)

    lr = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    fus = lr.getFusion(sat=sat)

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

    ax = fig.add_subplot(2, 1, 1)
    index = 3
    px2, f2 = ax.psd(x=data2_q[0], Fs=1, NFFT=len(data2_q[index]), color='b')
    px1, f1 = ax.psd(x=data1_q[0], Fs=1, NFFT=len(data1_q[index]), color='g')
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    px1 = np.sqrt(2 * px1 * 1)
    px2 = np.sqrt(2 * px2 * 1)

    ax = fig.add_subplot(2, 1, 2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(f2, px2, color='b', label='Fusion-SCA')
    ax.plot(f1, px1, color='g', label='JPL-SCA')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    ax.legend()

    plt.show()


def PointingError(date=None, sat=SatID.D):
    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    # date = '2019-01-06'
    # sat = SatID.D

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    SCA = L1B.getSCA(sat=sat)

    lr = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    fus = lr.getFusion(sat=sat)

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
    ax.plot(time2, pitch2, label='SCA-only')
    ax.plot(time1, pitch1, label='JPL-pitch')
    ax.plot(time, pitch, label='Fusion-pitch')
    ax.plot(time, pitch - pitch1, label='Diff-pitch')
    ax.set_title('JPL-SCA pointing error')
    ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('[rad]')
    ax.legend()
    ax.set_ylim([-1e-3, 1e-3])
    #
    plt.show()
    return time1, roll1, pitch1, yaw1, time2, roll2, pitch2, yaw2


def AngularVelocity():
    plt.style.use(['science', 'grid'])
    fig = plt.figure()
    w = np.load('w.npy')

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(np.rad2deg(w[:, 0]))

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(np.rad2deg(w[:, 1]))

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(np.rad2deg(w[:, 2]))
    plt.show()


if __name__ == '__main__':
    # SCAgapAndFlip()
    # FusionResult2()
    # FusionResultPSD1()
    PointingError(date='2019-01-01', sat=SatID.C)
    # AngularVelocity()
