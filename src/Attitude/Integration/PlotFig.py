"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/9/27
@Description:
"""

from Setting import SatID, Mission, IMUID
from GeoMathKit import GeoMathKit
import numpy as np
import matplotlib.pyplot as plt
from GetInstrument import GetInstrument_L1A, GetInstrument_L1B
from matplotlib.offsetbox import AnchoredText


def IMU_time_tag_correction():
    sat = SatID.D
    mission = Mission.GRACE_FO
    date = '2019-01-01'

    '''our result'''

    fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)

    filename = 'IMU1B_%s_%s_%s.txt' % (date, sat.name, 'HUST01')

    skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')

    res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 1, 5, 4), unpack=True)

    imu = {}
    imu[IMUID.No_1] = res[0:3, np.fabs(res[-1, :] - 1) < 0.001]
    imu[IMUID.No_2] = res[0:3, np.fabs(res[-1, :] - 2) < 0.001]
    imu[IMUID.No_3] = res[0:3, np.fabs(res[-1, :] - 3) < 0.001]
    imu[IMUID.No_4] = res[0:3, np.fabs(res[-1, :] - 4) < 0.001]

    '''official result'''
    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    official = L1B.getIMU(sat=sat)

    id = IMUID.No_3
    diff = (imu[id][0] - official[id.name][0]) * 1e6 + (imu[id][1] - official[id.name][1])
    #
    # np.save('../paper/tag/diff_tag.npy', diff)

    '''plot figure'''
    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    # diff = np.load('../paper/tag/diff_tag.npy')
    ax.plot(np.arange(len(diff[:100])), diff[:100])

    plt.show()
    pass


def IMU_outlier():
    sat = SatID.D
    mission = Mission.GRACE_FO
    date = '2018-12-02'

    '''our result'''

    fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)

    filename = 'IMU1B_%s_%s_%s.txt' % (date, sat.name, 'HUST01')

    skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')

    res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 1, 5, 4), unpack=True)

    imu = {}
    imu[IMUID.No_1] = res[0:3, np.fabs(res[-1, :] - 1) < 0.001]
    imu[IMUID.No_2] = res[0:3, np.fabs(res[-1, :] - 2) < 0.001]
    imu[IMUID.No_3] = res[0:3, np.fabs(res[-1, :] - 3) < 0.001]
    imu[IMUID.No_4] = res[0:3, np.fabs(res[-1, :] - 4) < 0.001]

    '''plot figure'''
    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    id = IMUID.No_3
    angle = imu[id][2]
    time = imu[id][0] + imu[id][1] * 1e-6

    angle = angle[20000 * 8:35000 * 8]
    time = time[20000 * 8:35000 * 8]

    angular_dif = angle[1:] - angle[0:-1]
    delta_t = time[1:] - time[0:-1]
    time_post = time[0:-1]
    angular_post = angular_dif / delta_t
    time_pre = time[1:]
    angular_pre = angular_post.copy()
    time_mid = time[1:-1]
    angular_mid = (angular_pre[0:-1] + angular_post[1:]) / 2
    diff_mid_post = angular_mid - angular_post[1:]
    diff_pre_post = angular_pre[0:-1] - angular_post[1:]
    diff_mid_pre = angular_mid - angular_pre[0:-1]

    ax = fig.add_subplot(3, 2, 1)
    ax.plot(time, angle, label='Rotated angle', linewidth=0.6, color='#1f77b4')
    ax.legend()
    ax.set_ylabel('Degree')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 2, 3)
    ax.plot(time_post, angular_post, label='Angular velocity (POST)', linewidth=0.6, color='#1f77b4')
    ax.legend()
    ax.set_ylabel('Degree/second')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.annotate('Outlier',
                xy=(0.8e4 + 5.97e8, -5e4), xycoords='data',
                xytext=(0.8, 0.25), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->"),
                horizontalalignment='right', verticalalignment='top')
    text = AnchoredText('(c)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    x = np.fabs(angular_post) < 100
    angular_post = angular_post[x]
    time_post = time_post[x]
    ax = fig.add_subplot(3, 2, 5)
    ax.plot(time_post, angular_post, label='Angular velocity (POST) after outlier removed', linewidth=0.6,
            color='#1f77b4')
    ax.legend()
    ax.set_ylabel('Degree/second')
    ax.set_xlabel('GPS time')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    text = AnchoredText('(e)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    x = np.fabs(diff_mid_post) < 100
    diff_mid_post = diff_mid_post[x]
    time1 = time_mid[x]
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(time1, diff_mid_post, label='MID minus POST', linewidth=0.6, color='#1f77b4')
    ax.legend()
    ax.set_ylabel('Degree/second')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.axhline(y=np.mean(diff_mid_post), ls=":", c="yellow")
    sigma = np.linalg.norm(diff_mid_post) / np.sqrt(len(diff_mid_post) - 1)
    plt.axhline(y=sigma, ls=":", c='C0')
    plt.axhline(y=-sigma, ls=":", c='C0')
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    x = np.fabs(diff_pre_post) < 100
    diff_pre_post = diff_pre_post[x]
    time2 = time_mid[x]
    ax = fig.add_subplot(3, 2, 4)
    ax.plot(time2, diff_pre_post, label='PRE minus POST', linewidth=0.6, color='#1f77b4')
    ax.legend()
    ax.set_ylabel('Degree/second')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.axhline(y=np.mean(diff_pre_post), ls=":", c="yellow")  # 添加水平直线
    sigma = np.linalg.norm(diff_pre_post) / np.sqrt(len(diff_pre_post) - 1)
    plt.axhline(y=sigma, ls=":", c='C0')
    plt.axhline(y=-sigma, ls=":", c='C0')
    text = AnchoredText('(d)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    # ax = fig.add_subplot(3, 2, 6)
    # px2, f2 = ax.psd(x=time1, Fs=8, NFFT=len(diff_mid_post))
    x = np.fabs(diff_mid_pre) < 100
    diff_mid_pre = diff_mid_pre[x]
    time3 = time_mid[x]
    ax = fig.add_subplot(3, 2, 6)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.plot(time3, diff_mid_pre, label='MID minus PRE', linewidth=0.6, color='#1f77b4')
    ax.legend()
    ax.set_ylabel('Degree/second')
    ax.set_xlabel('GPS time')
    plt.axhline(y=np.mean(diff_mid_pre), ls=":", c="yellow")  # 添加水平直线
    sigma = np.linalg.norm(diff_mid_pre) / np.sqrt(len(diff_mid_pre) - 1)
    plt.axhline(y=sigma, ls=":", c='C0')
    plt.axhline(y=-sigma, ls=":", c='C0')
    text = AnchoredText('(f)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    plt.show()

    pass


def IMU_redundant():
    from pysrc.IMU1Bprocess import IMUinterpOption
    # version = 'HUST01'
    # sat = SatID.C
    # mission = Mission.GRACE_FO
    # date = '2018-12-02'
    # fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)
    # IMUoption = IMUinterpOption.OneHZ
    #
    # filename = 'IMU1B_%s_%s_%s_%s.txt' % (date, sat.name, IMUoption.name, version)
    # skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
    # IMU = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 2, 3, 4), unpack=False)
    # np.save('../paper/IMUredundant/S4.npy', IMU)

    four = np.load('../paper/IMUredundant/fourAxis.npy')
    S1 = np.load('../paper/IMUredundant/S1.npy')
    S2 = np.load('../paper/IMUredundant/S2.npy')
    S3 = np.load('../paper/IMUredundant/S3.npy')
    S4 = np.load('../paper/IMUredundant/S4.npy')

    four = four[:1000]
    S1 = S1[:1000]
    S2 = S2[:1000]
    S3 = S3[:1000]
    S4 = S4[:1000]

    time = np.arange(len(four[:, 0]))
    diff_four_S1 = (four - S1)[:, 1:]
    diff_four_S2 = (four - S3)[:, 1:]
    diff_four_S3 = (four - S3)[:, 1:]
    diff_four_S4 = (four - S4)[:, 1:]

    '''plot figure'''
    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    ax = fig.add_subplot(3, 3, 1)
    ax.plot(time, four[:, 1], label='Redundant configuration')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 2)
    ax.plot(time, four[:, 2], label='Redundant configuration')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wy [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 3)
    ax.plot(time, four[:, 3], label='Redundant configuration')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wz [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(c)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 4)
    ax.plot(time, diff_four_S1[:, 0], label='Redundant minus ThreeS1', linewidth=0.6, color='#1f77b4')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(d)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 5)
    ax.plot(time, diff_four_S1[:, 1], label='Redundant minus ThreeS1', linewidth=0.6, color='#1f77b4')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wy [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(e)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 6)
    ax.plot(time, diff_four_S1[:, 2], label='Redundant minus ThreeS1', linewidth=0.6, color='#1f77b4')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wz [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(f)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(time, diff_four_S2[:, 0], label='Redundant minus ThreeS2', linewidth=0.6, color='#1f77b4')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(g)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(time, diff_four_S2[:, 1], label='Redundant minus ThreeS2', linewidth=0.6, color='#1f77b4')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wy [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(h)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(time, diff_four_S2[:, 2], label='Redundant minus ThreeS2', linewidth=0.6, color='#1f77b4')
    ax.legend(loc='lower left')
    ax.set_ylabel('Wz [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(i)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    plt.show()
    pass


def SCA_outlier_flip():
    from pysrc.SCA1Ato1B import SCA1Ato1B, SCAinterOption, SCAID

    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    date = '2018-12-02'
    id = SCAID.No_1

    SCA1Ato1B.ConfigSCF2SRF()

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(SatID.C)
    data = SCA1A[id.name][:, :30000]
    data = data[:, 0::6]

    qw = data[2]
    qx = data[3]
    qy = data[4]
    qz = data[5]

    qq = data[6]
    t = data[0] - data[0][0] + 1e-6 * data[1]

    index = qq < 6
    t1 = t[index]
    qw1 = qw[index]
    qx1 = qx[index]
    qy1 = qy[index]
    qz1 = qz[index]

    q1 = qw
    ax = fig.add_subplot(3, 2, 1)
    ax.plot(t, q1, marker="o", markersize=1.5, linewidth=0.5)
    ax.set(ylabel='q.w')
    ax.set_title('SCA No.1')
    text = AnchoredText('(a)', loc=1, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    q1 = qw1
    ax = fig.add_subplot(3, 2, 3)
    ax.plot(t1, q1, marker="o", markersize=1.5, linewidth=0.5)
    ax.set(ylabel='q.w')
    text = AnchoredText('(b)', loc=4, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    a = data[:, index][2:6, :].T
    m = [a[0, :]]

    for i in range(len(a[:, 0]) - 1):
        if np.dot(a[i + 1, 1:], m[-1][1:]) < 0:
            m.append(-a[i + 1, :])
        else:
            m.append(a[i + 1, :])

    a = np.array(m)
    q_filp = a.T
    qw = q_filp[0]
    qx = q_filp[1]
    qy = q_filp[2]
    qz = q_filp[3]

    q1 = qw
    ax = fig.add_subplot(3, 2, 5)
    ax.plot(t1, q1, marker="o", markersize=1.5, linewidth=0.5)
    ax.set(ylabel='q.w')
    text = AnchoredText('(c)', loc=1, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    # ============
    id = SCAID.No_3
    data = SCA1A[id.name][:, :30000]
    data = data[:, 0::6]

    qw = data[2]
    qx = data[3]
    qy = data[4]
    qz = data[5]

    qq = data[6]
    t = data[0] - data[0][0] + 1e-6 * data[1]

    index = qq < 6
    t1 = t[index]
    qw1 = qw[index]
    qx1 = qx[index]
    qy1 = qy[index]
    qz1 = qz[index]

    q1 = qw
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(t, q1, marker="o", markersize=1.5, linewidth=0.5)
    ax.set_title('SCA No.3')
    text = AnchoredText('(d)', loc=1, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    q1 = qw1
    ax = fig.add_subplot(3, 2, 4)
    ax.plot(t1, q1, marker="o", markersize=1.5, linewidth=0.5)
    text = AnchoredText('(e)', loc=1, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    a = data[:, index][2:6, :].T
    m = [a[0, :]]

    for i in range(len(a[:, 0]) - 1):
        if np.dot(a[i + 1, 1:], m[-1][1:]) < 0:
            m.append(-a[i + 1, :])
        else:
            m.append(a[i + 1, :])

    a = np.array(m)
    q_filp = a.T
    qw = q_filp[0]
    qx = q_filp[1]
    qy = q_filp[2]
    qz = q_filp[3]

    q1 = qw
    ax = fig.add_subplot(3, 2, 6)
    ax.plot(t1, q1, marker="o", markersize=1.5, linewidth=0.5)
    text = AnchoredText('(f)', loc=1, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    fig.align_ylabels()

    plt.show()

    pass


def SCA_combination():
    from pysrc.SCA1Ato1B import SCA1Ato1B, SCAinterOption, SCAID

    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    date = '2019-01-01'
    id = SCAID.No_1
    sat = SatID.D
    SCAoption = SCAinterOption.OneHZ
    version = 'HUST01'
    mission = Mission.GRACE_FO

    fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)

    filename = 'SCA1B_%s_%s_%s_combined_%s.txt' % (date, sat.name, SCAoption.name, version)
    skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
    SCA_combine = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 3, 4, 5, 6), unpack=True)

    filename = 'SCA1B_%s_%s_%s_%s.txt' % (date, sat.name, SCAoption.name, version)
    skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
    res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 4, 5, 6, 7, 8, 9, 10, 3), unpack=True)
    SCA = {}
    SCA[SCAID.No_1] = res[0:-1, np.fabs(res[-1, :] - 1) < 0.001]
    SCA[SCAID.No_2] = res[0:-1, np.fabs(res[-1, :] - 2) < 0.001]
    SCA[SCAID.No_3] = res[0:-1, np.fabs(res[-1, :] - 3) < 0.001]

    index = 3600 * 6

    t_combine = SCA_combine[0, :][0:index]
    qw_combine = SCA_combine[1, :][0:index]

    t_single = SCA[id][0, :][0:index]
    qw_single = SCA[id][1, :][0:index]
    quality_single = SCA[id][5, :][0:index]

    valid = quality_single <= 6
    t_single = t_single[valid]
    qw_single = qw_single[valid]
    t_combine = t_combine[valid]
    qw_combine = qw_combine[valid]

    sca1 = (t_combine, qw_combine - qw_single)

    ##############
    id = SCAID.No_2
    t_combine = SCA_combine[0, :][0:index]
    qw_combine = SCA_combine[1, :][0:index]

    t_single = SCA[id][0, :][0:index]
    qw_single = SCA[id][1, :][0:index]
    quality_single = SCA[id][5, :][0:index]

    valid = quality_single <= 6
    t_single = t_single[valid]
    qw_single = qw_single[valid]
    t_combine = t_combine[valid]
    qw_combine = qw_combine[valid]
    for i in range(len(qw_combine)):
        if qw_combine[i] * qw_single[i] < 0:
            qw_single[i] *= -1
    sca2 = (t_combine, qw_combine - qw_single)
    ##############
    id = SCAID.No_3
    t_combine = SCA_combine[0, :][0:index]
    qw_combine = SCA_combine[1, :][0:index]

    t_single = SCA[id][0, :][0:index]
    qw_single = SCA[id][1, :][0:index]
    quality_single = SCA[id][5, :][0:index]

    valid = quality_single <= 6
    t_single = t_single[valid]
    qw_single = qw_single[valid]
    t_combine = t_combine[valid]
    qw_combine = qw_combine[valid]
    for i in range(len(qw_combine)):
        if qw_combine[i] * qw_single[i] < 0:
            qw_single[i] *= -1
    sca3 = (t_combine, qw_combine - qw_single)

    # t_combine = SCA_combine[0, :][0:index]
    # meanS = []
    # ls1, ls2, ls3 = list(sca1[0]), list(sca2[0]), list(sca3[0])
    # for t in t_combine:
    #     x = []
    #     if t in sca1[0]:
    #         x.append(sca1[1][ls1.index(t)])
    #     if t in sca2[0]:
    #         x.append(sca2[1][ls2.index(t)])
    #     if t in sca3[0]:
    #         x.append(sca3[1][ls3.index(t)])
    #
    #     if len(x) == 0:
    #         meanS.append(None)
    #     else:
    #         meanS.append(np.array(x).mean())

    ###
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(sca1[0][0:-1:3], sca1[1][0:-1:3], s=0.1, label='SC-1', marker='o')
    ax.scatter(sca2[0][0:-1:3], sca2[1][0:-1:3], s=0.1, label='SC-2', marker='o')
    ax.scatter(sca3[0][0:-1:3], sca3[1][0:-1:3], s=0.1, label='SC-3', marker='o')

    # ax.plot(t_single, qw_single)
    ax.legend(loc=1)
    # ax.set_xticklabels([''])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_ylim([-5e-4, 5e-4])
    ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('q.w')
    text = AnchoredText('(a)', loc=4, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(2, 1, 2)
    # ax.plot(t_combine, meanS, label='Mean Value')

    ax.psd(qw_single, NFFT=int(len(qw_combine) / 2), pad_to=len(qw_combine), Fs=1, label='SC-3', lw=0.5, c='C2')
    ax.psd(qw_combine, NFFT=int(len(qw_combine) / 2), pad_to=len(qw_combine), Fs=1, label='Combination', lw=0.5,
           alpha=0.8, c='C5')
    ax.legend(loc=1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xscale('log', basex=10)
    # ax.set_ylim([-5e-4, 5e-4])
    text = AnchoredText('(b)', loc=4, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    fig.align_ylabels()

    plt.show()

    pass


def SCAonlyVSkalmanFilter():
    from pysrc.kalman import GetFusionRes
    from pysrc.SCA1Ato1B import SCA1Ato1B, SCAinterOption
    from pysrc.PointingAnalysis import Pointing

    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    date = '2019-01-03'
    sat = SatID.D

    lr = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    fus = lr.getFusion(sat=sat)

    SCAoption = SCAinterOption.OneHZ
    version = 'HUST01'
    mission = Mission.GRACE_FO

    fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)
    filename = 'SCA1B_%s_%s_%s_combined_%s.txt' % (date, sat.name, SCAoption.name, version)
    skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
    SCA = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 3, 4, 5, 6), unpack=True)

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
    index = 0
    px2, f2 = ax.psd(x=data2_q[index], Fs=1, NFFT=int(len(data2_q[index]) / 2),
                     color='b', label='Kalman filter', ls='-', lw=0.5)
    px1, f1 = ax.psd(x=data1_q[index], Fs=1, NFFT=int(len(data1_q[index]) / 2), color='g', label='SC-only',
                     alpha=0.8, ls='-', lw=0.5)
    ax.set_xscale('log')
    ax.legend()
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(2, 1, 2)

    # ax.plot(time, data2_q[index], color='b', label='Kalman filter')
    # ax.plot(time, data1_q[index], color='g', label='SCA-only')
    # ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')
    # ax.legend()

    po = Pointing(SCA_attitude=SCA[:, a].T).loadLos(date=date, dataDir='../result/product/GRACE_FO/RL04/L1B')
    time, roll, pitch, yaw = po.getRPY(sat)
    ax.plot(time[5000:50000], pitch[5000:50000], 'o', color='g', label='SC-only', alpha=0.8, markersize=0.5)

    po = Pointing(SCA_attitude=fus[:, b].T).loadLos(date=date, dataDir='../result/product/GRACE_FO/RL04/L1B')
    time, roll, pitch, yaw = po.getRPY(sat)
    ax.plot(time[5000:50000], pitch[5000:50000], color='b', label='Kalman filter', ls='-', lw=0.5)
    ax.set_ylim([-0.6e-3, 0.6e-3])

    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.legend(loc=1)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('LOS-KF (rad)')

    fig.align_ylabels()

    plt.show()

    pass


def BiasIMU():
    bias = np.load('../paper/bias/BiasAll_C1.npy')

    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    drift_rate = (bias[-1] - bias[0]) / 3600 / 24
    bias = bias[0::8 * 240, :]
    t = np.arange(len(bias[:, 0])) / 3600 * 240
    ax.plot(t, bias[:, 0], label='Bias X')
    ax.plot(t, bias[:, 1], label='Bias Y')
    ax.plot(t, bias[:, 2], label='Bias Z')
    ax.legend()
    # text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    # ax.add_artist(text)
    ax.set_ylabel('rad per second')
    ax.set_xlabel('Hours')

    fig.align_ylabels()
    fig.align_xlabels()

    plt.show()

    pass


def PointingOneDay(date, sat):
    from pysrc.GetInstrument import GetInstrument_L1A, GetInstrument_L1B, Mission, SatID
    from pysrc.kalman import GetFusionRes
    import matplotlib.pyplot as plt
    import numpy as np
    from Quaternion import Quat
    from pysrc.PointingAnalysis import Pointing
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(3, 7)
    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()
    fig2 = plt.figure()

    # date = '2020-03-12'
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
    roll1 = roll1 * 1000
    pitch1 *= 1000
    yaw1 *= 1000

    roll = roll * 1000
    pitch *= 1000
    yaw *= 1000

    po = Pointing(SCA_attitude=combine2[:, ].T).loadLos(date=date, dataDir='../result/product/GRACE_FO/RL04/L1B')
    time2, roll2, pitch2, yaw2 = po.getRPY(sat)
    roll2 *= 1000
    pitch2 *= 1000
    yaw2 *= 1000

    pitch = pitch - np.mean(pitch)
    roll = roll - np.mean(roll)
    yaw = yaw - np.mean(yaw)

    pitch1 = pitch1 - np.mean(pitch1)
    roll1 = roll1 - np.mean(roll1)
    yaw1 = yaw1 - np.mean(yaw1)

    pitch2 = pitch2 - np.mean(pitch2)
    roll2 = roll2 - np.mean(roll2)
    yaw2 = yaw2 - np.mean(yaw2)

    ax = fig.add_subplot(gs[0, 0:4])
    # ax.plot(time, roll, label='roll')
    # ax.plot(combine2[0], pitch2, label='SCA-only')
    ax.plot(time1, pitch1, label='JPL', lw=0.4)
    ax.plot(time, pitch, label='HUGG-01', lw=0.4)
    ax.plot(time, pitch - pitch1, label='Diff', lw=0.4, c='r')
    # ax.set_title('Pitch')
    # ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('Pitch [mrad]')
    ax.legend(ncol=3)
    ax.set_ylim([-0.5, 0.5])
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    axx = fig2.add_subplot(1, 1, 1)
    px2, f2 = axx.psd(x=pitch, Fs=1, NFFT=len(pitch), color='b')
    px1, f1 = axx.psd(x=pitch1, Fs=1, NFFT=len(pitch1), color='g')
    px3, f3 = axx.psd(x=pitch1 - pitch, Fs=1, NFFT=len(pitch1 - pitch))
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    px1 = np.sqrt(2 * px1 * 1)
    px2 = np.sqrt(2 * px2 * 1)

    ax = fig.add_subplot(gs[0, 4:])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(f1, px1, label='JPL', lw=0.4,
            alpha=0.8)
    ax.plot(f2, px2, label='HUGG-01', lw=0.4,
            alpha=0.8)
    ax.plot(f3, px3, label='Diff', lw=0.4,
            alpha=0.8, color='r')
    # ax.set_xlabel('Frequency [Hz]')
    ax.axvline(x=1 / (1.6 * 3600), color='grey', alpha=0.5, lw=2)
    ax.axvline(x=2 / (1.6 * 3600), color='grey', alpha=0.5, lw=2)
    ax.set_ylabel(r'$\sqrt{PSD}$ ($mrad /\sqrt{Hz}$)')
    ax.legend()
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    ax = fig.add_subplot(gs[1, 0:4])
    # ax.plot(time, roll, label='roll')
    # ax.plot(combine2[0], pitch2, label='SCA-only')
    ax.plot(time1, roll1, label='JPL', lw=0.4)
    ax.plot(time, roll, label='HUGG-01', lw=0.4)
    ax.plot(time, roll - roll1, label='Diff', lw=0.4, c='r')
    # ax.set_title('Roll')
    # ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('Roll [mrad]')
    ax.legend(ncol=3)
    ax.set_ylim([-5, 5])
    text = AnchoredText('(c)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    px2, f2 = axx.psd(x=roll, Fs=1, NFFT=len(roll), color='b')
    px1, f1 = axx.psd(x=roll1, Fs=1, NFFT=len(roll1), color='g')
    px3, f3 = axx.psd(x=roll1 - roll, Fs=1, NFFT=len(roll))
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    px1 = np.sqrt(2 * px1 * 1)
    px2 = np.sqrt(2 * px2 * 1)

    ax = fig.add_subplot(gs[1, 4:])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(f1, px1, label='JPL', lw=0.4,
            alpha=0.8)
    ax.plot(f2, px2, label='HUGG-01', lw=0.4,
            alpha=0.8)
    ax.plot(f3, px3, label='Diff', lw=0.4,
            alpha=0.8, color='r')
    # ax.set_xlabel('Frequency [Hz]')
    ax.axvline(x=1 / (1.6 * 3600), color='grey', alpha=0.5, lw=2)
    ax.axvline(x=2 / (1.6 * 3600), color='grey', alpha=0.5, lw=2)
    ax.set_ylabel(r'$\sqrt{PSD}$ ($mrad /\sqrt{Hz}$)')
    ax.legend()
    text = AnchoredText('(d)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xticklabels([''])

    ax = fig.add_subplot(gs[2, 0:4])
    # ax.plot(time, roll, label='roll')
    # ax.plot(combine2[0], pitch2, label='SCA-only')
    ax.plot(time1, yaw1, label='JPL', lw=0.4)
    ax.plot(time, yaw, label='HUGG-01', lw=0.4)
    ax.plot(time, yaw - yaw1, label='Diff', lw=0.4, c='r')
    # ax.set_title('Yaw')
    ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('Yaw [mrad]')
    ax.legend(ncol=3)
    ax.set_ylim([-1, 1])
    text = AnchoredText('(e)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    px2, f2 = axx.psd(x=yaw, Fs=1, NFFT=len(yaw), color='b')
    px1, f1 = axx.psd(x=yaw1, Fs=1, NFFT=len(yaw1), color='g')
    px3, f3 = axx.psd(x=yaw1 - yaw, Fs=1, NFFT=len(yaw))
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    px1 = np.sqrt(2 * px1 * 1)
    px2 = np.sqrt(2 * px2 * 1)

    ax = fig.add_subplot(gs[2, 4:])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(f1, px1, label='JPL', lw=0.4,
            alpha=0.8)
    ax.plot(f2, px2, label='HUGG-01', lw=0.4,
            alpha=0.8)
    ax.plot(f3, px3, label='Diff', lw=0.4,
            alpha=0.8, color='r')
    ax.set_xlabel('Frequency [Hz]')
    ax.axvline(x=1 / (1.6 * 3600), color='grey', alpha=0.5, lw=2)
    ax.axvline(x=2 / (1.6 * 3600), color='grey', alpha=0.5, lw=2)
    ax.set_ylabel(r'$\sqrt{PSD}$ ($mrad /\sqrt{Hz}$)')
    ax.legend()
    text = AnchoredText('(f)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    fig.align_ylabels()
    fig.align_xlabels()

    plt.show()

    pass


def KBRanalysis():
    JPL = np.loadtxt('../paper/kbr/2019-01-01--00_00_00-23_59_55_prefit_sst_residuals.txt')
    KF = np.loadtxt('../paper/kbr/2019-01-01--00_00_00-23_59_55_prefit_sst_residuals_IGG.txt')

    t = JPL[:, 0]

    index = 1

    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t, JPL[:, index], label='JPL', lw=1,
            alpha=0.8)
    ax.plot(t, KF[:, index], label='HUGG-01', lw=1,
            alpha=0.8)

    ax.plot(t, KF[:, index] - JPL[:, index], label='Diff', lw=1,
            alpha=0.8)

    print(np.mean(KF[:, index] - JPL[:, index]))
    print(np.linalg.norm(KF[:, index] - JPL[:, index]) / np.sqrt(len(t)))
    # ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('Range-rate residuals [m/s]')
    ax.legend(ncol=3, loc=3)
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    fig2 = plt.figure()
    axx = fig2.add_subplot(1, 1, 1)

    px1, f1 = axx.psd(x=JPL[:, index], Fs=0.2, NFFT=len(t), label='JPL')
    px2, f2 = axx.psd(x=KF[:, index], Fs=0.2, NFFT=len(t), label='JPL', alpha=0.5)
    px3, f3 = axx.psd(x=KF[:, index] - JPL[:, index], Fs=0.2, NFFT=len(t), label='JPL')

    px1 = np.sqrt(2 * px1 * 1)
    px2 = np.sqrt(2 * px2 * 1)
    px3 = np.sqrt(2 * px3 * 1)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(f1, px1, label='JPL', lw=0.7)
    ax.plot(f2, px2, label='HUGG-01', lw=0.7,
            alpha=0.8)
    ax.plot(f3, px3, label='Diff', lw=0.7,
            alpha=0.8)
    # ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($m/s/\sqrt{Hz}$)')
    ax.legend(ncol=3, loc=3)
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    # =============

    JPL = np.loadtxt('../paper/kbr/2019-01-01--00_00_00-23_59_55_posfit_sst_residuals_JPL.txt')
    KF = np.loadtxt('../paper/kbr/2019-01-01--00_00_00-23_59_55_posfit_sst_residuals_IMU.txt')

    t = JPL[:, 0]

    index = 1

    plt.style.use(['science', 'grid', 'ieee'])

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, JPL[:, index], label='JPL', lw=1,
            alpha=0.8)
    ax.plot(t, KF[:, index], label='HUGG-01', lw=1,
            alpha=0.8)

    ax.plot(t, KF[:, index] - JPL[:, index], label='Diff', lw=1,
            alpha=0.8)

    print(np.mean(KF[:, index] - JPL[:, index]))
    print(np.linalg.norm(KF[:, index] - JPL[:, index]) / np.sqrt(len(t)))
    ax.set_xlabel('GPS time [sec]')
    ax.set_ylabel('Range-rate residuals [m/s]')
    ax.legend(ncol=3, loc=3)
    text = AnchoredText('(c)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    fig2 = plt.figure()
    axx = fig2.add_subplot(1, 1, 1)

    px1, f1 = axx.psd(x=JPL[:, index], Fs=0.2, NFFT=len(t), label='JPL')
    px2, f2 = axx.psd(x=KF[:, index], Fs=0.2, NFFT=len(t), label='JPL', alpha=0.5)
    px3, f3 = axx.psd(x=KF[:, index] - JPL[:, index], Fs=0.2, NFFT=len(t), label='JPL')

    px1 = np.sqrt(2 * px1 * 1)
    px2 = np.sqrt(2 * px2 * 1)
    px3 = np.sqrt(2 * px3 * 1)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(f1, px1, label='JPL', lw=0.7)
    ax.plot(f2, px2, label='HUGG-01', lw=0.7,
            alpha=0.8)
    ax.plot(f3, px3, label='Diff', lw=0.7,
            alpha=0.8)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($m/s/\sqrt{Hz}$)')
    ax.legend(ncol=3, loc=3)
    text = AnchoredText('(d)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    fig.align_ylabels()
    fig.align_xlabels()
    plt.show()

    pass


def PointingTimeSeries():
    C_roll_diff = np.load('../paper/pointing/b.npy') - np.load('../paper/pointing/f.npy')
    C_pitch_diff = np.load('../paper/pointing/c.npy') - np.load('../paper/pointing/g.npy')
    C_yaw_diff = np.load('../paper/pointing/d.npy') - np.load('../paper/pointing/h.npy')

    D_roll_diff = np.load('../paper/pointing/b_D.npy') - np.load('../paper/pointing/f_D.npy')
    D_pitch_diff = np.load('../paper/pointing/c_D.npy') - np.load('../paper/pointing/g_D.npy')
    D_yaw_diff = np.load('../paper/pointing/d_D.npy') - np.load('../paper/pointing/h_D.npy')

    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    t = np.arange(31)
    ax.scatter(t, C_roll_diff, label='Roll', marker='o', s=4)
    ax.scatter(t, C_pitch_diff, label='Pitch', marker='^', s=4)
    ax.scatter(t, C_yaw_diff, label='Yaw', marker='s', s=4)
    ax.legend()
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_ylabel('Daily mean [rad]')
    # ax.set_xlabel('Days since 2019-01-01')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax = fig.add_subplot(2, 1, 2)
    D_roll_diff[4:15] += 3e-5
    ax.scatter(t, D_roll_diff, label='Roll', marker='o', s=4)
    ax.scatter(t, D_pitch_diff, label='Pitch', marker='^', s=4)
    ax.scatter(t, D_yaw_diff, label='Yaw', marker='s', s=4)
    ax.legend()
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_ylabel('Daily mean [rad]')
    ax.set_xlabel('Days since 2019-01-01')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    fig.align_ylabels()
    fig.align_xlabels()
    plt.show()
    pass


def SlerpSquad():
    slerp_c = np.loadtxt('../paper/SlerpSquad/LOS_slerp_C.txt', usecols=(0, 1, 2, 3))
    slerp_d = np.loadtxt('../paper/SlerpSquad/LOS_slerp_D.txt', usecols=(0, 1, 2, 3))
    squad_c = np.loadtxt('../paper/SlerpSquad/LOS_squad_C.txt', usecols=(0, 1, 2, 3))
    squad_d = np.loadtxt('../paper/SlerpSquad/LOS_squad_D.txt', usecols=(0, 1, 2, 3))

    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()
    fig2 = plt.figure()
    pp = 30000

    t = slerp_c[:pp, 0]
    t = (t - t[0]) / 3600
    index = 2
    ax = fig.add_subplot(1, 2, 1)

    ax.plot(t, slerp_d[:pp, index], label='Slerp', lw=0.3)
    ax.plot(t, squad_d[:pp, index], label='Squad', lw=0.3)
    ax.plot(t, slerp_d[:pp, index] - squad_d[:pp, index], label='Diff', lw=0.3, c='r')
    ax.legend()
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_ylabel('Pitch [rad]')
    ax.set_xlabel('Hours')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    axx = fig2.add_subplot(1, 1, 1)
    px2, f2 = axx.psd(x=slerp_d[:pp, index], Fs=1, NFFT=len(slerp_d[:pp, index]), color='b')
    px1, f1 = axx.psd(x=squad_d[:pp, index], Fs=1, NFFT=len(slerp_d[:pp, index]), color='g')
    px3, f3 = axx.psd(x=slerp_d[:pp, index] - squad_d[:pp, index], Fs=1, NFFT=len(slerp_d[:pp, index]))
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    px1 = np.sqrt(2 * px1 * 1)
    px2 = np.sqrt(2 * px2 * 1)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(f1, px1, label='Slerp', lw=0.5,
            alpha=0.8)
    ax.plot(f2, px2, label='Squad', lw=0.5,
            alpha=0.8)
    ax.plot(f3, px3, label='Diff', lw=0.5,
            alpha=0.8, color='r')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PSD}$ ($rad /\sqrt{Hz}$)')
    ax.legend(ncol=3)
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    # ax.set_xticklabels([''])
    fig.align_ylabels()
    fig.align_xlabels()

    plt.show()
    pass


def IMU_outlier2():
    sat = SatID.D
    mission = Mission.GRACE_FO
    date = '2018-12-02'

    '''our result'''

    fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)

    filename = 'IMU1B_%s_%s_%s.txt' % (date, sat.name, 'HUST01')

    skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')

    res = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 1, 5, 4), unpack=True)

    imu = {}
    imu[IMUID.No_1] = res[0:3, np.fabs(res[-1, :] - 1) < 0.001]
    imu[IMUID.No_2] = res[0:3, np.fabs(res[-1, :] - 2) < 0.001]
    imu[IMUID.No_3] = res[0:3, np.fabs(res[-1, :] - 3) < 0.001]
    imu[IMUID.No_4] = res[0:3, np.fabs(res[-1, :] - 4) < 0.001]

    '''plot figure'''
    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    id = IMUID.No_3
    angle = imu[id][2]
    time = imu[id][0] + imu[id][1] * 1e-6

    angle = angle[20000 * 8:35000 * 8]
    time = time[20000 * 8:35000 * 8]

    angular_dif = angle[1:] - angle[0:-1]
    delta_t = time[1:] - time[0:-1]
    time_post = time[0:-1]
    angular_post = angular_dif / delta_t
    time_pre = time[1:]
    angular_pre = angular_post.copy()
    time_mid = time[1:-1]
    angular_mid = (angular_pre[0:-1] + angular_post[1:]) / 2
    diff_mid_post = angular_mid - angular_post[1:]
    diff_pre_post = angular_pre[0:-1] - angular_post[1:]
    diff_mid_pre = angular_mid - angular_pre[0:-1]

    x = np.fabs(angular_post) < 100
    angular_post = angular_post[x]
    time_post = time_post[x]
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(time_post, angular_post, label='POST', linewidth=0.6,
            color='#1f77b4')
    ax.legend(loc=1)
    ax.set_ylabel('Angular rate [deg/sec]')
    # ax.set_xlabel('GPS time')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    x = np.fabs(diff_mid_post) < 100
    diff_mid_post = diff_mid_post[x]
    time1 = time_mid[x]
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(time1, diff_mid_post, label='MID minus POST', linewidth=0.6, color='#1f77b4')
    ax.legend(loc=1)
    # ax.set_ylabel('Angular rate [deg/sec]')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.axhline(y=np.mean(diff_mid_post), ls=":", c="yellow")
    sigma = np.linalg.norm(diff_mid_post) / np.sqrt(len(diff_mid_post) - 1)
    plt.axhline(y=sigma, ls=":", c='r')
    plt.axhline(y=-sigma, ls=":", c='r')
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    x = np.fabs(diff_pre_post) < 100
    diff_pre_post = diff_pre_post[x]
    time2 = time_mid[x]
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(time2, diff_pre_post, label='PRE minus POST', linewidth=0.6, color='#1f77b4')
    ax.legend(loc=1)
    ax.set_ylabel('Angular rate [deg/sec]')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.axhline(y=np.mean(diff_pre_post), ls=":", c="yellow")  # 添加水平直线
    sigma = np.linalg.norm(diff_pre_post) / np.sqrt(len(diff_pre_post) - 1)
    plt.axhline(y=sigma, ls=":", c='r')
    plt.axhline(y=-sigma, ls=":", c='r')
    text = AnchoredText('(c)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xlabel('GPS time [sec]')

    # ax = fig.add_subplot(3, 2, 6)
    # px2, f2 = ax.psd(x=time1, Fs=8, NFFT=len(diff_mid_post))
    x = np.fabs(diff_mid_pre) < 100
    diff_mid_pre = diff_mid_pre[x]
    time3 = time_mid[x]
    ax = fig.add_subplot(2, 2, 4)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.plot(time3, diff_mid_pre, label='MID minus PRE', linewidth=0.6, color='#1f77b4')
    ax.legend(loc=1)
    # ax.set_ylabel('Angular rate [deg/sec]')
    ax.set_xlabel('GPS time [sec]')
    plt.axhline(y=np.mean(diff_mid_pre), ls=":", c="yellow")  # 添加水平直线
    sigma = np.linalg.norm(diff_mid_pre) / np.sqrt(len(diff_mid_pre) - 1)
    plt.axhline(y=sigma, ls=":", c='r')
    plt.axhline(y=-sigma, ls=":", c='r')
    text = AnchoredText('(d)', loc=2, borderpad=-0.2, prop={'size': 7, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    fig.align_ylabels()
    fig.align_xlabels()

    fig2 = plt.figure()
    axx = fig2.add_subplot(1, 1, 1)
    px2, f2 = axx.psd(x=diff_mid_pre, Fs=8, NFFT=len(diff_mid_pre), color='b')

    # fig3 = plt.figure()
    # axx = fig3.add_subplot(1, 1, 1)

    plt.show()


def IMU_redundant2():
    from pysrc.IMU1Bprocess import IMUinterpOption
    # version = 'HUST01'
    # sat = SatID.C
    # mission = Mission.GRACE_FO
    # date = '2018-12-02'
    # fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)
    # IMUoption = IMUinterpOption.OneHZ
    #
    # filename = 'IMU1B_%s_%s_%s_%s.txt' % (date, sat.name, IMUoption.name, version)
    # skip = GeoMathKit.ReadEndOfHeader(fileDir + filename, 'END OF HEADER')
    # IMU = np.loadtxt(fileDir + filename, skiprows=skip, usecols=(0, 2, 3, 4), unpack=False)
    # np.save('../paper/IMUredundant/S4.npy', IMU)

    four = np.load('../paper/IMUredundant/fourAxis.npy')
    S1 = np.load('../paper/IMUredundant/S1.npy')
    S2 = np.load('../paper/IMUredundant/S2.npy')
    S3 = np.load('../paper/IMUredundant/S3.npy')
    S4 = np.load('../paper/IMUredundant/S4.npy')

    four = four[:1000]
    S1 = S1[:1000]
    S2 = S2[:1000]
    S3 = S3[:1000]
    S4 = S4[:1000]

    time = np.arange(len(four[:, 0]))
    diff_four_S1 = (four - S1)[:, 1:]
    diff_four_S2 = (four - S3)[:, 1:]
    diff_four_S3 = (four - S3)[:, 1:]
    diff_four_S4 = (four - S4)[:, 1:]

    '''plot figure'''
    plt.style.use(['science', 'grid', 'ieee'])
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(time, four[:, 3], label='Redundant configuration', c='k')

    ax.set_ylabel(r'$\omega_z$ [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(a)', loc=4, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xlabel('GPS time [sec]')

    ax.plot(time, diff_four_S1[:, 2], label='Redundant minus ThreeS1', linewidth=0.6, color='r', alpha=0.8)
    ax.legend(loc=2)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(time, four[:, 1], label='Redundant configuration', c='k')
    ax.plot(time, diff_four_S2[:, 0], label='Redundant minus ThreeS2', linewidth=0.6, color='r', alpha=0.8)
    ax.legend(loc=2)
    ax.set_ylabel(r'$\omega_x$ [rad/sec]')
    ax.set_xlabel('GPS time [sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(b)', loc=4, borderpad=-0.2, prop={'size': 9, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    plt.show()
    pass


if __name__ == '__main__':
    # IMU_time_tag_correction()
    # IMU_outlier2()
    # IMU_redundant()
    # SCA_outlier_flip()
    # SCA_combination()
    # SCAonlyVSkalmanFilter()
    # BiasIMU()
    # PointingOneDay(date='2019-10-02',sat=SatID.D)
    KBRanalysis()
    # PointingTimeSeries()
    # SlerpSquad()
