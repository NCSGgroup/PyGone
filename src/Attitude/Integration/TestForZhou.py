"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/1/13
@Description:
"""

from GetInstrument import GetInstrument_L1B
from Setting import SatID, Mission
from Quaternion import Quat, Quaternion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def OneDay(day='2019-01-01', sat=SatID.C, mu=1):
    L1b_dir = '../data/GRACE_FO/RL04/L1B'
    gt = GetInstrument_L1B(mission=Mission.GRACE_FO, date=day).configDir(L1b_dir)

    sca0 = gt.getSCA(sat=sat)
    acc0 = gt.getACC(sat=sat)

    sca = sca0.T
    acc = acc0[1:].T
    index = [2, 3, 4, 1]
    q = Quaternion.normalize(sca[:, index])
    sca = Quat(q=q)

    equatorial = sca.equatorial

    mu = mu
    noise = np.random.randn(3, len(equatorial)).T * mu

    sca_add_noise = Quat(equatorial=equatorial + noise)

    acc_o = sca.transform @ acc[:, :, None]
    acc_n = sca_add_noise.transform @ acc[:, :, None]

    acc_o, acc_n = acc_o[:, :, 0], acc_n[:, :, 0]

    return acc_o - acc_n


def demo():

    '''plot figure'''
    plt.style.use(['science', 'grid'])
    fig = plt.figure()

    # error = 180/np.pi/1000*0.01
    error = 1 / 3600 * 1
    diff1 = OneDay(day='2019-01-01', sat=SatID.C, mu=error)
    # diff2 = OneDay(day='2019-01-01', sat=SatID.D, mu=error)

    resample =50
    time = np.arange(0,len(diff1),resample)

    diff1 = diff1[time]
    # diff2 = diff2[time]

    time = time/3600

    ax = fig.add_subplot(3, 3, 1)
    ax.plot(time, diff1[:, 0], label='X', c = 'C0')
    ax.legend(loc='lower left')
    ax.set_ylabel(r'$[nm/s^2]$')
    ax.set_title('Case-1: 1 [as] ')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(a)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 4)
    ax.plot(time, diff1[:, 1], label='Y', c = 'C1')
    ax.legend(loc='lower left')
    # ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_ylabel(r'$[nm/s^2]$')
    text = AnchoredText('(b)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(time, diff1[:, 2], label='Z', c = 'C2')
    ax.legend(loc='lower left')
    # ax.set_ylabel('Wx [rad/sec]')
    ax.set_ylabel(r'$[nm/s^2]$')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(c)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    # ---------------------------------
    # error = 180 / np.pi / 1000 * 0.1
    error = 0.1
    diff1 = OneDay(day='2019-01-01', sat=SatID.C, mu=error)
    # diff2 = OneDay(day='2019-01-01', sat=SatID.D, mu=error)

    time = np.arange(0, len(diff1), resample)

    diff1 = diff1[time]
    # diff2 = diff2[time]

    time = time / 3600

    ax = fig.add_subplot(3, 3, 3)
    ax.plot(time, diff1[:, 0], label='X', c='C0')
    ax.legend(loc='lower left')
    ax.set_title('Case-3: '+r'$1^{\circ}$')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(g)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 6)
    ax.plot(time, diff1[:, 1], label='Y', c='C1')
    ax.legend(loc='lower left')
    # ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(h)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(time, diff1[:, 2], label='Z', c='C2')
    ax.legend(loc='lower left')
    # ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(i)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    # ---------------------------------
    error = 1/3600*10
    diff1 = OneDay(day='2019-01-01', sat=SatID.C, mu=error)
    # diff2 = OneDay(day='2019-01-01', sat=SatID.D, mu=error)

    time = np.arange(0, len(diff1), resample)

    diff1 = diff1[time]
    # diff2 = diff2[time]

    time = time / 3600

    ax = fig.add_subplot(3, 3, 2)
    ax.plot(time, diff1[:, 0], label='X', c='C0')
    ax.legend(loc='lower left')
    # ax.set_ylabel('Case-3 '+r'$[nm/s^2]$')
    ax.set_title('Case-2: 10 [as] ')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(d)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)


    ax = fig.add_subplot(3, 3, 5)
    ax.plot(time, diff1[:, 1], label='Y', c='C1')
    ax.legend(loc='lower left')
    # ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(e)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(time, diff1[:, 2], label='Z', c='C2')
    ax.legend(loc='lower left')
    # ax.set_ylabel('Wx [rad/sec]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    text = AnchoredText('(f)', loc=2, borderpad=-0.2, prop={'size': 12, 'fontweight': "bold"}, frameon=False)
    ax.add_artist(text)
    ax.set_xlabel('2019-01-01 [Hours]')

    plt.show()
    pass


if __name__ == '__main__':
    demo()
