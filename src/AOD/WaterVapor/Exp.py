"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/11/17 下午3:29
@Description:
"""
import sys

sys.path.append('../')
from WetDrySeparation import WetDryMl_ERA5, WetDryMl_ERAinterim
from GeoMathKit import GeoMathKit
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import signal


def exp1():
    wd1 = WetDryMl_ERAinterim()
    wd2 = WetDryMl_ERA5()

    x1, Pw1, Ps1, Pd1, LatW, LatS, LatD, AreaAver = [], [], [], [], [], [], [], []

    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')

    for month in Months:
        print(month)

        if month.year == 1979 and month.month == 12:
            wd = wd1
        else:
            wd = wd2

        wd.setDate(date=str(month.year) + '-' + (str(month.month)).zfill(2))
        Pw = wd.getPw()
        Ps = wd.getPs()

        w1 = wd.getGloMean(Pw)
        s1 = wd.getGloMean(Ps)
        d1 = s1 - w1

        Pw1.append(w1)
        Ps1.append(s1)
        Pd1.append(d1)

        LatW.append(np.mean(Pw, 1))
        LatS.append(np.mean(Ps, 1))
        LatD.append(np.mean(Ps - Pw, 1))

        AreaAver.append([wd.getMeanLatRegion(Pw, LatMax=20, LatMin=10),
                         wd.getMeanLatRegion(Pw, LatMax=10, LatMin=-10),
                         wd.getMeanLatRegion(Pw, LatMax=-10, LatMin=-20),
                         wd.getMeanLatRegion(Pw, LatMax=20, LatMin=-20)])

        x1.append(month.year + (month.month - 1) / 12)

    # np.save('../result/TimeSeries_ERAinterim.npy', np.array([x1, Pw1, Ps1, Pd1]))
    # np.save('../result/TimeSeriesLat_ERAinterim.npy', np.array([LatW, LatS, LatD]))
    # np.save('../result/TimeTag_ERAinterim.npy', np.array(x1))
    # np.save('../result/TimeSeriesAreaAveragePw_ERAinterim.npy', np.array(AreaAver))

    np.save('../result/TimeSeries_ERA5.npy', np.array([x1, Pw1, Ps1, Pd1]))
    np.save('../result/TimeSeriesLat_ERA5.npy', np.array([LatW, LatS, LatD]))
    np.save('../result/TimeTag_ERA5.npy', np.array(x1))
    np.save('../result/TimeSeriesAreaAveragePw_ERA5.npy', np.array(AreaAver))

    Pw1 = np.array(Pw1)
    Ps1 = np.array(Ps1)
    Pd1 = np.array(Pd1)
    # Pw1 = Pw1 - np.mean(Pw1)
    # Ps1 = Ps1 - np.mean(Ps1)
    # Pd1 = Pd1 - np.mean(Pd1)

    '''plot'''

    plt.style.use(['science', 'grid', 'ieee'])

    plt.subplot(311)
    plt.plot(x1, Ps1, color='C1', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Total Pressure [Pa]')
    plt.title('a', x=0.02, y=0.8, weight='bold',
              bbox=dict(boxstyle='round', pad=0.1, fc='yellow', alpha=0.4, ec='k', lw=1))
    # plt.ylim(-25,25)

    plt.subplot(312)
    plt.plot(x1, Pw1, color='C1', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Wet air [Pa]')
    plt.title('b', x=0.02, y=0.8, weight='bold',
              bbox=dict(boxstyle='round', pad=0.1, fc='yellow', alpha=0.4, ec='k', lw=1))
    # plt.ylim(-25, 25)

    plt.subplot(313)
    plt.plot(x1, Pd1, color='C1', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Dry air [Pa]')
    plt.title('c', x=0.02, y=0.8, weight='bold',
              bbox=dict(boxstyle='round', pad=0.1, fc='yellow', alpha=0.4, ec='k', lw=1))
    # plt.ylim(-20, 20)

    plt.show()
    pass


def exp2():
    """
    Compute the mean annual cycle
    :return:
    """
    wd1 = WetDryMl_ERAinterim()
    wd2 = WetDryMl_ERA5()
    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')

    glo = [{1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: []
            },
           {1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: []
            },
           {1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: []
            }
           ]
    NH = [{1: [],
           2: [],
           3: [],
           4: [],
           5: [],
           6: [],
           7: [],
           8: [],
           9: [],
           10: [],
           11: [],
           12: []
           },
          {1: [],
           2: [],
           3: [],
           4: [],
           5: [],
           6: [],
           7: [],
           8: [],
           9: [],
           10: [],
           11: [],
           12: []
           },
          {1: [],
           2: [],
           3: [],
           4: [],
           5: [],
           6: [],
           7: [],
           8: [],
           9: [],
           10: [],
           11: [],
           12: []
           }
          ]
    SH = [{1: [],
           2: [],
           3: [],
           4: [],
           5: [],
           6: [],
           7: [],
           8: [],
           9: [],
           10: [],
           11: [],
           12: []
           },
          {1: [],
           2: [],
           3: [],
           4: [],
           5: [],
           6: [],
           7: [],
           8: [],
           9: [],
           10: [],
           11: [],
           12: []
           },
          {1: [],
           2: [],
           3: [],
           4: [],
           5: [],
           6: [],
           7: [],
           8: [],
           9: [],
           10: [],
           11: [],
           12: []
           }
          ]
    LatV = [{1: [],
             2: [],
             3: [],
             4: [],
             5: [],
             6: [],
             7: [],
             8: [],
             9: [],
             10: [],
             11: [],
             12: []
             },
            {1: [],
             2: [],
             3: [],
             4: [],
             5: [],
             6: [],
             7: [],
             8: [],
             9: [],
             10: [],
             11: [],
             12: []
             },
            {1: [],
             2: [],
             3: [],
             4: [],
             5: [],
             6: [],
             7: [],
             8: [],
             9: [],
             10: [],
             11: [],
             12: []
             }
            ]

    for month in Months:

        if month.year == 1979 and month.month == 12:
            wd = wd1
        else:
            wd = wd2

        print(month)
        wd.setDate(date=str(month.year) + '-' + (str(month.month)).zfill(2))
        Pw = wd.getPw()
        Ps = wd.getPs()

        w = wd.getGloMean(Pw)
        SHw = wd.getSHmean(Pw)
        NHw = wd.getNHmean(Pw)
        Lat_w = np.mean(Pw, 1)

        s = wd.getGloMean(Ps)
        SHs = wd.getSHmean(Ps)
        NHs = wd.getNHmean(Ps)
        Lat_s = np.mean(Ps, 1)

        d = s - w
        SHd = SHs - SHw
        NHd = NHs - NHw
        Lat_d = Lat_s - Lat_w

        glo[0][month.month].append(w)
        glo[1][month.month].append(s)
        glo[2][month.month].append(d)

        NH[0][month.month].append(NHw)
        NH[1][month.month].append(NHs)
        NH[2][month.month].append(NHd)

        SH[0][month.month].append(SHw)
        SH[1][month.month].append(SHs)
        SH[2][month.month].append(SHd)

        LatV[0][month.month].append(Lat_w)
        LatV[1][month.month].append(Lat_s)
        LatV[2][month.month].append(Lat_d)

        pass

    for i in range(1, 13):
        glo[0][i] = np.mean(np.array(glo[0][i]))
        glo[1][i] = np.mean(np.array(glo[1][i]))
        glo[2][i] = np.mean(np.array(glo[2][i]))

        NH[0][i] = np.mean(np.array(NH[0][i]))
        NH[1][i] = np.mean(np.array(NH[1][i]))
        NH[2][i] = np.mean(np.array(NH[2][i]))

        SH[0][i] = np.mean(np.array(SH[0][i]))
        SH[1][i] = np.mean(np.array(SH[1][i]))
        SH[2][i] = np.mean(np.array(SH[2][i]))

        LatV[0][i] = np.mean(np.array(LatV[0][i]), 0)
        LatV[1][i] = np.mean(np.array(LatV[1][i]), 0)
        LatV[2][i] = np.mean(np.array(LatV[2][i]), 0)

    '''plot'''
    plt.style.use(['science', 'grid'])
    tot = [list(glo[1].values()),
           list(NH[1].values()),
           list(SH[1].values())]
    wet = [list(glo[0].values()),
           list(NH[0].values()),
           list(SH[0].values())]
    dry = [list(glo[2].values()),
           list(NH[2].values()),
           list(SH[2].values())]

    # np.save('../result/MeanAnnualCycle_ERAinterim.npy', np.array([tot[0], wet[0], dry[0]]))
    # np.save('../result/MeanAnnualCycleLat_ERAinterim.npy', np.array([list(LatV[0].values()),
    #                                                                  list(LatV[1].values()),
    #                                                                  list(LatV[2].values())
    #                                                                  ]))

    np.save('../result/MeanAnnualCycle_ERA5.npy', np.array([tot[0], wet[0], dry[0]]))
    np.save('../result/MeanAnnualCycleLat_ERA5.npy', np.array([list(LatV[0].values()),
                                                               list(LatV[1].values()),
                                                               list(LatV[2].values())
                                                               ]))

    tot = np.array(tot) / 100
    wet = np.array(wet) / 100
    dry = np.array(dry) / 100

    # np.save('../result/MeanAnnualCycleNS_ERAinterim.npy', np.array([tot, wet, dry]))
    np.save('../result/MeanAnnualCycleNS_ERA5.npy', np.array([tot, wet, dry]))

    for i in range(3):
        tot[i] = tot[i] - np.mean(tot[i])
        wet[i] = wet[i] - np.mean(wet[i])
        dry[i] = dry[i] - np.mean(dry[i])

    plt.subplot(311)
    plt.plot(tot[0], color='C0', label='Gl')
    plt.plot(tot[1], color='C1', label='NH')
    plt.plot(tot[2], color='C2', label='SH')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Total [hPa]')
    plt.ylim(-2, 2)

    plt.subplot(312)
    plt.plot(wet[0], color='C0', label='Gl')
    plt.plot(wet[1], color='C1', label='NH')
    plt.plot(wet[2], color='C2', label='SH')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Water vapor [hPa]')
    plt.ylim(-2, 2)

    plt.subplot(313)
    plt.plot(dry[0], color='C0', label='Gl')
    plt.plot(dry[1], color='C1', label='NH')
    plt.plot(dry[2], color='C2', label='SH')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Dry air [hPa]')
    plt.ylim(-2, 2)

    plt.show()


def plot1():
    '''plot'''

    plt.style.use(['science', 'grid'])

    w = np.load('../result/TimeSeries_ERA5.npy')
    w1 = np.load('../result/TimeSeries_ERAinterim.npy')
    x, Ps, Pw, Pd = w[0], w[2] / 100, w[1] / 100, w[3] / 100
    x1, Ps1, Pw1, Pd1 = w1[0], w1[2] / 100, w1[1] / 100, w1[3] / 100

    plt.subplot(311)
    plt.plot(x, Ps, color='C0', label='ERA-5')
    plt.plot(x1, Ps1, color='C1', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Total Pressure [hPa]')
    plt.xlim(1979, 2019)

    plt.subplot(312)
    plt.plot(x, Pw, color='C0', label='ERA-5')
    plt.plot(x1, Pw1, color='C1', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Wet air [hPa]')
    # plt.ylim(-25, 25)
    plt.xlim(1979, 2019)

    plt.subplot(313)
    plt.plot(x, Pd, color='C0', label='ERA-5')
    plt.plot(x1, Pd1, color='C1', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Dry air [hPa]')
    # plt.ylim(-20, 20)
    plt.xlim(1979, 2019)

    plt.show()


def plot2():
    '''plot'''

    plt.style.use(['science', 'grid'])

    # ===================  ERA-interim =====================
    w1 = np.load('../result/TimeSeries_ERAinterim.npy')
    sd = np.load('../result/MeanAnnualCycle_ERAinterim.npy')
    x1, Ps1, Pw1, Pd1 = w1[0], w1[2], w1[1], w1[3]

    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')
    i = 0
    for mon in Months:
        Ps1[i] = Ps1[i] - sd[0][i % 12]
        Pw1[i] = Pw1[i] - sd[1][i % 12]
        Pd1[i] = Pd1[i] - sd[2][i % 12]
        i += 1
        pass

    plt.subplot(311)
    # plt.plot(x, Ps, color='C0', label='ERA-5')
    plt.plot(x1, Ps1 / 100, color='C0', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Total Pressure [hPa]')
    plt.axhline(y=0, ls="-", c="black")
    # plt.ylim(-25,25)

    plt.subplot(312)
    # plt.plot(x, Pw, color='C0', label='ERA-5')
    plt.plot(x1, Pw1 / 100, color='C0', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Wet air [hPa]')
    plt.axhline(y=0, ls="-", c="black")
    # plt.ylim(-25, 25)

    plt.subplot(313)
    # plt.plot(x, Pd, color='C0', label='ERA-5')
    plt.plot(x1, Pd1 / 100, color='C0', label='ERA-interim')
    plt.legend()
    plt.axis('tight')
    plt.ylabel('Dry air [hPa]')
    plt.axhline(y=0, ls="-", c="black")
    # plt.ylim(-20, 20)

    # ===================  ERA-5 =====================

    w1 = np.load('../result/TimeSeries_ERA5.npy')
    sd = np.load('../result/MeanAnnualCycle_ERA5.npy')
    x1, Ps1, Pw1, Pd1 = w1[0], w1[2], w1[1], w1[3]

    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')
    i = 0
    for mon in Months:
        Ps1[i] = Ps1[i] - sd[0][i % 12]
        Pw1[i] = Pw1[i] - sd[1][i % 12]
        Pd1[i] = Pd1[i] - sd[2][i % 12]
        i += 1
        pass

    plt.subplot(311)
    # plt.plot(x, Ps, color='C0', label='ERA-5')
    plt.plot(x1, Ps1 / 100, color='C1', label='ERA-5')
    plt.legend()
    plt.axis('tight')

    # plt.ylim(-25,25)

    plt.subplot(312)
    # plt.plot(x, Pw, color='C0', label='ERA-5')
    plt.plot(x1, Pw1 / 100, color='C1', label='ERA-5')
    plt.legend()
    plt.axis('tight')

    # plt.ylim(-25, 25)

    plt.subplot(313)
    # plt.plot(x, Pd, color='C0', label='ERA-5')
    plt.plot(x1, Pd1 / 100, color='C1', label='ERA-5')
    plt.legend()
    plt.axis('tight')

    # plt.ylim(-20, 20)

    plt.show()


def plot3():
    plt.style.use(['science', 'grid', 'ieee'])

    # =================  ERA-interim ===========================
    res = np.load('../result/MeanAnnualCycleNS_ERAinterim.npy')
    tot, wet, dry = res[0], res[1], res[2]

    for i in range(3):
        tot[i] = tot[i] - np.mean(tot[i])
        wet[i] = wet[i] - np.mean(wet[i])
        dry[i] = dry[i] - np.mean(dry[i])

    ax = plt.subplot(321)
    plt.plot(tot[0], color='C0', label='Gl')
    plt.plot(tot[1], color='C1', label='NH')
    plt.plot(tot[2], color='C2', label='SH', ls='-')
    plt.legend(fontsize=7, ncol=3)
    plt.axis('tight')
    plt.ylabel('Total [hPa]')
    plt.ylim(-2, 2)
    plt.title('ERA-interim')
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    plt.xlim(0, 11)

    ax = plt.subplot(323)
    plt.plot(wet[0], color='C0', label='Gl')
    plt.plot(wet[1], color='C1', label='NH')
    plt.plot(wet[2], color='C2', label='SH', ls='-')
    plt.legend(fontsize=7, ncol=3)
    plt.axis('tight')
    plt.ylabel('Water vapor [hPa]')
    plt.ylim(-2, 2)
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    plt.xlim(0, 11)

    ax = plt.subplot(325)
    plt.plot(dry[0], color='C0', label='Gl')
    plt.plot(dry[1], color='C1', label='NH')
    plt.plot(dry[2], color='C2', label='SH', ls='-')
    plt.legend(fontsize=7, ncol=3)
    plt.axis('tight')
    plt.ylabel('Dry air [hPa]')
    plt.ylim(-2, 2)
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    plt.xlim(0, 11)

    # =================  ERA-5 ===========================
    res = np.load('../result/MeanAnnualCycleNS_ERA5.npy')
    tot, wet, dry = res[0], res[1], res[2]

    for i in range(3):
        tot[i] = tot[i] - np.mean(tot[i])
        wet[i] = wet[i] - np.mean(wet[i])
        dry[i] = dry[i] - np.mean(dry[i])

    ax = plt.subplot(322)
    plt.plot(tot[0], color='C0', label='Gl')
    plt.plot(tot[1], color='C1', label='NH')
    plt.plot(tot[2], color='C2', label='SH', ls='-')
    plt.legend(fontsize=7, ncol=3)
    plt.axis('tight')
    plt.ylim(-2, 2)
    plt.title('ERA-5')
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    plt.xlim(0, 11)

    ax = plt.subplot(324)
    plt.plot(wet[0], color='C0', label='Gl')
    plt.plot(wet[1], color='C1', label='NH')
    plt.plot(wet[2], color='C2', label='SH', ls='-')
    plt.legend(fontsize=7, ncol=3)
    plt.axis('tight')
    plt.ylim(-2, 2)
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    plt.xlim(0, 11)

    ax = plt.subplot(326)
    plt.plot(dry[0], color='C0', label='Gl')
    plt.plot(dry[1], color='C1', label='NH')
    plt.plot(dry[2], color='C2', label='SH', ls='-')
    plt.legend(fontsize=7, ncol=3)
    plt.axis('tight')
    plt.ylim(-2, 2)
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    plt.xlim(0, 11)

    plt.show()

    pass


def plot4():
    plt.style.use(['science', 'grid'])

    # ================================ERA-interim===============================
    LatV = np.load('../result/MeanAnnualCycleLat_ERAinterim.npy')
    lat = np.arange(90, -90.1, -0.5)
    lat = np.tile(lat, (12, 1))

    for i in range(3):
        LatV[i] = LatV[i] - np.tile(np.mean(LatV[i], 0), (12, 1))
        LatV[i] = LatV[i] * np.cos(np.deg2rad(lat))
    pass

    fig = plt.figure()

    x = np.arange(12)
    y = np.arange(90, -90.1, -0.5)
    X, Y = np.meshgrid(x, y)

    ax = fig.add_subplot(3, 2, 1)
    ll = ax.contourf(X, Y, LatV[1].T / 100, levels=np.linspace(-4, 4, 9), cmap=plt.cm.seismic)
    llc = ax.contour(X, Y, LatV[1].T / 100, levels=np.linspace(-4, 4, 9), colors=['black'])
    ax.set_title('Total: ERA-interim', loc='left')
    fig.colorbar(ll)
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    # Use the line contours to place contour labels.
    # ax.clabel(
    #     llc,  # Typically best results when labelling line contours.
    #     colors=['black'],
    #     manual=False,  # Automatic placement vs manual placement.
    #     inline=True,  # Cut the line where the label will be placed.
    #     fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    # )

    ax = fig.add_subplot(3, 2, 3)
    ll = ax.contourf(X, Y, LatV[0].T / 100, levels=np.linspace(-1, 1, 9), cmap=plt.cm.seismic)
    llc = ax.contour(X, Y, LatV[0].T / 100, levels=np.linspace(-1, 1, 9), colors=['black'])
    ax.set_title('Water Vapor: ERA-interim', loc='left')
    fig.colorbar(ll)
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    # Use the line contours to place contour labels.
    # ax.clabel(
    #     llc,  # Typically best results when labelling line contours.
    #     colors=['black'],
    #     manual=False,  # Automatic placement vs manual placement.
    #     inline=True,  # Cut the line where the label will be placed.
    #     fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    # )

    ax = fig.add_subplot(3, 2, 5)
    ll = ax.contourf(X, Y, LatV[2].T / 100, levels=np.linspace(-4, 4, 9), cmap=plt.cm.seismic)
    llc = ax.contour(X, Y, LatV[2].T / 100, levels=np.linspace(-4, 4, 9), colors=['black'])
    ax.set_title('Dry Air: ERA-interim', loc='left')
    fig.colorbar(ll)
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    # Use the line contours to place contour labels.
    # ax.clabel(
    #     llc,  # Typically best results when labelling line contours.
    #     colors=['black'],
    #     manual=False,  # Automatic placement vs manual placement.
    #     inline=True,  # Cut the line where the label will be placed.
    #     fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    # )

    # ================================ERA-5===============================
    LatV = np.load('../result/MeanAnnualCycleLat_ERA5.npy')
    lat = np.arange(90, -90.1, -0.5)
    lat = np.tile(lat, (12, 1))

    for i in range(3):
        LatV[i] = LatV[i] - np.tile(np.mean(LatV[i], 0), (12, 1))
        LatV[i] = LatV[i] * np.cos(np.deg2rad(lat))
    pass

    x = np.arange(12)
    y = np.arange(90, -90.1, -0.5)
    X, Y = np.meshgrid(x, y)

    ax = fig.add_subplot(3, 2, 2)
    ll = ax.contourf(X, Y, LatV[1].T / 100, levels=np.linspace(-4, 4, 9), cmap=plt.cm.seismic)
    llc = ax.contour(X, Y, LatV[1].T / 100, levels=np.linspace(-4, 4, 9), colors=['black'])
    fig.colorbar(ll)
    ax.set_title('Total: ERA-5', loc='left')
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)

    # Use the line contours to place contour labels.
    # ax.clabel(
    #     llc,  # Typically best results when labelling line contours.
    #     colors=['black'],
    #     manual=False,  # Automatic placement vs manual placement.
    #     inline=True,  # Cut the line where the label will be placed.
    #     fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    # )

    ax = fig.add_subplot(3, 2, 4)
    ll = ax.contourf(X, Y, LatV[0].T / 100, levels=np.linspace(-1, 1, 9), cmap=plt.cm.seismic)
    llc = ax.contour(X, Y, LatV[0].T / 100, levels=np.linspace(-1, 1, 9), colors=['black'])
    fig.colorbar(ll)
    ax.set_title('Water Vapor: ERA-5', loc='left')
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)

    # Use the line contours to place contour labels.
    # ax.clabel(
    #     llc,  # Typically best results when labelling line contours.
    #     colors=['black'],
    #     manual=False,  # Automatic placement vs manual placement.
    #     inline=True,  # Cut the line where the label will be placed.
    #     fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    # )

    ax = fig.add_subplot(3, 2, 6)
    ll = ax.contourf(X, Y, LatV[2].T / 100, levels=np.linspace(-4, 4, 9), cmap=plt.cm.seismic)
    llc = ax.contour(X, Y, LatV[2].T / 100, levels=np.linspace(-4, 4, 9), colors=['black'])
    fig.colorbar(ll)
    ax.set_title('Dry Air: ERA-5', loc='left')
    xticks = range(0, 12, 1)
    xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)

    # Use the line contours to place contour labels.
    # ax.clabel(
    #     llc,  # Typically best results when labelling line contours.
    #     colors=['black'],
    #     manual=False,  # Automatic placement vs manual placement.
    #     inline=True,  # Cut the line where the label will be placed.
    #     fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    # )

    plt.show()


def plot5():
    plt.style.use(['science'])

    # =====================ERA-interim=========================
    res = np.load('../result/TimeSeriesLat_ERAinterim.npy')
    LatV = np.load('../result/MeanAnnualCycleLat_ERAinterim.npy')
    lat1 = np.arange(90, -90.1, -0.5)
    lat = np.tile(lat1, (12, 1))
    for i in range(3):
        LatV[i] = LatV[i]
    pass

    W, S, D = res[0], res[1], res[2]
    # a = np.shape(W)[0]
    # W = W - np.tile(np.mean(W, 0) ,(a, 1))
    # S = S - np.tile(np.mean(S, 0), (a, 1))
    # D = D - np.tile(np.mean(D, 0), (a, 1))

    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')
    i = 0
    for mon in Months:
        W[i] = W[i] - LatV[0, i % 12]
        S[i] = S[i] - LatV[1, i % 12]
        D[i] = D[i] - LatV[2, i % 12]
        i += 1
        pass

    y = np.load('../result/TimeTag_ERAinterim.npy')
    x = np.arange(90, -90.1, -0.5)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ll = ax.pcolormesh(X, Y, S / 100, cmap="RdBu_r", vmin=-10, vmax=10)
    ax.invert_yaxis()
    fig.colorbar(ll)
    ax.set_title('ERA-interim Ps (hPa)')

    ax = fig.add_subplot(2, 2, 2)
    ll = ax.pcolormesh(X, Y, W / 100, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.invert_yaxis()
    fig.colorbar(ll)
    ax.set_title('ERA-interim Pw (hPa)')

    # =====================ERA-5=========================
    res = np.load('../result/TimeSeriesLat_ERA5.npy')
    LatV = np.load('../result/MeanAnnualCycleLat_ERA5.npy')
    lat1 = np.arange(90, -90.1, -0.5)
    lat = np.tile(lat1, (12, 1))
    for i in range(3):
        LatV[i] = LatV[i]
    pass

    W, S, D = res[0], res[1], res[2]
    # a = np.shape(W)[0]
    # W = W - np.tile(np.mean(W, 0) ,(a, 1))
    # S = S - np.tile(np.mean(S, 0), (a, 1))
    # D = D - np.tile(np.mean(D, 0), (a, 1))

    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')
    i = 0
    for mon in Months:
        W[i] = W[i] - LatV[0, i % 12]
        S[i] = S[i] - LatV[1, i % 12]
        D[i] = D[i] - LatV[2, i % 12]
        i += 1
        pass

    y = np.load('../result/TimeTag_ERA5.npy')
    x = np.arange(90, -90.1, -0.5)
    X, Y = np.meshgrid(x, y)

    ax = fig.add_subplot(2, 2, 3)
    ll = ax.pcolormesh(X, Y, S / 100, cmap="RdBu_r", vmin=-10, vmax=10)
    ax.invert_yaxis()
    fig.colorbar(ll)
    ax.set_title('ERA5 Ps (hPa)')

    ax = fig.add_subplot(2, 2, 4)
    ll = ax.pcolormesh(X, Y, W / 100, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.invert_yaxis()
    fig.colorbar(ll)
    ax.set_title('ERA5 Pw (hPa)')

    plt.show()

    pass


def plot6():
    plt.style.use(['science'])
    # ===========================ERAinterim==========================
    aa = np.load('../result/TimeSeriesAreaAveragePw_ERAinterim.npy')
    LatV = np.load('../result/MeanAnnualCycleLat_ERAinterim.npy')
    x = np.load('../result/TimeTag_ERAinterim.npy')
    LatPw = LatV[0]
    lat1 = np.arange(90, -90.1, -0.5)
    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')
    '''20N-10N'''
    LatMin = 10
    LatMax = 20
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 0] = aa[i, 0] - PwCycle[i % 12]
        i += 1
        pass
    '''10N-10S'''
    LatMin = -10
    LatMax = 10
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 1] = aa[i, 1] - PwCycle[i % 12]
        i += 1
        pass
    '''10S-20S'''
    LatMin = -20
    LatMax = -10
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 2] = aa[i, 2] - PwCycle[i % 12]
        i += 1
        pass
    '''20N-20S'''
    LatMin = -20
    LatMax = 20
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 3] = aa[i, 3] - PwCycle[i % 12]
        i += 1
        pass

    '''butterworth low-pass filter to get the interanual variability'''
    order = 3
    cutoff = 0.8  # year
    sr = 1  # month
    cutoff = 1 / (cutoff * 12)
    wn = 2 * cutoff / sr
    assert wn <= 1
    b, a = signal.butter(order, wn, 'low')

    plt.subplot(4, 2, 1)
    y = aa[:, 0]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='20N-10N', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)
    plt.title('ERA-interim [hPa]')

    plt.subplot(4, 2, 3)
    y = aa[:, 1]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='10N-10S', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)

    plt.subplot(4, 2, 5)
    y = aa[:, 2]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='10S-20S', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)

    plt.subplot(4, 2, 7)
    y = aa[:, 3]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='20N-20S', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)

    # ===========================ERA-5==========================
    aa = np.load('../result/TimeSeriesAreaAveragePw_ERA5.npy')
    LatV = np.load('../result/MeanAnnualCycleLat_ERA5.npy')
    x = np.load('../result/TimeTag_ERA5.npy')
    LatPw = LatV[0]
    lat1 = np.arange(90, -90.1, -0.5)
    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')
    '''20N-10N'''
    LatMin = 10
    LatMax = 20
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 0] = aa[i, 0] - PwCycle[i % 12]
        i += 1
        pass
    '''10N-10S'''
    LatMin = -10
    LatMax = 10
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 1] = aa[i, 1] - PwCycle[i % 12]
        i += 1
        pass
    '''10S-20S'''
    LatMin = -20
    LatMax = -10
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 2] = aa[i, 2] - PwCycle[i % 12]
        i += 1
        pass
    '''20N-20S'''
    LatMin = -20
    LatMax = 20
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 3] = aa[i, 3] - PwCycle[i % 12]
        i += 1
        pass

    '''butterworth low-pass filter to get the interanual variability'''
    order = 3
    cutoff = 0.8  # year
    sr = 1  # month
    cutoff = 1 / (cutoff * 12)
    wn = 2 * cutoff / sr
    assert wn <= 1
    b, a = signal.butter(order, wn, 'low')

    plt.subplot(4, 2, 2)
    y = aa[:, 0]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='20N-10N', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)
    plt.title('ERA5 [hPa]')

    plt.subplot(4, 2, 4)
    y = aa[:, 1]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='10N-10S', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)

    plt.subplot(4, 2, 6)
    y = aa[:, 2]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='10S-20S', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)

    plt.subplot(4, 2, 8)
    y = aa[:, 3]
    yn = signal.filtfilt(b, a, y)
    plt.plot(x, yn / 100, label='filtered', color='red', lw=2)
    plt.plot(x, y / 100, label='20N-20S', lw=0.5, color='blue')
    plt.legend(fontsize=10, ncol=3)
    plt.xlim(1979, 2019)

    plt.show()

    pass


def plot7():
    from ClimateIndex import Nino34

    '''butterworth low-pass filter to get the interanual variability'''
    order = 3
    cutoff = 0.8  # year
    sr = 1  # month
    cutoff = 1 / (cutoff * 12)
    wn = 2 * cutoff / sr
    assert wn <= 1
    b, a = signal.butter(order, wn, 'low')

    ni = Nino34()
    res = ni.setDate('1979-01', '2018-12')

    plt.style.use(['science', 'grid', 'ieee'])
    ax = plt.subplot(3, 1, 1)

    x = res[:, 0]
    y1 = res[:, 1]
    y1 = y1 - np.mean(y1)
    y2 = np.zeros(len(x))

    y1 = signal.filtfilt(b, a, y1)
    Nino = y1

    plt.plot(x, y1, label='SSTs', color='black', lw=1)
    plt.plot(x, y2, color='black', lw=1)

    ax.fill_between(x, y1, y2, where=(y1 > y2), color='red', alpha=0.3,
                    interpolate=True)
    ax.fill_between(x, y1, y2, where=(y1 <= y2), color='blue', alpha=0.3,
                    interpolate=True)
    plt.legend(fontsize=10, ncol=3)
    # plt.xlim(1979, 2019)
    plt.title('El-nino 3.4 Index')

    #--------------------
    # ===========================ERAinterim==========================
    aa = np.load('../result/TimeSeriesAreaAveragePw_ERAinterim.npy')
    LatV = np.load('../result/MeanAnnualCycleLat_ERAinterim.npy')
    x = np.load('../result/TimeTag_ERAinterim.npy')
    LatPw = LatV[0]
    lat1 = np.arange(90, -90.1, -0.5)
    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')

    '''10N-10S'''
    LatMin = -10
    LatMax = 10
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 1] = aa[i, 1] - PwCycle[i % 12]
        i += 1
        pass

    ax=plt.subplot(3,1,2)
    y = aa[:, 1]
    y1 = signal.filtfilt(b, a, y)
    y1 = y1/100

    ERAinterim = y1
    plt.plot(x, y1, label='Correlation 0.637', color='black', lw=1)
    plt.plot(x, y2, color='black', lw=1)
    # plt.plot(x, y / 100, label='10N-10S', lw=0.5, color='blue')
    ax.fill_between(x, y1, y2, where=(y1 > y2), color='red', alpha=0.3,
                    interpolate=True)
    ax.fill_between(x, y1, y2, where=(y1 <= y2), color='blue', alpha=0.3,
                    interpolate=True)
    plt.legend(fontsize=10, ncol=3)
    plt.title('ERA-interim')
    # plt.xlim(1979, 2019)


    # ===========================ERA-5==========================
    aa = np.load('../result/TimeSeriesAreaAveragePw_ERA5.npy')
    LatV = np.load('../result/MeanAnnualCycleLat_ERA5.npy')
    x = np.load('../result/TimeTag_ERA5.npy')
    LatPw = LatV[0]
    lat1 = np.arange(90, -90.1, -0.5)
    Months = GeoMathKit.monthListByMonth(begin='1979-01', end='2018-12')

    '''10N-10S'''
    LatMin = -10
    LatMax = 10
    index = (lat1 >= LatMin) * (lat1 <= LatMax)
    lat = lat1[index]
    PwCycle = LatPw[:, index] @ np.cos(np.deg2rad(lat))
    area = np.sum(np.cos(np.deg2rad(lat)))
    PwCycle = PwCycle / area
    i = 0
    for mon in Months:
        aa[i, 1] = aa[i, 1] - PwCycle[i % 12]
        i += 1
        pass

    ax=plt.subplot(3, 1, 3)
    y = aa[:, 1]
    y1 = signal.filtfilt(b, a, y)
    y1=y1/100

    ERA5=y1
    plt.plot(x, y1, label='Correlation 0.666', color='black', lw=1)
    plt.plot(x, y2, color='black', lw=1)
    # plt.plot(x, y / 100, label='10N-10S', lw=0.5, color='blue')
    ax.fill_between(x, y1, y2, where=(y1 > y2), color='red', alpha=0.3,
                    interpolate=True)
    ax.fill_between(x, y1, y2, where=(y1 <= y2), color='blue', alpha=0.3,
                    interpolate=True)
    plt.legend(fontsize=10, ncol=3)
    plt.title('ERA5')
    # plt.xlim(1979, 2019)


    """
    correlation
    """

    cor1 = np.corrcoef(ERA5, Nino)
    cor2 = np.corrcoef(ERAinterim, Nino)
    plt.show()


if __name__ == '__main__':
    # exp2()
    plot4()
