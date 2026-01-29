"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2023/2/13
@Description:
"""
import sys

sys.path.append('../')

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

from pysrc.Harmonic import Harmonic, SynthesisType
from pysrc.LoveNumber import LoveNumber, LoveNumberType
from pysrc.GeoMathKit import GeoMathKit

LN = LoveNumber('../data/Auxiliary/')
HM = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)


def demo1_sp_vi():
    """VI vs SP for one single month before de-mean"""

    from LoadSH import AOD_GFZ
    from Setting import AODtype

    date = '2002-01-01'
    time1 = '06:00:00'

    Nmax = 100

    rd1 = AOD_GFZ().load('../result/sp').setType(AODtype.ATM)
    # rd2 = AOD_GFZ().load('../result/vi').setType(AODtype.ATM)
    rd2 = AOD_GFZ().load('../result/upperair').setType(AODtype.ATM)

    first = rd1.setTime(date, time1).getCS(Nmax=Nmax)
    second = rd2.setTime(date, time1).getCS(Nmax=Nmax)

    lat = np.arange(90, -90.1, -0.5)
    lon = np.arange(0, 360, 0.5)

    PnmMat = GeoMathKit.getPnmMatrix(lat, Nmax, 2)

    grids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(first[0]), Sqlm=GeoMathKit.CS_1dTo2d(first[1]), PnmMat=PnmMat,
                     lat=lat, lon=lon, Nmax=Nmax, kind=SynthesisType.Pressure)
    b = HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(second[0]), Sqlm=GeoMathKit.CS_1dTo2d(second[1]), PnmMat=PnmMat,
                     lat=lat, lon=lon, Nmax=Nmax, kind=SynthesisType.Pressure)

    grids[0] = a/100
    grids[1] = b/100
    grids[2] = (a-b)/100

    projection = ccrs.Robinson(central_longitude=180)
    transform = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    fig = plt.figure(figsize=(12, 6))

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 3),
                    axes_pad=0.3,
                    cbar_location='bottom',
                    cbar_mode='each',
                    cbar_pad=0,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode

    for i in range(3):
        ax = axgr[i]
        lon2d, lat2d = np.meshgrid(lon, lat)
        # ax.contourf(lon2d, lat2d, grids, 60)

        p = ax.pcolormesh(lon2d, lat2d, grids[i], cmap="RdBu_r", transform=transform)
        # p = ax.pcolormesh(lon2d, lat2d, grids[i], cmap="RdBu_r", transform=transform)
        # ax.set_global()
        ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
        ax.coastlines()
        # ax.set_xticks(np.arange(-180, 180 + 60, 60), crs=transform)
        # ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
        # ax.set_yticks(np.arange(-90, 90 + 30, 30), crs=transform)
        # ax.yaxis.set_minor_locator(plt.MultipleLocator(15))
        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        # ax.yaxis.set_major_formatter(LatitudeFormatter())
        cb = axgr.cbar_axes[i].colorbar(p)
        # cb.set_label_text('pressure [hpa]')

        if i == 0:
            ax.set_title('SP')
        elif i == 1:
            ax.set_title('VI')
        elif i == 2:
            ax.set_title('SP-VI')


    plt.show()

    pass

def demo_tide_res():
    """
    compare tides from my own with the tides from official RL06
    :return:
    """

    from pysrc.Setting import AODtype, TidesType, SynthesisType
    from pysrc.LoadSH import AODtides
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np
    from cartopy.mpl.geoaxes import GeoAxes
    from mpl_toolkits.axes_grid1 import AxesGrid

    kind = AODtype.ATM

    Anm = {}
    Bnm = {}
    Cnm = {}
    Dnm = {}
    grids = {}
    VminMax = {}

    ti1 =TidesType.S1
    ti2 = TidesType.S2

    VminMax[ti1] = (0, 150)
    VminMax[ti2] = (0, 150)

    Nmax = 180
    lat = np.arange(90, -90.1, -0.5)
    lon = np.arange(0, 360, 0.5)

    Pnm = GeoMathKit.getPnmMatrix(lat, Nmax=Nmax, option=1)

    ad = AODtides().load('../data/Products/RL06_tides')
    # ad = AODtides().load('../result/tide/2007_2014/TideGeoCS_topography')
    for tide in [ti1, ti2]:
        print(tide)
        a, b = ad.setInfo(tide, kind, sincos='sin').getCS(Nmax)
        c, d = ad.setInfo(tide, kind, sincos='cos').getCS(Nmax)
        grid_cos = HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(a), Sqlm=GeoMathKit.CS_1dTo2d(b), Nmax=Nmax, lat=lat, lon=lon, PnmMat=Pnm, kind=SynthesisType.Pressure)
        grid_sin = HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(c), Sqlm=GeoMathKit.CS_1dTo2d(d), Nmax=Nmax, lat=lat, lon=lon, PnmMat=Pnm, kind=SynthesisType.Pressure)

        grids[tide] = np.sqrt(grid_cos ** 2 + grid_sin ** 2)
        # grids[tide] = np.sqrt(grid_sin ** 2)

    my = {}
    # my[TidesType.S1] = np.load('../result/temp/AS1.npy')
    # my[TidesType.S1] = np.load('../result/temp/AS1_synthesis.npy')
    # my[TidesType.S2] = np.load('../result/temp/AS2.npy')

    # s1 = np.load('../result/tides/ERA5/S1_grid.npy')
    # s2 = np.load('../result/tides/ERA5/S2_grid.npy')
    s1 = np.load('../result/tide/2007_2014/TidePressure/'+ti1.name+'.npy')
    s2 = np.load('../result/tide/2007_2014/TidePressure/'+ti2.name+'.npy')
    # s1 = np.load('../result/tide/2007/TidePressure/'+ti1.name+'.npy')
    # s2 = np.load('../result/tide/2007/TidePressure/'+ti2.name+'.npy')

    mm = np.sqrt(s1[0, :, :] ** 2 + s1[1, :, :] ** 2)
    my[ti1] = mm
    mm2 = np.sqrt(s2[0, :, :] ** 2 + s2[1, :, :] ** 2)
    my[ti2] = mm2

    projection = ccrs.Robinson(central_longitude=180)
    transform = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    fig = plt.figure(figsize=(12, 6))

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(2, 2),
                    axes_pad=(0.2, 0.5),
                    cbar_location='bottom',
                    cbar_mode='each',
                    cbar_pad=0.1,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode

    i = 0
    for tide in [ti1, ti2]:
        ax = axgr[i]
        lon2d, lat2d = np.meshgrid(lon, lat)
        # ax.contourf(lon2d, lat2d, grids, 60)
        min, max = VminMax[tide]
        p = ax.pcolormesh(lon2d, lat2d, grids[tide], cmap="RdBu_r", transform=transform, vmin=min, vmax=max)
        # ax.set_global()
        ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
        ax.coastlines()
        cb = axgr.cbar_axes[i].colorbar(p, extend='both')
        ax.set_title('RL06.' + tide.name, loc='left')

        ax = axgr[i + 1]
        p = ax.pcolormesh(lon2d, lat2d, my[tide], cmap="RdBu_r", transform=transform, vmin=min, vmax=max)
        # ax.set_global()
        ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
        ax.coastlines()
        cb = axgr.cbar_axes[i + 1].colorbar(p, extend='both')

        ax.set_title('HUST-ERA5.' + tide.name, loc='left')
        i += 2

    # cor = np.array([grids[TidesType.S1].flatten(), grids[TidesType.S2].flatten(),
    #                 my[TidesType.S1].flatten(), my[TidesType.S2].flatten()])
    # coref = np.corrcoef(cor)
    #
    # errS1 = np.linalg.norm(grids[TidesType.S1].flatten() - my[TidesType.S1].flatten())
    # errS2 = np.linalg.norm(grids[TidesType.S2].flatten() - my[TidesType.S2].flatten())
    # errS1r = errS1 / np.linalg.norm(grids[TidesType.S1].flatten())
    # errS2r = errS2 / np.linalg.norm(grids[TidesType.S2].flatten())

    # fig, axs = plt.subplots(2, 1)
    # i = 0
    # axs[0].plot(np.mean(grids[TidesType.S1], 1), lat, label='RL06.S1')
    # axs[0].plot(np.mean(my[TidesType.S1], 1), lat, label='HUST-ERA5.S1')
    # axs[0].legend(loc="upper right")
    # axs[0].set_ylim([-90, 90])
    # axs[0].set_xlim(xmin=0)
    # axs[0].set_ylabel('Latitude [deg]')
    # axs[0].grid()
    # i = 1
    # axs[1].plot(np.mean(grids[TidesType.S2], 1), lat, label='RL06.S2')
    # axs[1].plot(np.mean(my[TidesType.S2], 1), lat, label='HUST-ERA5.S2')
    # axs[1].legend(loc="upper right")
    # axs[1].set_ylim([-90, 90])
    # axs[1].set_ylabel('Latitude [deg]')
    # axs[1].set_xlim(xmin=0)
    # axs[1].set_xlabel('Mean by latitude [pa]')

    # plt.show()

    plt.grid()
    plt.show()


if __name__ == '__main__':
    demo1_sp_vi()
    # demo_tide_res()