import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from Harmonic import Harmonic,SynthesisType
from LoveNumber import LoveNumber, LoveNumberType
from GeoMathKit import GeoMathKit
from LoadSH import AOD_GFZ
from Setting import AODtype

LN = LoveNumber('../data/Auxiliary/')
HM = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)


class Draw:

    def __init__(self):
        self.__date = '2002-01-01'
        self.__time = '00:00:00'

        pass

    def setConfig(self,date='2002-01-01',time='00:00:00'):
        self.__date = date
        self.__time = time

        return self
    def GetDate(self):
        Nmax = 100

        rd1 = AOD_GFZ().load('../result/Primsp').setType(AODtype.ATM)
        rd2 = AOD_GFZ().load('../result/vi').setType(AODtype.ATM)

        first = rd1.setTime(self.__date,self.__time).getCS(Nmax=Nmax)
        second = rd2.setTime(self.__date,self.__time).getCS(Nmax=Nmax)
        # third = np.array(first)-np.array(second)
        first1 = rd1.setTime(self.__date,'06:00:00').getCS(Nmax=Nmax)
        second1 = rd2.setTime(self.__date,'06:00:00').getCS(Nmax=Nmax)

        f = np.array(first1)-np.array(first)
        s = np.array(second1)-np.array(second)

        lat = np.arange(90, -90.1, -0.5)
        lon = np.arange(0, 360, 0.5)

        PnmMat = GeoMathKit.getPnmMatrix(lat,Nmax,2)
        a = HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(f[0]),Sqlm=GeoMathKit.CS_1dTo2d(f[1]),PnmMat=PnmMat,
                         lat=lat, lon=lon, Nmax=Nmax, kind=SynthesisType.Pressure)
        b = HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(s[0]),Sqlm=GeoMathKit.CS_1dTo2d(s[1]),PnmMat=PnmMat,
                         lat=lat, lon=lon, Nmax=Nmax, kind=SynthesisType.Pressure)
        # c = HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(third[0]),Sqlm=GeoMathKit.CS_1dTo2d(third[1]),PnmMat=PnmMat,
        #                  lat=lat, lon=lon, Nmax=Nmax, kind=SynthesisType.Pressure)


        d = a - b
        return a,b,d

    def SpatialMap(self):
        Date = self.GetDate()
        lat = np.arange(90, -90.1, -0.5)
        lon = np.arange(0, 360, 0.5)


        grids = [0,1,2]
        grids[0] = Date[0]
        grids[1] = Date[1]
        grids[2] = Date[2]

        projection =ccrs.Robinson(central_longitude=180)
        transform = ccrs.PlateCarree()
        axes_class = (GeoAxes, dict(map_projection=projection))
        fig = plt.figure(figsize=(10,5),dpi=150)


        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                        nrows_ncols=(1,3),
                        axes_pad=0.3,
                        cbar_location='bottom',
                        cbar_mode='each',
                        cbar_pad=0,
                        cbar_size='3%',
                        label_mode='')


        for i in range(3):
            ax = axgr[i]
            lon2d, lat2d = np.meshgrid(lon, lat)
            p = ax.pcolormesh(lon2d, lat2d, grids[i], cmap='RdBu_r',transform=transform)
            ax.gridlines(draw_labels=False)
            ax.coastlines()

            cb = axgr.cbar_axes[i].colorbar(p)


            if i == 0:
                plt.title('PrimSP')

            elif i == 1:
                plt.title('VI')

            elif i == 2:
                plt.title('PrimSP - VI')

        plt.show()

    def StasticData(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False


        x = np.arange(4)
        xsv_label = ['<500000','50000-70000','70000-100000','>=100000']
        ysp = self.Test(k=0)
        yvi = self.Test(k=1)

        ys_v = self.Test(k=2)

        xs_v_label = ['<1000','1000-3000','3000-5000','>=5000']
        bar_width = 0.3

        ax1 = plt.subplot(131)
        ax1.bar(x,ysp,tick_label=xsv_label,width=bar_width)
        ax1.set_title('Total the Value of Primary SP')
        ax1.set_xlabel('Pa')
        ax1.set_ylabel('Number')

        ax2 = plt.subplot(132)
        ax2.bar(x, yvi, tick_label=xsv_label, width=bar_width)
        ax2.set_title('Total the Value of VI')
        ax2.set_xlabel('Pa')
        # ax2.set_ylabel('Number')

        ax3 = plt.subplot(133)
        ax3.bar(x, ys_v,tick_label= xs_v_label, width=bar_width)
        ax3.set_title('Total the Value of Primary SP -VI')
        ax3.set_xlabel('Pa')
        # ax3.set_ylabel('Number')

        plt.show()


    def Test(self,k:int):

        Data = self.GetDate()
        a,b,c,d = 0,0,0,0
        if k == 0 or k ==1:
            for i in Data[k].flatten():

                if i < 50000:
                    a += 1

                elif i >= 50000 and i < 70000:
                    b += 1
                elif i >= 70000 and i < 100000:
                    c += 1
                elif i >= 100000:
                    d += 1
        elif k == 2:
            for i in Data[k].flatten():
                if i < 1000:
                    a += 1

                elif i >= 1000 and i < 3000:
                    b += 1
                elif i >= 3000 and i < 5000:
                    c += 1
                elif i >= 5000:
                    d += 1
        return a,b,c,d














def demo():
    a = Draw()
    a.SpatialMap()
    # a.StasticData()
    # a.Test(k=0)
    pass

if __name__ == '__main__':
    demo()
