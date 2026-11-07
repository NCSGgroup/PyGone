import numpy as np
import xarray as xr
import pygmt
import os
import h5py
from tqdm import tqdm
from pysrc.core.settings import Model
from datetime import datetime, timedelta

import os
os.environ['GMT_LIBRARY_PATH'] = 'D:/App/GMT/gmt6/bin'

pygmt.config(FONT_ANNOT='11p, Times-Roman,black',FONT_LABEL='11p, Times-Roman,black')

class Config():
    def __init__(self, monthdata: dict, factor):
        self.monthdata = monthdata
        self.factor = factor
        self.Path_GF = None
        self.Path_GF1 = None
        self.Path_GF2 = None
        self.Path_KBR1 = None
        self.Path_KBR2 = None
        self.Path_Root = None

    def day(self):
        year, month = map(int,self.monthdata.split('-'))
        first_day=datetime(year,month,1)
        last_day=(datetime(year,month+1,1)-timedelta(days=1)) if month<12 else (datetime(year+1,1,1)-timedelta(days=1))

        return first_day,last_day


    def setPathRoot(self,rootpath='../KBR_result'):
        factor = self.factor
        self.Path_Root = rootpath
        if factor == 'Tide' or factor == 'tide':
            self.Path_Root = rootpath+'/1_Tide/'
            self.Path_GF1 = self.Path_Root + '/atmos-2007/'
            self.Path_GF2 = self.Path_Root + '/atmos-2007_2012/'
            self.Path_KBR1 = self.Path_Root+'/atmos-2007/2009-05-01_2009-05-31/'
            self.Path_KBR2 = self.Path_Root+'/atmos-2007_2012/2009-05-01_2009-05-31/'
        elif factor == 'GeoHeight' or factor == 'Geo':
            self.Path_Root = rootpath+'/3_GeoHeight/'
            self.Path_GF1 = self.Path_Root + '/model_KG/'
            self.Path_GF2 = self.Path_Root + '/pressure_KG/'
            self.Path_KBR1 = self.Path_GF1 + '/2009-08-01_2009-08-31/'
            self.Path_KBR2 = self.Path_GF2 + '/2009-08-01_2009-08-31/'

        print('Root Path is: {}'.format(self.Path_Root))
        print('Gravity Field Path 1: {}'.format(self.Path_GF1))
        print('Gravity Field Path 2: {}'.format(self.Path_GF2))
        print('KBR Path 1: {}'.format(self.Path_KBR1))
        print('KBR Path 2: {}'.format(self.Path_KBR2))
        return self


    def setPath(self,path='../KBR_result/'):
        self.Path_Root = path + '%s'%(self.monthdata)
        return self
    def setSpacePath(self,KBR_SPath='/result/'):
        self.Path_GF = self.setPath(path='../KBR_result/').Path_Root + KBR_SPath
        return self
    def setTimePath(self):
        first_day,last_day=self.day()
        KBR_TPath1 = '/Normal/'
        KBR_TPath2 = '/GSCF-2/'
        KBR_TPath3 = '/GSCF-3/'
        self.Path_KBR1 = self.setPath(path='../KBR_result/').Path_Root+KBR_TPath1
        self.Path_KBR2 = self.setPath(path='../KBR_result/').Path_Root+KBR_TPath2
        self.Path_KBR3 = self.setPath(path='../KBR_result/').Path_Root+KBR_TPath3
        return self


class KBR(Config):
    def __init__(self,facotr='Geo', monthdata='2020-3'):
        super().__init__(factor=facotr,monthdata=monthdata)
        self.date=datetime.strptime(monthdata,'%Y-%m').strftime('%Y-%m')
        self.Name1=self.date+'-normal'
        self.Name2=self.date+'-GSCF-2'
        self.Name3=self.date+'-GSCF-3'
    def GetKBR_TimeSeries(self):
        print(os.listdir(self.Path_KBR1))
        N1 = len(os.listdir(self.Path_KBR1))
        N2 = len(os.listdir(self.Path_KBR2))
        N3 = len(os.listdir(self.Path_KBR3))
        RMS_Set1, RMS_Set2, RMS_Set3 = [],[], []
        for i in np.arange(N1):
            load_path1 = self.Path_KBR1+'/RangeRate_{}.hdf5'.format(i)
            with h5py.File(load_path1,'r') as f1:
                data1=f1[list(f1.keys())[1]]
                data1 = np.array(data1)
                rms1 = np.sqrt(np.sum(np.power(data1,2),axis=0)/len(data1))
                RMS_Set1.append(rms1)
        for j in np.arange(N2):
            load_path2 =self.Path_KBR2+'/RangeRate_{}.hdf5'.format(j)
            with h5py.File(load_path2, 'r') as f:
                data = f[list(f.keys())[1]]
                data = np.array(data)
                rms2 = np.sqrt(np.sum(np.power(data, 2), axis=0) / len(data))
                RMS_Set2.append(rms2)
        for k in np.arange(N3):
            load_path3 = self.Path_KBR3+'/RangeRate_{}.hdf5'.format(k)
            with h5py.File(load_path3,'r') as f:
                data = f[list(f.keys())[1]]
                data = np.array(data)
                rms3 = np.sqrt(np.sum(np.power(data, 2), axis=0) / len(data))
                RMS_Set3.append(rms3)
            pass
        RMS_Set1 = np.array(RMS_Set1)
        RMS_Set2 = np.array(RMS_Set2)
        RMS_Set3 = np.array(RMS_Set3)
        return RMS_Set1,RMS_Set2,RMS_Set3

    def DrawKBR_TimeSeries(self):
        pygmt.config()
        RMS1, RMS2, RMS3 = self.GetKBR_TimeSeries()
        RMS1 = RMS1 * (10 ** 6)
        RMS2 = RMS2 * (10 ** 6)
        RMS3 = RMS3 * (10 ** 6)
        Dif1 = RMS2-RMS1
        difmax1=np.max(Dif1)
        difmin1=np.min(Dif1)
        print(difmin1,difmax1)
        Dif2 = RMS3-RMS1
        difmax2=np.max(Dif2)
        difmin2=np.min(Dif2)
        print(difmin2,difmax2)
        Dif3 = RMS3-RMS2
        difmax3=np.max(Dif3)
        difmin3=np.min(Dif3)
        print(difmin3,difmax3)
        x = np.linspace(1,31,len(RMS1))
        fig = pygmt.Figure()

        with fig.subplot(
                nrows=1, ncols=2, figsize=("15c", "4c"),
                frame=['afg', 'WSne'],
                # sharey=True,
                # sharex=True,
                # autolabel="(a)+jTL",
                autolabel=False,
                # autolabel="+JML+o0.5/0.5",
                # margins="0.5c",
                margins=["0.4c", "0.5c"],
        ):
            fig.basemap(
                region=[0,32,0.26,0.56],
                frame=['x+lday+f8p,Times-Bold','ya0.1f0.02g0.1+lKBRR residuals (um/s)+f8p,Times-Bold'],
                panel=[0, 0]
            )
            fig.plot(x=x,y=RMS1,pen='2p,grey',label='Normal')
            fig.plot(x=x,y=RMS2,pen='2p,blue',label='GSCF-2')
            fig.plot(x=x, y=RMS3, pen='2p,red', label='GSCF-3')
            fig.legend(position="JBL+o0c/1c+jTR")

            fig.basemap(
                frame=['WSen','x+lday+f12p,Times-Roman'],
                region=[0,32,-0.014,0.014],
                panel=[0,1]
            )
            fig.plot(x=x, y=Dif1, pen='2p,44/145/224', label='GSCF-2_normal')
            fig.plot(x=x, y=Dif2, pen='2p,58/191/153', label='GSCF-3_normal')
            # fig.plot(x=x, y=Dif3, pen='2p,240/167/58', label='GSCF-3_GSCF-2',region=[-0.015,0.015])
            fig.plot(x=x, y=Dif3, pen='2p,240/167/58', label='GSCF-3_GSCF-2')
            fig.legend(position="JBL+o0c/1c+jTR")
        # fig.shift_origin(yshift="h+1c")
        # data1 = np.load('{}/{}.npz'.format(self.setSpacePath().Path_GF, self.Name1))
        # data2 = np.load('{}/{}.npz'.format(self.setSpacePath().Path_GF, self.Name2))
        # data3 = np.load('{}/{}.npz'.format(self.setSpacePath().Path_GF, self.Name3))
        # lat = data1['lat']
        # lon = data1['lon']
        # dif = (data1['data']-data2['data'])*(10**6)
        # print(np.min(dif),np.max(dif))

        # with fig.subplot(nrows=1, ncols=1, figsize=("15c", "6c"), autolabel="(c)"):
        #     fig.basemap(
        #         region='g', projection='Q15c', frame=True, panel=[0, 0]
        #     )
        #     cmaps = ['purple,darkblue,blue,cyan,white,yellow,orange,red,pink']
        #     # cmaps = ['blue,cyan,white,yellow,red']
        #     pygmt.makecpt(cmap=cmaps, series=[-0.05, 0.05])
        #     # pygmt.makecpt(cmap='batlow')
        #     # fig.basemap(region='g', projection='Q10c', frame=['a', '+t GeoMod minus GeoPre'])
        #     fig.plot(x=lon, y=lat, fill=dif, style='c0.05c', cmap=True)
        #     # fig.colorbar(cmap=cmaps,frame=['a0.01', 'x+lDifference of KBRR residuals (um/s)'])
        #     fig.colorbar(frame=['x+lDifference of KBRR residuals (um/s)'])
        #     fig.coast(shorelines="1/0.5p, black")


        fig.savefig('../paper_figure/10-1.png')
        fig.show()

        fig.show()

        return Dif1,Dif2,Dif3
        pass

    def GetKBR_Space(self):
        Spatia_Data1 = self.Path_GF+'/{}.hdf5'.format(self.Name1)
        Spatia_Data2 = self.Path_GF+'/{}.hdf5'.format(self.Name2)
        Spatia_Data3 = self.Path_GF+'/{}.hdf5'.format(self.Name3)
        with h5py.File(Spatia_Data1,'r') as f1:
            print('Key: %s' % f1.keys())
            lat1 = np.array(list(f1[list(f1.keys())[0]]))
            lon1 = np.array(list(f1[list(f1.keys())[1]]))
            data1 = np.array(list(f1[list(f1.keys())[2]]))
        with h5py.File(Spatia_Data2,'r') as f2:
            lat2 = np.array(list(f2[list(f2.keys())[0]]))
            lon2 = np.array(list(f2[list(f2.keys())[1]]))
            data2 = np.array(list(f2[list(f2.keys())[2]]))
        with h5py.File(Spatia_Data3,'r') as f3:
            lat3 = np.array(list(f3[list(f3.keys())[0]]))
            lon3 = np.array(list(f3[list(f3.keys())[1]]))
            data3 = np.array(list(f3[list(f3.keys())[2]]))
        print(lat1.shape,lon1.shape,data1.shape)
        print(lat2.shape,lon2.shape,data2.shape)
        print(lat3.shape, lon3.shape, data3.shape)

        # for i in tqdm(np.arange(len(lat1))):
        #     if lat1[i] in lat2:
        #         continue
        #     else:
        #         lat1[i] = np.nan
        #         lon1[i] = np.nan
        #         data1[i] = np.nan
        # for j in tqdm(np.arange(len(lat2))):
        #     if lat2[j] in lat1:
        #         continue
        #     else:
        #         lat2[j] = np.nan
        #         lon2[j] = np.nan
        #         data2[j] = np.nan
        for k in tqdm(np.arange(len(lat3))):
            if lat3[k] in lat1:
                continue
            else:
                lat3[k] = np.nan
                lon3[k] = np.nan
                data3[k] = np.nan
        for n in tqdm(np.arange(len(lat1))):
            if lat1[n] in lat3:
                continue
            else:
                lat1[n] = np.nan
                lon1[n] = np.nan
                data1[n] = np.nan
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        data3 = data3[~np.isnan(data3)]
        lat1 = lat1[~np.isnan(lat1)]
        lon1 = lon1[~np.isnan(lon1)]
        lat2 = lat2[~np.isnan(lat2)]
        lon2 = lon2[~np.isnan(lon2)]
        lat3 = lat3[~np.isnan(lat3)]
        lon3 = lon3[~np.isnan(lon3)]
        np.savez('{}/{}.npz'.format(self.Path_GF,self.Name1),lat=lat1,lon=lon1,data=data1)
        np.savez('{}/{}.npz'.format(self.Path_GF,self.Name2),lat=lat2,lon=lon2,data=data2)
        np.savez('{}/{}.npz'.format(self.Path_GF, self.Name3), lat=lat3, lon=lon3, data=data3)
        print('The KBR data has been updated!')

    def DrawKBR_Space(self):
        data1 = np.load('{}/{}.npz'.format(self.Path_GF,self.Name1))
        data2 = np.load('{}/{}.npz'.format(self.Path_GF,self.Name2))
        data3 = np.load('{}/{}.npz'.format(self.Path_GF, self.Name3))
        print(np.min(data1['data']),np.max(data1['data']))
        print(np.min(data2['data']),np.max(data2['data']))
        print(np.min(data3['data']), np.max(data3['data']))
        lat = data1['lat']
        lon = data1['lon']
        dif = (data3['data']-data1['data'])*(10**6)
        print(np.min(dif),np.max(dif))
        fig = pygmt.Figure()
        fig.basemap(
            region='g', frame=True, projection='Q15c'
        )
        # cmaps = ['purple,darkblue,blue,cyan,white,yellow,orange,red,pink']
        cmaps = ['darkblue,blue,cyan,white,yellow,orange,red']
        # cmaps = ['blue,cyan,white,yellow,red']
        pygmt.makecpt(cmap=cmaps, series=[-0.05, 0.05])
        # pygmt.makecpt(cmap='batlow')
        # fig.basemap(region='g', projection='Q10c', frame=['a', '+t GeoMod minus GeoPre'])

        fig.plot(x=lon, y=lat, fill=dif, style='c0.05c', cmap=True)
        # fig.colorbar(cmap=cmaps,frame=['a0.01', 'x+lDifference of KBRR residuals (um/s)'])
        fig.colorbar(frame=['xa0.01f0.002+lDifference of KBRR residuals (um/s)'], position='JBC+o0c/1.2c+w15c/0.5c+ml')
        fig.coast(shorelines="1/0.5p, black")

        fig.savefig('../paper_figure/10-2.png')
        fig.show()

    def draw(self):
        # res1 = np.load("BeforeRes.npz", allow_pickle=True)
        res1 = h5py.File("../KBR_result/%s/result/%s-GSCF-2.hdf5"%(self.monthdata,self.date), "r")
        x1 = np.array(res1['lon'][:])
        y1 = np.array(res1['lat'][:])
        z1 = res1['postfit'][:] * 1e7

        # res = np.load("AfterRes.npz", allow_pickle=True)
        res = h5py.File("../KBR_result/%s/result/%s-GSCF-3.hdf5"%(self.monthdata,self.date), "r")
        x = np.array(res['lon'][:])
        y = np.array(res['lat'][:])
        z = res['postfit'][:] * 1e7

        grid1 = pygmt.xyz2grd(x=x1, y=y1, z=z1, spacing=(1, 1), region="d", duplicate='r')
        grid1 = pygmt.grdfill(grid=grid1, mode="n")

        grid = pygmt.xyz2grd(x=x, y=y, z=z, spacing=(1, 1), region="d", duplicate='r')
        grid = pygmt.grdfill(grid=grid, mode="n")

        fig = pygmt.Figure()
        # pygmt.makecpt(cmap=cpt, series=[0, 10, 2]) # color bar range

        with fig.subplot(
                nrows=2, ncols=1, subsize=("12c", "6c"),
                frame="afg",
                # sharey=True,
                # sharex=True,
                # autolabel="+JTL+o0.5/0.5",
                autolabel=False,
                # margins="0.5c",
                margins=["0.5c", "0.8c"],
        ):
            with fig.set_panel(panel=0):
                pygmt.makecpt(series=[0, 18, 2], background="red")  # color bar range
                # fig.basemap(projection="Q12c", frame=["+tKBRR residual"])

                fig.grdimage(grid=grid1, frame="f")
                fig.coast(region=[-180, 180, -90, 90], shorelines="1/0.5p", projection="Q12c")
                fig.basemap(frame=["f", "+t2020-03-GSCF-2"])
                # fig.colorbar(frame=["y+lum/s"])

            with fig.set_panel(panel=1):
                pygmt.makecpt(series=[0, 18, 2], background="red")  # color bar range
                fig.basemap(region="d", frame=["f", "+t2020-03-GSCF-3"])
                fig.grdimage(grid=grid)
                fig.coast(shorelines="1/0.5p", projection="Q12c")
                fig.colorbar(frame=["y+lum/s"])
        fig.show()


def demo():
    '''
    When drawing the spatial distribution of KBRR residuals,
    please first run GetKBR_Space() to get the update KBR files.
    Then, run the DrawKBR_Space() can get the figure.
    '''
    a = KBR(monthdata='2020-3')

    # a.draw()

    # a.setTimePath()
    # # a.GetKBR_TimeSeries()
    # a.DrawKBR_TimeSeries()

    a.setSpacePath()
    # a.GetKBR_Space()
    a.DrawKBR_Space()


def demo1():
    a = KBR(facotr='Tide')
    a.setPathRoot(rootpath='H:/Paper3/paper_result')
    a.DrawKBR_TimeSeries()

def demo2():
    a = KBR(facotr='Geo')
    a.setPathRoot(rootpath='H:/Paper3/paper_result/')
    # a.GetKBR_Space(Name1='2009-08-model',Name2='2009-08-pressure')
    # a.DrawKBR_Space(Name1='2009-08-model',Name2='2009-08-pressure')

if __name__ == '__main__':
    # demo2()
    demo()
