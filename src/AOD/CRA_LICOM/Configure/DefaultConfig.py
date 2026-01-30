import numpy as np
from src.AOD.CRA_LICOM.Configure.GeoMathKit import GeoMathKit
from src.AOD.CRA_LICOM.Configure.Harmonic import Harmonic,SynthesisType
from src.AOD.CRA_LICOM.Configure.LoveNumber import LoveNumber,LoveNumberType
from tqdm import tqdm
import os

class Config():
    def __init__(self):
        self.Path = None
        self.Path2 = None
        self.Nmax = 180
        self.BeginDate = None
        self.EndDate = None
        self.daylist = None
        self.TimeEpoch = None
        self.lat = np.arange(90,-90.1,-1)
        self.lon = np.arange(0,360,1)
        self.SavePath = None
        self.Grid = 0.5
        self.Save_Name = None
        self.Name_Series = None
        self.interval = None
        self.LovePath = '../../data/Auxiliary/'
        self.LN = LoveNumber(self.LovePath)
        self.HM = Harmonic(self.LN).setLoveNumMethod(LoveNumberType.Wang)
        self.Seriess = []
        self.ticks = None
        self.strss = None
        self.titles = None
        self.color = None

    def setColor(self,colors):
        self.color = colors
        return self

    def setTitles(self,titles):
        self.titles = titles
        return self

    def setLovePath(self,path):
        self.LovePath = path
        self.LN = LoveNumber(self.LovePath)
        self.HM = Harmonic(self.LN).setLoveNumMethod(LoveNumberType.Wang)
        return self

    def setPath(self,path='H:/AOD1B07/'):
        self.Path = path
        return self
    def setPath2(self,path):
        self.Path2 = path
        return self
    def setDuration(self,BeginDate='2020-01-01',EndDate='2020-01-31'):
        self.BeginDate = BeginDate
        self.EndDate = EndDate
        daylist = GeoMathKit.dayListByDay(begin=BeginDate,end=EndDate)
        self.daylist = daylist
        return self
    def setInterval(self,interval=3):
        TimeEpoch = []
        self.interval = interval
        for i in np.arange(0,24,interval):
            time = '{}:00:00'.format(str(i).rjust(2,'0'))
            TimeEpoch.append(time)
        self.TimeEpoch = TimeEpoch
        return self

    def setMaxdegree(self,Nmax=180):
        self.Nmax = Nmax
        return self

    def setSeriess(self, series):
        self.Seriess = series
        return self

    def setLatLon(self,grid=0.5):
        self.Grid = grid
        self.lat = np.arange(90,-90.1,-grid)
        self.lon = np.arange(0,360,grid)
        return self

    def setSavePath(self,path='H:/Paper3/paper_result/mean_sp/'):
        self.SavePath = path
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)
        print(f'Save path is: {self.SavePath}')
        return self
    def setSaveName(self,save_name):
        self.Save_Name = save_name
        return self
    def setNameSeries(self,nameseries):
        self.Name_Series = nameseries
        return self
    def setTicks(self,ticks):
        self.ticks=ticks
        return self

    def setStrss(self,strss):
        self.strss = strss
        return self

    def get_CS_Grid(self,value: np.ndarray):
        lat = self.lat
        lon = self.lon
        # print(len(lon))
        Nmax = self.Nmax
        PnmMat = GeoMathKit.getPnmMatrix(lat, Nmax, 2)
        Space = self.HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(value[0]), Sqlm=GeoMathKit.CS_1dTo2d(value[1]),
                                  PnmMat=PnmMat, lat=lat, lon=lon, Nmax=Nmax, kind=SynthesisType.Pressure)
        return Space

    def get_Area_weight(self,lat,value: np.ndarray):
        # lat = self.lat
        lat = lat
        # Nlon = len(self.lon)
        Nlon = len(value[0,:])
        weights = np.cos(np.deg2rad(lat))
        weights_2d = np.repeat(weights[:, np.newaxis], Nlon, axis=1)
        weighted_sum = np.sum(value * weights_2d)
        total_weight = np.sum(weights_2d)
        area_weight_average = weighted_sum / total_weight
        return area_weight_average

    def get_RMS(self,value: np.ndarray):
        rms = np.sqrt(np.mean(value**2,axis=0))
        return rms

    def get_STD(self,value:np.ndarray):
        std = np.std(value,axis=0)
        return std

def demo():
    a = Config()
    a.setLatLon(grid=1)
    a.get_Area_weight()