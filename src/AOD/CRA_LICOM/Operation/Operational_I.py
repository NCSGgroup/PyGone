import sys

import numpy as np


import os
from src.AOD.CRA_LICOM.Configure.GeoMathKit import GeoMathKit
from src.AOD.CRA_LICOM.Configure.LoadSH import AOD_GFZ
from src.AOD.CRA_LICOM.Configure.Format import CnmSnm, FormatWrite
from src.AOD.CRA_LICOM.Configure.Harmonic import Harmonic, LoveNumber, LoveNumberType
from src.AOD.CRA_LICOM.Configure.IntegralMethods import innerIn, InnerIntegral, InterpOption
from src.AOD.CRA_LICOM.Configure.LoadFields import LoadFields,DataType
from src.AOD.CRA_LICOM.Configure.Setting import AODtype, Constants, EllipsoidType, HarAnalysisType
from src.AOD.CRA_LICOM.Configure.GeopotentialHeight import GeopotentialHeight

import time as ti
import json
from src.AOD.CRA_LICOM.Configure.RefEllipsoid import RefEllipsoid
from src.AOD.CRA_LICOM.Configure.GeoidUndulation import GeoidUndulation
from src.AOD.CRA_LICOM.Configure.SurPres2CS import SurPres2CS
from src.AOD.CRA_LICOM.Configure.IntegralMethods import innerIn
from src.AOD.CRA_LICOM.Configure.IBcorrection import IBcorrection
from src.AOD.CRA_LICOM.Operation.TideFit import Detide,TideFit
import calendar
import datetime

import warnings

class SetConfigure():
    def __init__(self):
        self.GeoHeight = 'model level'
        self.Integral = 'SP'
        self.Path_DataLoad = 'H:/ERA5/model level/'
        self.Path_Tide = 'I:/Paper3/paper_result/tide/2019_2022/TidePressure/'
        self.Path_SP = 'I:/Paper3/paper_result/2_Scenario/sp/'
        self.Path_Upper = 'I:/Paper3/paper_result/upper_model/'
        self.LN = 'H:/Paper3/paper_data/Auxiliary/'

        self.daylist = None
        self.Nmax = 180,
        self.TimeEpoch = ["00:00:00", "03:00:00", "06:00:00",
                          "09:00:00", "12:00:00", "15:00:00",
                          "18:00:00", "21:00:00"]
        self.OnlyPressure = '1'
    def setGeoHeight(self,GeoHeight='model level'):
        self.GeoHeight = GeoHeight
        return self
    def setIntegral(self,Integral='SP'):
        self.Integral = Integral
        return self
    def setDataLoad(self,path='H:/Paper3/model level/'):
        self.Path_DataLoad = path
        return self
    def setPathTide(self,path='I:'):
        self.Path_Tide = path
        return self
    def setPathSP(self,path='I:/'):
        self.Path_SP = path
        return self
    def setPathUpper(self,path='I:/'):
        self.Path_Upper = path
        return self
    def setLoveNmuberPath(self,path='H:/Paper3/paper_data/Auxiliary/'):
        self.LN = path
        return self
    def setDuration(self,BeginDate='2022-01-01',EndDate='2022-12-31'):
        self.daylist = GeoMathKit.dayListByDay(begin=BeginDate,end=EndDate)
        return self.daylist
    def setMaxdegree(self,Nmax=180):
        self.Nmax = Nmax
        return self
    def setTimeEpoch(self,interval=3):
        TimeEpoch = []
        for i in np.arange(0, 24, interval):
            time = '{}:00:00'.format(str(i).rjust(2, '0'))
            TimeEpoch.append(time)
        self.TimeEpoch = TimeEpoch
        return self
    def setOnlyPressure(self,OnlyPressure=True):
        if OnlyPressure == True or OnlyPressure == '1' or OnlyPressure == 1:
            self.OnlyPressure = '1'
        elif OnlyPressure == False or OnlyPressure == '0' or OnlyPressure == 0:
            self.OnlyPressure = '0'
        return self

class OperationalRun(SetConfigure):
    def __init__(self):
        super().__init__()

    def SP(self):
        warnings.filterwarnings('ignore')
        TimeEpoch = self.TimeEpoch
        config_integral = InnerIntegral.defaultConfig(isWrite=True)
        Nmax = config_integral['MaxDegree']
        elltype = EllipsoidType[config_integral['Ellipsoid']]
        print(f"elltype:{elltype}")
        ell = RefEllipsoid(elltype)
        LN = LoveNumber(self.LN)
        loveNum = LN.getNumber(Nmax, LoveNumberType.Wang)
        ld = LoadFields(data_path=self.Path_DataLoad)
        ld.setTime()
        orography = ld.getField(DataType.PHISFC)*Constants.g_wmo
        lat, lon = ld.getLatLon_v2()
        PnmMat = GeoMathKit.getPnmMatrix(lat,Nmax, 2)
        hm = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)
        '''IBcorrection'''
        # ib = IBcorrection(lat, lon)
        '''remove tides'''
        dt = Detide()
        tides = {
            'P1': True,
            'S1': True,
            'K1': True,
            'N2': True,
            'M2': True,
            'L2': True,
            'T2': True,

            'S2': False,
            'R2': False,
            'T3': False,
            'S3': False,
            'R3': False,
        }
        dt.setTides(tides=tides, tideDir=self.Path_Tide)
        dt.setRefPoint(refpoint='2007-01-01,00:00:00')
        undulation = GeoidUndulation(elltype).getGeoid(lat, lon).flatten()

        '''Configure for the surface pressure integration'''

        sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
                                    loveNumber=LN)
        '''Configure for output'''
        if not os.path.exists(self.Path_SP):
            os.makedirs(self.Path_SP)
        fm_sp = FormatWrite().setRootDir(self.Path_SP)
        print('Save Path is:',self.Path_SP)

        daylist = self.daylist
        for day in daylist:
            date = day.strftime("%Y-%m-%d")
            print('---------Date: %s-------' % date)
            cs_sp = CnmSnm(date=date, Nmax=Nmax)

            for time in TimeEpoch:
                print('\nComputing: %s' % time)
                begin = ti.time()

                ld.setTime(date,time,OnlyPressure=True)

                print('Surface Integration...')
                sp = ld.getField(DataType.PSFC)
                '''SP integration'''
                deltaI_sp = sf2cs.setPressure_inner(pressure=sp, maxDeg=Nmax)

                '''Obtain the Surface Component'''
                print('Obtain the SP Component...')
                '''de-tide'''
                sp_af = dt.remove(pressure=sp, date=date, time=time)
                '''IB correction'''
                # sp_af = ib.correct(sp_af)

                deltaI_sp_removal = sf2cs.setPressure_inner(pressure=sp_af, maxDeg=Nmax)
                deltaI_sp_removal = deltaI_sp_removal.reshape((Nmax + 1, len(lat), len(lon)))
                cnm_sp, snm_sp = hm.analysis(Nmax=Nmax, Gqij=deltaI_sp_removal, lat=lat, lon=lon, PnmMat=PnmMat,
                                             kind=HarAnalysisType.InnerIntegral)

                '''record the stokes coefficients'''
                cs_sp.add(Cnm=cnm_sp, Snm=snm_sp,
                          epoch=time, date=date, attribute=AODtype.ATM.name)

                '''counting time'''
                print("Cost time: %s ms" % ((ti.time() - begin) * 1000))
                print('Finish!')

            '''write results'''
            fm_sp.setCS(cs_sp).AODstyle(date=date)
    def Upper(self):
        if not os.path.exists(self.Path_Upper):
            os.makedirs(self.Path_Upper)
        print('Save Path is:', self.Path_Upper)

        TimeEpoch = self.TimeEpoch
        config_integral = InnerIntegral.defaultConfig(isWrite=True)
        Nmax = config_integral['MaxDegree']

        elltype = EllipsoidType[config_integral['Ellipsoid']]
        ell = RefEllipsoid(elltype)
        LN = LoveNumber(self.LN)

        loveNum = LN.getNumber(Nmax, LoveNumberType.Wang)

        ld = LoadFields(data_path=self.Path_DataLoad)
        ld.setTime()
        orography = ld.getField(DataType.PHISFC)*Constants.g_wmo

        lat, lon = ld.getLatLon_v2()
        PnmMat = GeoMathKit.getPnmMatrix(lat, Nmax, 2)

        hm = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)
        undulation = GeoidUndulation(elltype).getGeoid(lat, lon).flatten()

        '''Configure for the surface pressure integration'''
        sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
                                    loveNumber=LN)
        '''Configure for output'''

        fm_upper = FormatWrite().setRootDir(self.Path_Upper)
        daylist = self.daylist

        for day in daylist:
            date = day.strftime("%Y-%m-%d")
            print('---------Date: %s-------' % date)
            cs_upper = CnmSnm(date=date, Nmax=Nmax)
            for time in TimeEpoch:
                print('\nComputing: %s' % time)
                begin = ti.time()
                ld.setTime(date, time, OnlyPressure=False)

                print('Vertical Integration...')
                gh = GeopotentialHeight(ld)
                gh.produce_z()
                delta_vi = innerIn(config_integral, gh, ld).setGeoid(Geoid=undulation).deltaI()


                print('Surface Integration...')

                sp = ld.getField(DataType.PSFC)
                '''SP integration'''
                deltaI_sp = sf2cs.setPressure_inner(pressure=sp, maxDeg=Nmax)

                '''Obtain the Upper Air Component'''
                print('Obtain the Upper Air Component...')
                delta_upper = (delta_vi - deltaI_sp) / (1 + loveNum[:, None])
                deltaI_upper = delta_upper.reshape((Nmax + 1, len(lat), len(lon)))
                cnm_upper, snm_upper = hm.analysis(Nmax=Nmax, Gqij=deltaI_upper, lat=lat, lon=lon, PnmMat=PnmMat,
                                                   kind=HarAnalysisType.InnerIntegral)


                '''record the stokes coefficients'''
                cs_upper.add(Cnm=cnm_upper, Snm=snm_upper,
                             epoch=time, date=date, attribute=AODtype.ATM.name)


                '''counting time'''
                print("Cost time: %s ms" % ((ti.time() - begin) * 1000))
                print('Finish!')

            '''write results'''
            fm_upper.setCS(cs_upper).AODstyle(date=date)

    @staticmethod
    def DefaultConfig(isWrite=True):
        config = {
                  'Integral':'both',
                  'BeginDate':'2002-01-01',
                  'EndDate':'2020-12-31',
                  'PathDataLoad':'H:/Paper3/model/',
                  'PathTide':'I:/Paper3/paper_result/tide/2019_2020/TidePressure/',
                  'Path_SP':'I:/Paper3/paper_result/2_Scenario/sp/',
                  'Path_Upper':'I:/Paper3/paper_result/upper_model/',
                  'Path_LN':'H:/Paper3/paper_data/Auxiliary',
                  'MaxDegree':180,
                  'Interval':3,
                  }
        if isWrite:
            with open('Operational_I.json', 'w') as f:
                f.write(json.dumps(config,indent=11))
        return config


def demo1():
    a = OperationalRun()
    a.DefaultConfig()
def demo_json():
    with open('Operational_I.json','r') as f:
        config = json.load(f)
    a = OperationalRun()
    a.setDataLoad(path=config['PathDataLoad'])
    a.setDuration(BeginDate=config['BeginDate'],EndDate=config['EndDate'])
    a.setPathTide(path=config['PathTide'])
    a.setPathSP(path=config['Path_SP'])
    a.setPathUpper(path=config['Path_Upper'])
    a.setLoveNmuberPath(path=config['Path_LN'])
    a.setMaxdegree(Nmax=config['MaxDegree'])
    a.setTimeEpoch(interval=config['Interval'])
    if config['Integral'] == 'SP' or config['Integral'] == 'sp':
        a.SP()
    elif config['Integral'] == 'Upper' or config['Integral'] == 'Upp' or config['Integral'] == 'upper':
        a.Upper()
    elif config['Integral'] == 'Both' or config['Integral'] == 'both' or config['Integral'] == 'All':
        a.SP()
        a.Upper()

def demo111():
    a = RefEllipsoid(ellipsoid=EllipsoidType.GRS80_IERS2010)


if __name__ == '__main__':
    demo_json()
    # demo111()

