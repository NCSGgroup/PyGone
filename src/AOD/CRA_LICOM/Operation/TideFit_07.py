"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/10 10:15
@Description:
"""

import sys

sys.path.append('../Configure')

from src.AOD.CRA_LICOM.Configure.GeoMathKit import GeoMathKit
from scipy import signal
from src.AOD.CRA_LICOM.Configure.LoadFields import LoadFields, DataType
from src.AOD.CRA_LICOM.Configure.Harmonic import Harmonic, LoveNumber, LoveNumberType, HarAnalysisType
from src.AOD.CRA_LICOM.Configure.Format import FormatWrite, CnmSnm
from src.AOD.CRA_LICOM.Configure.SurPres2CS import SurPres2CS, RefEllipsoid, EllipsoidType
from src.AOD.CRA_LICOM.Configure.Setting import Constants
from src.AOD.CRA_LICOM.Configure.GeoidUndulation import GeoidUndulation
import numpy as np
import os
import json


class TideFit:
    """Only For CRA 6 hours"""
    tideFreq = {

        'Pi1':14.91786609,
        'P1': 14.95893277,
        'S1': 15.00000141,
        'K1': 15.04107005,
        'Psi1':15.08213673,
        # 'N2': 28.4397295,
        'M2': 28.98410705,
        # 'L2': 29.5284789,
        'T2': 29.95893612,
        'S2': 30.00000282,
        'R2': 30.04106950,
        'K2': 30.08214010,
        'T3': 44.95893559,
        'S3': 45.00000423,
        'R3': 45.04107287,
        'S4': 60.00000564,
        'S5': 75.00000705,
        'S6': 90.00000846,


    }
    ref_point = '2007-01-01, 00:00:00'  # 00:00:00
    ref_point_mjd = 54101.000754444445  # MJD of the ref point.

    def __init__(self):
        self._timeEpoch = ["00:00:00","01:00:00","02:00:00","03:00:00",
                           "04:00:00","05:00:00","06:00:00","07:00:00",
                           "08:00:00","09:00:00","10:00:00","11:00:00",
                           "12:00:00","13:00:00","14:00:00","15:00:00",
                           "16:00:00","17:00:00","18:00:00","19:00:00",
                           "20:00:00","21:00:00","22:00:00","23:00:00",]
        self._sr = 1 / (int(self._timeEpoch[1][:2]) - int(self._timeEpoch[0][:2]))

        self._daylist = None
        self._dataDirIn = None
        self._dataDirOut = None
        self._butterworth = None

    def setDuration(self, begin='2007-01-01', end='2007-12-31'):
        """
        set the duration that tide fits will last
        :param begin:
        :param end:
        :return:
        """
        self._daylist = GeoMathKit.dayListByDay(begin, end)
        return self

    def setDataDir(self, dataDirIn: str, dataDirOut: str):
        """
        set the directory of surface pressure data deployed and the output tides
        :param dataDirIn:
        :param dataDirOut:
        :return:
        """
        self._dataDirIn = dataDirIn
        self._dataDirOut = dataDirOut

        isExists = os.path.exists(self._dataDirOut)
        if not isExists:
            os.makedirs(self._dataDirOut)
        return self

    def setButterworth(self, cutoff=3, order=3):
        """
        define the butterworth highpass filter
        :param cutoff: cutoff-frequency, [days], eg. cutoff=3
        :param order: order of the butterworth filter
        :return:
        """

        cutoff = 1 / (cutoff * 24)
        wn = 2 * cutoff / self._sr
        assert wn <= 1
        self._butterworth = signal.butter(order, wn, 'high')
        return self

    def fit(self):
        """
        :return:
        """
        Tide = []
        b, a = self._butterworth[0], self._butterworth[1]
        SR = int(self._timeEpoch[1][:2]) - int(self._timeEpoch[0][:2])

        ld = LoadFields(data_path=self._dataDirIn)
        ld.setTime_for_tide()
        Nlat, NLon = ld.getNlatNlon()

        Nlen = 0
        for date in self._daylist:
            for time in self._timeEpoch:
                Nlen += 1

        interval = np.round(1 / self._sr)

        refDate = TideFit.ref_point.split(',')[0]
        refTime = int(TideFit.ref_point.split(',')[1].split(':')[0])

        date = self._daylist[0]
        days = len(GeoMathKit.dayListByDay(begin=refDate, end=date.strftime("%Y-%m-%d")))
        '''calculate how may hours between the given time and the starting point'''
        if days == 0:
            days = len(GeoMathKit.dayListByDay(begin=date.strftime("%Y-%m-%d"), end=refDate))
            days = days * (-1)
            hours = (days + 1) * 24 - refTime
        else:
            hours = (days - 1) * 24 - refTime

        xdata = np.arange(Nlen) * interval + hours

        dm = self.designMatrix(xdata)
        if interval == 6:
            dm = dm[:, :16]
            # dm = dm[:, :18]

        # for indexByLat in range(Nlat):
        #     print('Progress: {:.1%}'.format(indexByLat / Nlat))
        Assemble = []
        N = 0
        Ntoll = len(self._daylist)
        for date in self._daylist:
            print('Progress: {:.1%}'.format(N / Ntoll))
            for time in self._timeEpoch:
                ld.setTime_for_tide(date=date.strftime("%Y-%m-%d"), time=time)

                Assemble.append(ld.getField(kind=DataType.PSFC).flatten())
            N += 1

        Assemble = np.array(Assemble)

        '''butterworth filter'''
        sf = signal.filtfilt(b, a, Assemble, axis=0)
        pass

        sf_arr = np.array(sf)
        pp = np.linalg.lstsq(dm.copy(), sf_arr)[0]
        # Tide.append(pp)

        '''Complete'''

        Tide = np.array(pp)
        Tide = Tide.reshape(len(Tide[:, 0]), 361, 720).transpose(1, 0, 2)

        np.save(self._dataDirOut + 'All_grid.npy', Tide)

        pass

    def designMatrix(self, xdata):
        """
        use the least square to speed-up computation.
        :param x:
        :return:
        """
        pi1 = np.deg2rad(self.tideFreq['Pi1'])
        p1 = np.deg2rad(self.tideFreq['P1'])
        s1 = np.deg2rad(self.tideFreq['S1'])
        k1 = np.deg2rad(self.tideFreq['K1'])
        psi1 = np.deg2rad(self.tideFreq['Psi1'])

        # n2 = np.deg2rad(self.tideFreq['N2'])
        m2 = np.deg2rad(self.tideFreq['M2'])
        # l2 = np.deg2rad(self.tideFreq['L2'])

        t2 = np.deg2rad(self.tideFreq['T2'])
        s2 = np.deg2rad(self.tideFreq['S2'])
        r2 = np.deg2rad(self.tideFreq['R2'])

        k2 = np.deg2rad(self.tideFreq['K2'])
        t3 = np.deg2rad(self.tideFreq['T3'])
        s3 = np.deg2rad(self.tideFreq['S3'])
        r3 = np.deg2rad(self.tideFreq['R3'])

        s4 = np.deg2rad(self.tideFreq['S4'])
        s5 = np.deg2rad(self.tideFreq['S5'])
        s6 = np.deg2rad(self.tideFreq['S6'])

        a = np.ones(len(xdata))
        b = xdata
        Pi1_c = np.cos(xdata * pi1)
        Pi1_s = np.sin(xdata * pi1)
        P1_c = np.cos(xdata * p1)
        P1_s = np.sin(xdata * p1)
        S1_c = np.cos(xdata * s1)
        S1_s = np.sin(xdata * s1)
        K1_c = np.cos(xdata * k1)
        K1_s = np.sin(xdata * k1)
        Psi1_c = np.cos(xdata * psi1)
        Psi1_s = np.sin(xdata * psi1)
        # N2_c = np.cos(xdata * n2)
        # N2_s = np.sin(xdata * n2)
        M2_c = np.cos(xdata * m2)
        M2_s = np.sin(xdata * m2)
        # L2_c = np.cos(xdata * l2)
        # L2_s = np.sin(xdata * l2)
        T2_c = np.cos(xdata * t2)
        T2_s = np.sin(xdata * t2)
        S2_c = np.cos(xdata * s2)
        S2_s = np.sin(xdata * s2)
        R2_c = np.cos(xdata * r2)
        R2_s = np.sin(xdata * r2)
        K2_c = np.cos(xdata * k2)
        K2_s = np.sin(xdata * k2)
        T3_c = np.cos(xdata * t3)
        T3_s = np.sin(xdata * t3)
        S3_c = np.cos(xdata * s3)
        S3_s = np.sin(xdata * s3)
        R3_c = np.cos(xdata * r3)
        R3_s = np.sin(xdata * r3)
        S4_c = np.cos(xdata * s4)
        S4_s = np.sin(xdata * s4)
        S5_c = np.cos(xdata * s5)
        S5_s = np.sin(xdata * s5)
        S6_c = np.cos(xdata * s6)
        S6_s = np.sin(xdata * s6)

        dm = [a, b,
              Pi1_c, Pi1_s,
              P1_c, P1_s,
              S1_c, S1_s,
              K1_c, K1_s,
              # N2_c, N2_s,
              Psi1_c, Psi1_s,
              M2_c, M2_s,
              # L2_c, L2_s,
              T2_c, T2_s,
              S2_c, S2_s,
              R2_c, R2_s,
              K2_c, K2_s,
              T3_c, T3_s,
              S3_c, S3_s,
              R3_c, R3_s,
              S4_c, S4_s,
              S5_c, S5_s,
              S6_c, S6_s,
              ]

        dm = np.array(dm).transpose()
        return dm

    def separateTidePres(self, isLessThanSix=False):

        AllTide = np.load(self._dataDirOut + 'All_grid.npy')

        out = self._dataDirOut + 'TidePressure/'
        isExists = os.path.exists(out)
        if not isExists:
            os.makedirs(out)
        pi1Cos = AllTide[:,2,:]
        pi1Sin = AllTide[:,3,:]
        np.save(out + 'Pi1.npy', np.array([pi1Cos, pi1Sin]))

        p1Cos = AllTide[:, 4, :]
        p1Sin = AllTide[:, 5, :]
        np.save(out + 'P1.npy', np.array([p1Cos, p1Sin]))

        s1Cos = AllTide[:, 6, :]
        s1Sin = AllTide[:, 7, :]
        np.save(out + 'S1.npy', np.array([s1Cos, s1Sin]))

        k1Cos = AllTide[:, 8, :]
        k1Sin = AllTide[:, 9, :]
        np.save(out + 'K1.npy', np.array([k1Cos, k1Sin]))

        psi1Cos = AllTide[:,10,:]
        psi1Sin = AllTide[:,11,:]
        np.save(out+'Psi1.npy', np.array([psi1Cos, psi1Sin]))


        # n2Cos = AllTide[:, 8, :]
        # n2Sin = AllTide[:, 9, :]
        # np.save(out + 'N2.npy', np.array([n2Cos, n2Sin]))

        m2Cos = AllTide[:, 12, :]
        m2Sin = AllTide[:, 13, :]
        np.save(out + 'M2.npy', np.array([m2Cos, m2Sin]))

        # l2Cos = AllTide[:, 12, :]
        # l2Sin = AllTide[:, 13, :]
        # np.save(out + 'L2.npy', np.array([l2Cos, l2Sin]))

        t2Cos = AllTide[:, 14, :]
        t2Sin = AllTide[:, 15, :]
        np.save(out + 'T2.npy', np.array([t2Cos, t2Sin]))

        if isLessThanSix:
            s2Cos = AllTide[:, 16, :]
            s2Sin = AllTide[:, 17, :]
            np.save(out + 'S2.npy', np.array([s2Cos, s2Sin]))

            r2Cos = AllTide[:, 18, :]
            r2Sin = AllTide[:, 19, :]
            np.save(out + 'R2.npy', np.array([r2Cos, r2Sin]))

            k2Cos = AllTide[:,20,:]
            k2Sin = AllTide[:,21,:]
            np.save(out+ 'K2.npy', np.array([k2Cos, k2Sin]))

            t3Cos = AllTide[:, 22, :]
            t3Sin = AllTide[:, 23, :]
            np.save(out + 'T3.npy', np.array([t3Cos, t3Sin]))

            s3Cos = AllTide[:, 24, :]
            s3Sin = AllTide[:, 25, :]
            np.save(out + 'S3.npy', np.array([s3Cos, s3Sin]))

            r3Cos = AllTide[:, 26, :]
            r3Sin = AllTide[:, 27, :]
            np.save(out + 'R3.npy', np.array([r3Cos, r3Sin]))

            s4Cos = AllTide[:,28,:]
            s4Sin = AllTide[:,29,:]
            np.save(out+ 'S4.npy', np.array([s4Cos, s4Sin]))

            s5Cos = AllTide[:,30,:]
            s5Sin = AllTide[:,31,:]
            np.save(out+ 'S5.npy', np.array([s5Cos, s5Sin]))

            s6Cos = AllTide[:,32,:]
            s6Sin = AllTide[:,33,:]
            np.save(out+ 'S6.npy', np.array([s6Cos,s6Sin]))

        pass

    def generateTide_bySphere(self, tides: dict, Nmax: int, lat, lon):
        """
        The tide is generated assuming the Earth is sphere
        :param Nmax: Max degree of SH expansion for the tide
        :param tides:
        example
        tides = {
            'S1': True,
            'S2': True,
            'M2': False
        }
        :param lat: e.g., np.arange(90, -90.1, -0.5)
        :param lon: e.g., np.arange(0, 360, 0.5)
        :return:
        """

        indir = self._dataDirOut + 'TidePressure/'
        outdir = self._dataDirOut + 'TideGeoCS_sphere/'

        isExists = os.path.exists(outdir)
        if not isExists:
            os.makedirs(outdir)

        tidesInfo = {}
        tidesDoodsonMatrix = {}

        LN = LoveNumber('../../../Data/Auxiliary/')
        HM = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)

        MaxDeg = Nmax
        nutarg_first = GeoMathKit.doodsonArguments(TideFit.ref_point_mjd)
        Pnm = GeoMathKit.getPnmMatrix(lat, Nmax=MaxDeg, option=1)
        fm = FormatWrite().setRootDir(outdir)

        for tide in tides.keys():
            if tides[tide]:
                # self.__tidesInfo[tide] = np.load(tideDir + tide + '_grid' + '.npy')
                tidesInfo[tide] = np.load(indir + tide + '.npy')
                tidesDoodsonMatrix[tide] = self.__doodsonMatrix(tide)

        for tide in tidesInfo.keys():
            print(tide)
            cs = CnmSnm(date=tide, Nmax=MaxDeg)
            st = tidesInfo[tide]
            # A = st[:, :, 0].flatten()
            # B = st[:, :, 1].flatten()
            A = st[0, :, :]
            B = st[1, :, :]

            wt0 = np.matmul(tidesDoodsonMatrix[tide], nutarg_first)
            Anew = A * np.cos(wt0) - B * np.sin(wt0)
            Bnew = A * np.sin(wt0) + B * np.cos(wt0)
            cnmCos, snmCos = HM.analysis(Nmax=MaxDeg, Gqij=Anew, lat=lat, lon=lon, PnmMat=Pnm,
                                         kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmCos, Snm=snmCos, epoch='00:00:00', date=tide, attribute='cos')

            cnmSin, snmSin = HM.analysis(Nmax=MaxDeg, Gqij=Bnew, lat=lat, lon=lon, PnmMat=Pnm,
                                         kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmSin, Snm=snmSin, epoch='01:00:00', date=tide, attribute='sin')
            '''format writing'''
            '''write results'''
            fm.setCS(cs).TideStyle(tide)

        pass

    def generateTide_byTopography(self, tides: dict, Nmax: int, lat, lon):
        """
        The tide is generated considering an actual Earth.
        :param Nmax: Max degree of SH expansion for the tide
        :param tides:
        example
        tides = {
            'S1': True,
            'S2': True,
            'M2': False
        }
        :param lat: e.g., np.arange(90, -90.1, -0.5)
        :param lon: e.g., np.arange(0, 360, 0.5)
        :return:
        """

        indir = self._dataDirOut + 'TidePressure/'
        outdir = self._dataDirOut + 'TideGeoCS_topography/'

        '''Configure for the surface pressure integration'''
        LN = LoveNumber('H:/Paper3/paper_data/Auxiliary/')
        ell = RefEllipsoid(EllipsoidType.GRS80_IERS2010)
        undulation = GeoidUndulation(EllipsoidType.GRS80_IERS2010).getGeoid(lat, lon).flatten()
        ld = LoadFields(data_path='H:/ERA5/model level/')
        ld.setTime()
        orography = ld.getField(DataType.PHISFC) * Constants.g_wmo
        sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
                                    loveNumber=LN)

        isExists = os.path.exists(outdir)
        if not isExists:
            os.makedirs(outdir)

        tidesInfo = {}
        tidesDoodsonMatrix = {}

        LN = LoveNumber('H:/Paper3/paper_data/Auxiliary/')
        HM = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)

        MaxDeg = Nmax
        nutarg_first = GeoMathKit.doodsonArguments(TideFit.ref_point_mjd)
        Pnm = GeoMathKit.getPnmMatrix(lat, Nmax=MaxDeg, option=1)
        fm = FormatWrite().setRootDir(outdir)

        for tide in tides.keys():
            if tides[tide]:
                # self.__tidesInfo[tide] = np.load(tideDir + tide + '_grid' + '.npy')
                tidesInfo[tide] = np.load(indir + tide + '.npy')
                tidesDoodsonMatrix[tide] = self.__doodsonMatrix(tide)

        for tide in tidesInfo.keys():
            print(tide)
            cs = CnmSnm(date=tide, Nmax=MaxDeg)
            st = tidesInfo[tide]
            # A = st[:, :, 0].flatten()
            # B = st[:, :, 1].flatten()
            A = st[0, :, :]
            B = st[1, :, :]

            wt0 = np.matmul(tidesDoodsonMatrix[tide], nutarg_first)
            Anew = A * np.cos(wt0) - B * np.sin(wt0)
            Bnew = A * np.sin(wt0) + B * np.cos(wt0)

            cnmCos, snmCos = sf2cs.setPressure(pressure=Anew, maxDeg=MaxDeg)
            # cnmCos, snmCos = HM.analysis(Nmax=MaxDeg, Gqij=Anew, lat=lat, lon=lon, PnmMat=Pnm,
            #                              kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmCos, Snm=snmCos, epoch='00:00:00', date=tide, attribute='cos')

            cnmSin, snmSin = sf2cs.setPressure(pressure=Bnew, maxDeg=MaxDeg)
            # cnmSin, snmSin = HM.analysis(Nmax=MaxDeg, Gqij=Bnew, lat=lat, lon=lon, PnmMat=Pnm,
            #                              kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmSin, Snm=snmSin, epoch='01:00:00', date=tide, attribute='sin')
            '''format writing'''
            '''write results'''
            fm.setCS(cs).TideStyle(tide)

        pass

    def __doodsonMatrix(self, tide='S1'):

        DoodsonNumber = {'Pi1': '141.000','S1': '065.555', 'S2': '075.555', 'Sa': '056.554', 'Ssa': '057.555',
                         'P1': '163.555', 'K1': '165.555', 'Psi1': '145.555', 'N2': '245.655', 'M2': '255.555',
                         'L2': '265.455', 'T2': '272.556', 'K2': '273.555', 'R2': '274.554', 'S4': '095.555',
                         'T3': '381.555', 'S3': '085.555', 'R3': '383.555', 'S5': '105.555', 'S6': '115.555'}

        doodson = DoodsonNumber[tide]

        doodsonMatrix = np.zeros(6)

        doodsonMatrix[0] = int(doodson[0]) - 0
        doodsonMatrix[1] = int(doodson[1]) - 5
        doodsonMatrix[2] = int(doodson[2]) - 5
        doodsonMatrix[3] = int(doodson[4]) - 5
        doodsonMatrix[4] = int(doodson[5]) - 5
        doodsonMatrix[5] = int(doodson[6]) - 5

        return doodsonMatrix

    @staticmethod
    def DefaultConfig(isWrite=True):
        config = {'dataDirIn': '../../../Data/ERA5/model level/',
                  'dataDirOut': '../../../Result/Paper3/model_tide_07/2007_2014/',
                  'BeginDate':'2007-01-01',
                  'EndDate':'2014-12-31'}

        if isWrite:
            with open('../Settings/TideFit_07.json', 'w') as f:
                f.write(json.dumps(config,indent=4))
        return config

class Detide:

    def __init__(self):

        self.__refpoint = TideFit.ref_point
        self.__tidesInfo = {}
        pass

    def setTides(self, tides: dict, tideDir: str):
        """
        :param tides:
        example
        tides = {
            'S1': True,
            'S2': True,
            'M2': False
        }
        :param tideDir:
        :return:
        """
        for tide in tides.keys():
            if tides[tide]:
                # self.__tidesInfo[tide] = np.load(tideDir + tide + '_grid' + '.npy')
                self.__tidesInfo[tide] = np.load(tideDir + tide + '.npy')

        return self

    def setRefPoint(self, refpoint):
        """
        for easy use, the tide phase are derived with self-defined time system, inferring that the
        de-tide process needs the info of start point to get the tide value at given time.
        :param refpoint: e.g., '2007-01-01,00:00:00'
        :return:
        """
        self.__refpoint = refpoint
        return self

    def remove(self, pressure, date: str, time: str):
        """
        remove tides from the given pressure
        :param pressure: input pressure with the same lat and lon with tide grids, see Tidefit.cls
        :param date: date of the pressure
        :param time: time of the pressure like 18:00:00; the basic unit is hour.
        :return: pressure after de-tides. One-dimension [N*M]
        """
        refDate = self.__refpoint.split(',')[0]
        refTime = int(self.__refpoint.split(',')[1].split(':')[0])
        endTime = int(time.split(':')[0])

        days = len(GeoMathKit.dayListByDay(begin=refDate, end=date))
        '''calculate how may hours between the given time and the starting point'''
        if days == 0:
            days = len(GeoMathKit.dayListByDay(begin=date, end=refDate))
            days = days * (-1)
            hours = (days + 1) * 24 + endTime - refTime
        else:
            hours = (days - 1) * 24 + endTime - refTime

        for tide in self.__tidesInfo.keys():
            st = self.__tidesInfo[tide]
            wt = np.deg2rad(TideFit.tideFreq[tide]) * hours
            pressure -= st[0, :, :].flatten() * np.cos(wt) + st[1, :, :].flatten() * np.sin(wt)
            # pressure -= st[:, :, 0] * np.cos(wt) + st[:, :, 1] * np.sin(wt)
            pass

        return pressure


def demo():
    cutoff = 3
    order = 3
    cutoff = 1 / (cutoff * 24)
    wn = 2 * cutoff / 6
    assert wn <= 1
    b, a = signal.butter(order, wn, 'high')

    pass

  # tf.setDataDir(dataDirIn='../data/CRA.grib2/', dataDirOut='../result/tide/2007_2014/')
    # tf.setDuration(begin='2007-01-01', end='2014-12-31')

    # tf.setDataDir(dataDirIn='../data/CRA.grib2/',dataDirOut='../result/tide/2012_2019/')
    # tf.setDuration(begin='2012-01-01', end='2019-12-31')
def demo2():
    tf = TideFit().setButterworth()
    tf.setDataDir(dataDirIn='H:/ERA5/model level/', dataDirOut='H:/Paper3/paper_result/model_tide/2007/')
    tf.setDuration(begin='2007-01-01', end='2007-01-10')
    tf.fit()
    tf.separateTidePres(isLessThanSix=True)
    tides = {
        'Pi1':True,
        'P1': True,
        'S1': True,
        'K1': True,
        'Psi1':True,
        # 'N2': True,
        'M2': True,
        # 'L2': True,
        'T2': True,
        'S2': True,
        'R2': True,
        'K2': True,
        'T3': True,
        'S3': True,
        'R3': True,
        'S4': True,
        'S5': True,
        'S6': True,
    }
    lat = np.arange(90, -90.1, -0.5)
    lon = np.arange(0, 360, 0.5)
    tf.generateTide_bySphere(tides=tides, Nmax=180, lat=lat, lon=lon)
    tf.generateTide_byTopography(tides=tides, Nmax=180, lat=lat, lon=lon)

def demo3():
    with open('../Settings/TideFit_07.json', 'r') as f:
        config = json.load(f)
    tf = TideFit().setButterworth()
    tf.setDataDir(dataDirIn=config['dataDirIn'], dataDirOut=config['dataDirOut'])
    tf.setDuration(begin=config['BeginDate'], end=config['EndDate'])
    tf.fit()
    tf.separateTidePres(isLessThanSix=True)
    tides = {
        'Pi1': True,
        'P1': True,
        'S1': True,
        'K1': True,
        'Psi1': True,
        # 'N2': True,
        'M2': True,
        # 'L2': True,
        'T2': True,
        'S2': True,
        'R2': True,
        'K2': True,
        'T3': True,
        'S3': True,
        'R3': True,
        'S4': True,
        'S5': True,
        'S6': True,
    }
    lat = np.arange(90, -90.1, -0.5)
    lon = np.arange(0, 360, 0.5)
    tf.generateTide_bySphere(tides=tides, Nmax=180, lat=lat, lon=lon)
    # tf.generateTide_byTopography(tides=tides, Nmax=180, lat=lat, lon=lon)
def demo4():
    a = TideFit()
    a.DefaultConfig()
if __name__ == '__main__':
    demo3()
    # demo_tide_res()
