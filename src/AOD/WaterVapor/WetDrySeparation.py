"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/11/16 下午4:23
@Description: Separate the wet and dry air in the scale of globe, with the time resolution of one-month
"""

from ExtractNC import ReadNC, DataType
import pandas as pd
import numpy as np
from Setting import Constants


class WetDryMl_ERA5:
    model_level_file = '../data/Auxiliary/Model137.csv'

    def __init__(self):
        self._level = None
        self._modelLevel = None
        self._readModelLevel(WetDryMl_ERA5.model_level_file)
        self._lat, self._lon = None, None
        self._SHUM, self._PSFC = None, None
        pass

    def setDate(self, date='2017-02'):
        """
        get the wet and dry air at given date
        :param date: str, e.g., '2017-02'
        :return:
        """
        pathRt1 = '../data/ModelLevel/ERA5/'
        pathRt2 = '../data/PressureLevel/ERA5/'

        SHUM = 'SHUM_ERA5_yearly_' + date[0:4] + '.nc'
        PSFC = 'PSFC_ERA5_monthly_' + date[0:4] + date[5:7] + '.nc'

        rd = ReadNC()
        rd.setPar(file=pathRt1 + SHUM, datatype=DataType.SHUM)
        self._SHUM, lat, lon = rd.read(seqN=int(date[5:7]) - 1)

        rd = ReadNC()
        rd.setPar(file=pathRt2 + PSFC, datatype=DataType.PSFC)
        self._PSFC, lat2, lon2 = rd.read(seqN=0)

        self._lat, self._lon = lat, lon

        # Mean_Ps = self.getGloMean(PSFC)
        # Tot_mass = self.getTotMass(Mean_Ps)

        pass

    def _readModelLevel(self, file: str):
        """

        :param file: full path of the file
        :return:
        """
        a = pd.read_csv(file)['a [Pa]'].tolist()
        b = pd.read_csv(file)['b'].tolist()
        ph = pd.read_csv(file)['ph [hPa]'].tolist()
        self._modelLevel = {'A': a,
                            'B': b,
                            'Pressure Height': ph}
        self._level = len(self._modelLevel['A']) - 1
        pass

    def __get_ph_levs(self, level):
        """
        Return the pressure at a given level (half level) and the one at next level (half level)
        :param level:
        :return:
        """
        a_coef = np.array(self._modelLevel['A'])
        b_coef = np.array(self._modelLevel['B'])
        sp = self._PSFC

        ph_lev = a_coef[level - 1] + (b_coef[level - 1] * sp)
        ph_levplusone = a_coef[level] + (b_coef[level] * sp)

        return ph_lev, ph_levplusone

    def getGloMean(self, fields: np.ndarray):
        """
        Calculate the global mean in terms of the pressure
        :param fields: a physical field in a mesh grid, like 180*360
        :return: mean pressure [Pa]
        """
        assert np.shape(fields)[0] == len(self._lat) and np.shape(fields)[1] == len(self._lon)

        lonMesh, latMesh = np.meshgrid(np.deg2rad(self._lon), np.deg2rad(self._lat))

        Grid = fields * np.cos(latMesh)
        total = np.sum(np.sum(Grid))
        area = np.sum(np.sum(np.cos(latMesh)))

        GloMean = total / area

        return GloMean

    def getSHmean(self, fields: np.ndarray):
        """
        Calculate the south hemisphere mean in terms of the pressure
        :param fields: a physical field in a mesh grid, like 180*360
        :return: mean pressure [Pa]
        """
        assert np.shape(fields)[0] == len(self._lat) and np.shape(fields)[1] == len(self._lon)

        N = len(self._lat)
        lat = self._lat[int(N / 2):]

        lonMesh, latMesh = np.meshgrid(np.deg2rad(self._lon), np.deg2rad(lat))

        Grid = fields[int(N / 2):, :] * np.cos(latMesh)
        total = np.sum(np.sum(Grid))
        area = np.sum(np.sum(np.cos(latMesh)))

        GloMean = total / area

        return GloMean

    def getNHmean(self, fields: np.ndarray):
        """
        Calculate the north hemisphere mean in terms of the pressure
        :param fields: a physical field in a mesh grid, like 180*360
        :return: mean pressure [Pa]
        """
        assert np.shape(fields)[0] == len(self._lat) and np.shape(fields)[1] == len(self._lon)

        N = len(self._lat)
        lat = self._lat[0:(int(N / 2) + 1)]

        lonMesh, latMesh = np.meshgrid(np.deg2rad(self._lon), np.deg2rad(lat))

        Grid = fields[0:(int(N / 2) + 1), :] * np.cos(latMesh)
        total = np.sum(np.sum(Grid))
        area = np.sum(np.sum(np.cos(latMesh)))

        GloMean = total / area

        return GloMean

    def getMeanLatRegion(self, fields: np.ndarray, LatMin, LatMax):
        assert np.shape(fields)[0] == len(self._lat) and np.shape(fields)[1] == len(self._lon)

        N = len(self._lat)
        index = (self._lat >= LatMin) * (self._lat <= LatMax)
        lat = self._lat[index]

        lonMesh, latMesh = np.meshgrid(np.deg2rad(self._lon), np.deg2rad(lat))

        Grid = fields[index, :] * np.cos(latMesh)
        total = np.sum(np.sum(Grid))
        area = np.sum(np.sum(np.cos(latMesh)))

        GloMean = total / area

        return GloMean

        pass

    def getTotMass(self, Pm):
        """
        compute the total mass following the formulation given by
        The mass of the atmosphere: a constraint on global analyses, Kevin E. Trenberth, 2004
        :param Pm: global mean pressure
        :return: total mass [kg]
        """

        radius = 6371000
        f = 1.0043
        mass = 4 * np.pi * radius ** 2 * 1.0043 / Constants.g_wmo * Pm

        return mass

    def getPw(self):
        """
        Calculate the Pressure caused by water vapor with model level data
        :return: a field of Pw like 180*360
        """

        pw = np.zeros(np.shape(self._SHUM[0]))
        for lev in range(1, self._level + 1):
            ph_lev, ph_levplusone = self.__get_ph_levs(lev)
            # pw += (ph_levplusone - ph_lev) * (self._SHUM[lev, :, :] + self._SHUM[lev - 1, :, :]) * 1 / 2
            pw += (ph_levplusone - ph_lev) * (self._SHUM[lev - 1, :, :])
            pass

        return pw

    def getPs(self):
        return self._PSFC


class WetDryMl_ERAinterim(WetDryMl_ERA5):
    model_level_file = '../data/Auxiliary/Model60.csv'

    def __init__(self):
        WetDryMl_ERA5.__init__(self)
        self._readModelLevel(WetDryMl_ERAinterim.model_level_file)
        pass

    def setDate(self, date='2017-02'):
        """
        get the wet and dry air at given date
        :param date: str, e.g., '2017-02'
        :return:
        """
        pathRt1 = '../data/ModelLevel/ERAinterim/'
        # pathRt2 = '../data/PressureLevel/ERA5/'

        SHUM = 'SHUM_ERAinterim_yearly_' + date[0:4] + '.nc'
        PSFC = 'PSFC_ERAinterim_monthly_' + date[0:4] + date[5:7] + '.nc'

        rd = ReadNC()
        rd.setPar(file=pathRt1 + SHUM, datatype=DataType.SHUM)
        self._SHUM, lat, lon = rd.read(seqN=int(date[5:7]) - 1)

        rd = ReadNC()
        rd.setPar(file=pathRt1 + PSFC, datatype=DataType.PSFC)
        self._PSFC, lat2, lon2 = rd.read(seqN=0)

        self._lat, self._lon = lat, lon


def demo1():
    wd = WetDryMl_ERA5()
    wd.setDate(date='2017-02')
    Pw = wd.getGloMean(wd.getPw())
    Ps = wd.getGloMean(wd.getPs())
    pass


def demo2():
    wd = WetDryMl_ERAinterim()
    wd.setDate(date='2017-02')
    Pw = wd.getGloMean(wd.getPw())
    Ps = wd.getGloMean(wd.getPs())
    pass


if __name__ == '__main__':
    demo1()
