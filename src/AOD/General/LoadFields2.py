"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/3/31
@Description:
"""

"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/3 17:44
@Description:
"""

import json
import os

# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

from GeoMathKit import GeoMathKit
from LoadSH import AOD_GFZ, AODtype
from Setting import DataType, ForceFields
import Ngl, Nio


class LoadFields:
    """
    This class deals with loading all necessary inputs for one cycle of AOD computation.
    """
    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]

    def __init__(self, data_path='../data/CRA.grib2/'):
        self.__epoch_data = {}
        self.__meanfield = None
        self.__modelLevel = None
        self.__path = data_path
        self.__lat, self.__lon = None, None
        self.__nLat, self.__nLon = None, None
        self.__level = None
        self.__q_level = None

        pass

    def setTime(self, date='2002-01-01', time='06:00:00', OnlyPressure=False):
        """
        set the time epoch of data that will be used in later computation.
        :param OnlyPressure: if True, then only pressure is read.
        :param date:
        :param time:
        :return:
        """

        assert time in self.TimeEpoch
        str1 = date.split('-')
        str2 = time.split(':')

        sstr1 = str1[0] + str1[1] + str1[2]
        sstr2 = sstr1 + str2[0]
        path_dir = os.path.join(self.__path, str1[0], sstr1)

        flag_sp = sstr2 + '.ps.grib2'
        flag_t = 'CRA40_TEM_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.grib2'
        flag_q = 'CRA40_SHU_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.grib2'
        flag_z = 'HGT_SURFACE.grib2'

        '''name of each data'''
        PHISFC = os.path.join(self.__path, flag_z)
        PSFC = os.path.join(path_dir, flag_sp)
        TEMP = os.path.join(path_dir, flag_t)
        SHUM = os.path.join(path_dir, flag_q)

        f = Nio.open_file(PHISFC)
        lat0, lon0 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        self.__epoch_data[DataType.PHISFC] = f.variables['HGT_P0_L1_GLL0'][:].flatten()

        f = Nio.open_file(PSFC)
        lat1, lon1 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        ps = f.variables['PRES_P0_L1_GLL0'][:].flatten()
        self.__epoch_data[DataType.PSFC] = ps

        if OnlyPressure:
            assert (lat0 == lat1).all()
            assert (lon0 == lon1).all()
            return self

        f = Nio.open_file(TEMP)
        lat2, lon2 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        TEMP_level = f.variables['lv_ISBL0'][:]
        self.__epoch_data[DataType.TEMP] = f.variables['TMP_P0_L100_GLL0'][:].reshape(len(TEMP_level), -1)

        f = Nio.open_file(SHUM)
        lat3, lon3 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        SHUM_level = f.variables['lv_ISBL0'][:]
        self.__epoch_data[DataType.SHUM] = f.variables['SPFH_P0_L100_GLL0'][:].reshape(len(SHUM_level), -1)


        assert (lat0 == lat1).all() and (lat0 == lat2).all() and (lat0 == lat3).all()
        assert (lon0 == lon1).all() and (lon0 == lon2).all() and (lon0 == lon3).all()

        iso_pres = []

        for i in range(len(TEMP_level)):
            pres_level = np.ones(len(ps)) * TEMP_level[i]
            if (TEMP_level[i] - ps < 0).all():
                iso_pres.append(pres_level)
                continue
            index = TEMP_level[i] - ps > 0

            pres_level[index] = ps[index]

            iso_pres.append(pres_level)
            # print(i)
            pass

        self.__level = len(TEMP_level)
        self.__q_level = len(SHUM_level)
        self.__epoch_data['PsLevel'] = np.array(iso_pres)
        self.__lat, self.__lon = lat0, lon0

        return self

    def setTime_for_tide(self, date='2002-01-01', time='06:00:00'):
        """
        set the time epoch of data that will be used in later computation.
        :param OnlyPressure: if True, then only pressure is read.
        :param date:
        :param time:
        :return:
        """

        assert time in self.TimeEpoch
        str1 = date.split('-')
        str2 = time.split(':')

        sstr1 = str1[0] + str1[1] + str1[2]
        sstr2 = sstr1 + str2[0]
        path_dir = os.path.join(self.__path, str1[0], sstr1)

        flag_sp = sstr2 + '.ps.grib2'

        '''name of each data'''
        PSFC = os.path.join(path_dir, flag_sp)
        f = Nio.open_file(PSFC)
        self.__lat, self.__lon = f.variables['lat_0'][:], f.variables['lon_0'][:]
        ps = f.variables['PRES_P0_L1_GLL0'][:]
        self.__epoch_data[DataType.PSFC] = ps

        return self

    def getField(self, kind: DataType = DataType.TEMP):
        return self.__epoch_data[kind]

    def getPressureLevel(self):
        return self.__epoch_data['PsLevel']

    def getLevel(self):
        return self.__level

    def getQLevel(self):
        return self.__q_level

    def getLatLon(self):
        """
        position of N*M points
        :return: geodetic latitude [dimension: N*M] and longitude [dimension: N*M] in degree
        """

        lon, lat = np.meshgrid(self.__lon, self.__lat)

        return lat.flatten(), lon.flatten()

    def getLatLon_v2(self):
        """
        position of N*M points
        :return: geodetic latitude [dimension: N] and longitude [dimension: M] in degree
        """
        return self.__lat, self.__lon

    def getNlatNlon(self):
        """
        position of N*M points
        :return: N, M
        """
        return len(self.__lat), len(self.__lon)


def demo1():
    ld = LoadFields()
    ld.setTime()
    pass


if __name__ == '__main__':
    demo1()
