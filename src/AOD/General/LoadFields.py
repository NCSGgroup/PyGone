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
import pygrib
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

from GeoMathKit import GeoMathKit
from LoadSH import AOD_GFZ, AODtype
from Setting import DataType, ForceFields
# import Ngl, Nio
import cfgrib

import netCDF4 as nc

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

        # '''path of nc'''
        # sstr1 = 'nc_' + sstr1


        path_dir = os.path.join(self.__path, str1[0], sstr1)

        flag_sp = sstr2 + '.ps.grib2'
        flag_t = 'CRA40_TEM_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.grib2'
        flag_q = 'CRA40_SHU_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.grib2'
        flag_z = 'HGT_SURFACE.grib2'

        # '''way of nc'''
        # flag_sp = sstr2 + '.ps.nc'
        # flag_t = 'CRA40_TEM_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.nc'
        # flag_q = 'CRA40_SHU_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.nc'
        # flag_z = 'HGT_SURFACE.nc'


        '''name of each data'''
        PHISFC = os.path.join(self.__path, flag_z)
        PSFC = os.path.join(path_dir, flag_sp)
        TEMP = os.path.join(path_dir, flag_t)
        SHUM = os.path.join(path_dir, flag_q)

        # f = Nio.open_file(PHISFC)
        # lat0, lon0 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        # self.__epoch_data[DataType.PHISFC] = f.variables['HGT_P0_L1_GLL0'][:].flatten()

        f1 = pygrib.open(PHISFC)
        f = f1.select(name='Orography')[0]
        lat0,lon0 = f.latlons()
        self.__epoch_data[DataType.PHISFC] = f['values'].flatten()

        # '''way of nc'''
        # f = nc.Dataset(PHISFC)
        # lat0,lon0 = f.variables['latitude'][:],f.variables['longitude'][:]
        # lat0 = np.array(lat0)
        # lon0 = np.array(lon0)
        # self.__epoch_data[DataType.PHISFC] = f.variables['orog'][:].flatten()

        # f = Nio.open_file(PSFC)
        # lat1, lon1 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        # ps = f.variables['PRES_P0_L1_GLL0'][:].flatten()
        # self.__epoch_data[DataType.PSFC] = ps

        # '''way of nc'''
        # f1 = nc.Dataset(PSFC)
        # lat1,lon1 = f1.variables['latitude'][:],f1.variables['longitude'][:]
        # lat1 = np.array(lat1)
        # lon1 = np.array(lon1)
        # ps = f1.variables['sp'][:].flatten()
        # self.__epoch_data[DataType.PSFC] = ps


        f2 = pygrib.open(PSFC)
        fp = f2.select(name='Surface pressure')[0]
        lat1,lon1 = fp.latlons()
        # lat1 = lat1[:,0]
        # lon1 = lon1[0,:]
        ps = fp['values'].flatten()
        self.__epoch_data[DataType.PSFC] = ps

        if OnlyPressure:
            assert (lat0 == lat1).all()
            assert (lon0 == lon1).all()
            return self

        # f = Nio.open_file(TEMP)
        # lat2, lon2 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        # TEMP_level = f.variables['lv_ISBL0'][:]
        # self.__epoch_data[DataType.TEMP] = f.variables['TMP_P0_L100_GLL0'][:].reshape(len(TEMP_level), -1)

        # '''way of nc'''
        # f2 = nc.Dataset(TEMP)
        # lat2,lon2 = f2.variables['latitude'][:],f2.variables['longitude'][:]
        # lat2 = np.array(lat2)
        # lon2 = np.array(lon2)
        #
        # TEMP_level = f2.variables['isobaricInhPa'][:]
        # TEMP_level = list(reversed(list(TEMP_level)))
        # TEMP_level = 100*np.array(TEMP_level)
        # self.__epoch_data[DataType.TEMP] = f2.variables['t'][:].reshape(len(TEMP_level),-1)



        ft = pygrib.open(TEMP)
        lat2,lon2 = ft.select(name='Temperature')[0].latlons()
        # lat2 = lat2[:,0]
        # lon2 = lon2[0,:]
        TEMP_level = []
        for i in range(0, ft.messages):
            f = ft.select(name='Temperature')[i]
            TEMP_level.append(float(f['level']) * 100)
            data = f['values'].flatten()
            if i ==0:
                data_TEMP = data
            else:
                data_TEMP = np.vstack((data_TEMP, data))
        TEMP_level = np.array(TEMP_level)
        self.__epoch_data[DataType.TEMP] = data_TEMP
        # print(data_TEMP)

        # f = Nio.open_file(SHUM)
        # lat3, lon3 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        # SHUM_level = f.variables['lv_ISBL0'][:]
        # self.__epoch_data[DataType.SHUM] = f.variables['SPFH_P0_L100_GLL0'][:].reshape(len(SHUM_level), -1)

        # '''way of nc'''
        # f3 = nc.Dataset(SHUM)
        # lat3,lon3 = f3.variables['latitude'][:],f3.variables['longitude'][:]
        # SHUM_level = f3.variables['isobaricInhPa'][:]
        # SHUM_level = list(reversed(list(SHUM_level)))
        # SHUM_level = 100*np.array(SHUM_level)
        # self.__epoch_data[DataType.SHUM] = f3.variables['q'][:].reshape(len(SHUM_level),-1)


        ft = pygrib.open(SHUM)
        lat3,lon3 = ft.select(name='Specific humidity')[0].latlons()
        # lat3 = lat3[:,0]
        # lon3 = lon3[0,:]
        SHUM_level = []
        for i in range(0,ft.messages):
            f = ft.select(name='Specific humidity')[i]
            SHUM_level.append(float(f['level'])*100)
            data = f['values'].flatten()
            if i == 0:
                data_SHUM = data
            else:
                data_SHUM = np.vstack((data_SHUM,data))
        SHUM_level = np.array(SHUM_level)
        self.__epoch_data[DataType.SHUM] = data_SHUM


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
        self.__lat, self.__lon = lat0[:,0], lon0[0,:]

        # self.__lat, self.__lon = lat0,lon0


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
        # '''path of nc'''
        # sstr1 = 'nc_' + sstr1


        path_dir = os.path.join(self.__path, str1[0], sstr1)



        flag_sp = sstr2 + '.ps.grib2'


        # flag_sp = sstr2 + '.ps.nc'

        '''name of each data'''
        PSFC = os.path.join(path_dir, flag_sp)
        # f = Nio.open_file(PSFC)
        # self.__lat, self.__lon = f.variables['lat_0'][:], f.variables['lon_0'][:]
        # ps = f.variables['PRES_P0_L1_GLL0'][:]
        # self.__epoch_data[DataType.PSFC] = ps

        # '''way of nc'''
        # f = nc.Dataset(PSFC)
        # self.__lat,self.__lon = f.variables['latitude'][:], f.variables['longitude'][:]
        # ps = f.variables['sp'][:]
        # self.__epoch_data[DataType.PSFC] = ps

        # '''way of cfgrib'''
        f = cfgrib.open_dataset(PSFC,engine='cfgrib')
        self.__lat,self.__lon = f['latitude'].values,f['longitude'].values
        ps = f['sp'].values
        self.__epoch_data[DataType.PSFC] = ps
        print(self.__epoch_data[DataType.PSFC])

        #
        # f1 = pygrib.open(PSFC)
        # f = f1.select(name='Surface pressure')[0]
        #
        # lat,lon = f.latlons()
        # self.__lat = lat[:,0]
        # # a=len(self.__lat)
        # self.__lon = lon[0,:]
        # # n = len(self.__lon)
        # ps = f['values']
        # self.__epoch_data[DataType.PSFC] = ps

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
    ld.setTime_for_tide()

if __name__ == '__main__':
    demo1()