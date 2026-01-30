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

from src.AOD.CRA_LICOM.Configure.GeoMathKit import GeoMathKit
from src.AOD.CRA_LICOM.Configure.LoadSH import AOD_GFZ, AODtype
from src.AOD.CRA_LICOM.Configure.Setting import DataType, ForceFields
# import Ngl, Nio
import cfgrib
import xarray as xr
import datetime

# import netCDF4 as nc

class LoadFields:
    """
    This class deals with loading all necessary inputs for one cycle of AOD computation.
    """
    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]

    def __init__(self, data_path='../../paper_data/'):
        self.__epoch_data = {}
        self.__meanfield = None
        self.__modelLevel = None
        self.__path = data_path
        self.__lat, self.__lon = None, None
        self.__nLat, self.__nLon = None, None
        self.__level = None
        self.__q_level = None

        pass

    def setTime(self, date='2007-12-01', time='00:00:00', OnlyPressure=False):
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

        flag_sp = 'CRA40_SINGLE_'+sstr2+'_GLB_0P50_HOUR_V1_0_0.grib2'

        # flag_sp = sstr2 + '.ps.grib2'
        flag_t = 'CRA40_TEM_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.grib2'
        flag_q = 'CRA40_SHU_' + sstr2 + '_GLB_0P50_HOUR_V1_0_0.grib2'
        flag_z = 'HGT_SURFACE.grib2'



        '''name of each data'''
        PHISFC = os.path.join(self.__path, flag_z)
        PSFC = os.path.join(path_dir, flag_sp)
        TEMP = os.path.join(path_dir, flag_t)
        SHUM = os.path.join(path_dir, flag_q)

        if not os.path.exists(PSFC):
            flag_sp = f'{sstr2}.ps.grib2'
            PSFC = os.path.join(path_dir,flag_sp)

        # f = Nio.open_file(PHISFC)
        # lat0, lon0 = f.variables['lat_0'][:], f.variables['lon_0'][:]
        # self.__epoch_data[DataType.PHISFC] = f.variables['HGT_P0_L1_GLL0'][:].flatten()


        f1 = cfgrib.open_dataset(PHISFC, engine='cfgrib')
        lat0, lon0 = f1['latitude'].values, f1['longitude'].values
        phisfc = f1['orog'].values.flatten()
        self.__epoch_data[DataType.PHISFC] = phisfc



        f = cfgrib.open_dataset(PSFC, engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'})
        lat1, lon1 = f['latitude'].values, f['longitude'].values
        ps = f['sp'].values.flatten()
        self.__epoch_data[DataType.PSFC] = ps



        if OnlyPressure:
            assert (lat0 == lat1).all()
            assert (lon0 == lon1).all()
            return self


        ft = cfgrib.open_dataset(TEMP)
        lat2, lon2 = ft['latitude'].values, ft['longitude'].values
        TEMP_level = ft['isobaricInhPa'].values[::-1] * 100
        date_TEMP = []
        for i in range(len(ft['t'][:, 0, 0])):
            date_TEMP.append(ft['t'].values[i].flatten())
        data_TEMP = np.array(date_TEMP[::-1])
        self.__epoch_data[DataType.TEMP] = data_TEMP


        ft = cfgrib.open_dataset(SHUM)
        lat3, lon3 = ft['latitude'].values, ft['longitude'].values
        SHUM_level = ft['isobaricInhPa'].values[::-1] * 100
        date_SHUM = []
        for i in range(len(ft['q'][:, 0, 0])):
            date_SHUM.append(ft['q'].values[i].flatten())
        data_SHUM = np.array(date_SHUM[::-1])
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
        # self.__lat, self.__lon = lat0[:,0], lon0[0,:]

        self.__lat, self.__lon = lat0,lon0


        return self

    def setTime_for_tide(self, date='2007-01-01', time='00:00:00'):
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

        flag_sp = sstr2+'.ps.grib2'



        # flag_sp = sstr2 + '.ps.grib2'




        '''name of each data'''
        PSFC = os.path.join(path_dir, flag_sp)

        f = cfgrib.open_dataset(PSFC,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'})
        self.__lat,self.__lon = f['latitude'].values,f['longitude'].values
        ps = f['sp'].values
        self.__epoch_data[DataType.PSFC] = ps

        return self

    def setTime_for_tide_Bai(self, date='2007-01-01', time='00:00:00'):
        """
        set the time epoch of data that will be used in later computation.
        :param OnlyPressure: if True, then only pressure is read.
        :param date:
        :param time:
        :return:
        """
        pbo_data = xr.open_dataset("I:/CRALICOM/result/Tide/Tide_test/pbo_cra3_2007.nc")
        pso_data = xr.open_dataset("I:\CRALICOM/result/Tide/Tide_test/pso_cra3_2007.nc")



        input_datetime = datetime.datetime.strptime(f"{date} {time}","%Y-%m-%d %H:%M:%S")
        start_time = datetime.datetime(2007,1,1)

        delta = input_datetime - start_time
        total_seconds = delta.total_seconds()
        index = int(total_seconds // (3*3600))

        pbo = np.nan_to_num(pbo_data['pbo'].values[index],nan=0)
        pso = np.nan_to_num(pso_data['slp'].values[index],nan=0)
        # print(f"pbo and its shape: {pbo,pbo.shape}")

        # print(f"{date}-{time} index is: {index}")
        f = pbo
        # f = cfgrib.open_dataset(PSFC, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'})
        self.__lat, self.__lon = pbo_data['lat'].values, pbo_data['lon'].values
        # ps = f['sp'].values
        self.__epoch_data[DataType.PSFC] = f
        # print(f.shape)

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
def demo2():
    path = 'H:/Paper_CRALICOM/data/CRA.grib2/2020/20201231/CRA40_SINGLE_2020123100_GLB_0P50_HOUR_V1_0_0.grib2'
    data = cfgrib.open_dataset(path, filter_by_keys={'typeOfLevel': 'surface'})
    print(data['sp'])
    print(data['latitude'])
    print('\n')
    path1 = 'H:/CRA/data/CRA.grib2/2020/20201229/2020122900.ps.grib2'
    data2 = cfgrib.open_dataset(path1)
    print(data2['sp'])

def demo3():
    ld = LoadFields()
    ld.setTime_for_tide_Bai(date='2007-12-31',time='21:00:00')
    print(ld.getNlatNlon())

if __name__ == '__main__':
    demo3()