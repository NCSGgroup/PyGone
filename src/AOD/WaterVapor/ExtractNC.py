"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/3 10:45
@Description:
"""

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

from Setting import DataType


class NCtoAscii:
    """
    The format is referred from ECMWF, which might be unsuitable for other data set.
    """

    def __init__(self):
        self.__nameOfData = {DataType.TEMP: "t",
                             DataType.SHUM: "q",
                             DataType.PSFC: "sp",
                             DataType.PHISFC: "z"}

        self.__fileIn = None
        self.__fileOut = None
        self.__DataType = None
        self.__nc = None

        pass

    def setPar(self, fileIn: str, fileOut: str, datatype: DataType):
        """
        parameter setting
        :param fileIn: the complete path (directory and filename) of the NC data
        :param fileOut: the complete path of output
        :param datatype: PSFC, SHUM, PHI, TEMP
        :return:
        """
        self.__fileIn = fileIn
        self.__fileOut = fileOut
        self.__DataType = datatype
        self.__nc = Dataset(self.__fileIn)

        return self

    def info(self):
        """
        Show the abstract of the given NC data.
        :return:
        """

        print(self.__nc)

        print(self.__nc.variables.keys())
        for i in self.__nc.variables.keys():
            print('___________________________________________')
            print(i)
            print(self.__nc.variables[i])

        return self

    def convert(self):
        """
        This only works for converting NC to Ascii
        :return:
        """
        print('Start converting %s' % self.__fileIn)

        nc = self.__nc

        if self.__DataType == DataType.PSFC or self.__DataType == DataType.PHISFC:
            '''surface pressure and geo-potential'''

            scale_factor = nc.variables[self.__nameOfData[self.__DataType]].scale_factor
            offset = nc.variables[self.__nameOfData[self.__DataType]].add_offset

            value = nc.variables[self.__nameOfData[self.__DataType]][0, :, :]
            longitude = nc.variables['longitude'][:]
            latitude = nc.variables['latitude'][:]
            # value_add_offset=value*scale_factor+offset
            # print(longitude.size, latitude.size)

            with open(self.__fileOut, 'w') as f:

                for i in range(latitude.size):
                    for j in range(longitude.size):
                        # f.writelines("%7.3f%10.3f%20.10e\n"%(latitude[i],longitude[j],value_add_offset[i,j]))
                        f.writelines("%7.3f%10.3f%20.10e\n" % (latitude[i], longitude[j], value[i, j]))

        else:
            '''temperature and humidity'''
            scale_factor = nc.variables[self.__nameOfData[self.__DataType]].scale_factor
            offset = nc.variables[self.__nameOfData[self.__DataType]].add_offset
            level = nc.variables['level']
            value = nc.variables[self.__nameOfData[self.__DataType]][0, :, :, :]
            longitude = nc.variables['longitude'][:]
            latitude = nc.variables['latitude'][:]
            # value_add_offset = value * scale_factor + offset

            with open(self.__fileOut, 'w') as f:

                for k in tqdm(range(level.size), desc=self.__DataType.name):
                    # print("Recording level-%s" % k)
                    for i in range(latitude.size):
                        for j in range(longitude.size):
                            # f.writelines("%7.3f%10.3f%20.10e\n"%(latitude[i],longitude[j],value_add_offset[k,i,j]))
                            f.writelines("%7.3f%10.3f%20.10e\n" % (latitude[i], longitude[j], value[k, i, j]))

        print("%s has been successfully converted!\n" % self.__DataType.name)

        pass


class ReadNC:
    """
    Only work for reading ECMWF temperature/humidity/surface pressure/geo-potential

    """
    def __init__(self):
        self.__nameOfData = {DataType.TEMP: "t",
                             DataType.SHUM: "q",
                             DataType.PSFC: "sp",
                             DataType.PHISFC: "z"}

        self.__DataType = None
        self.__nc = None
        pass

    def setPar(self, file, datatype: DataType):
        self.__DataType = datatype
        self.__nc = Dataset(file)
        return self

    def read(self, seqN: int = 0):
        """
        read data from NC
        :param seqN: the sequence number of given date-time, only for the surface data
        :return:
        """
        assert isinstance(seqN, int)

        try:
            '''multi-level data'''
            level = self.__nc.variables['level'].size
            res = self.__nc.variables[self.__nameOfData[self.__DataType]][seqN, :, :, :]
        except Exception as e:
            '''surface data'''
            res = self.__nc.variables[self.__nameOfData[self.__DataType]][seqN, :, :]
        finally:
            longitude = self.__nc.variables['longitude'][:]
            latitude = self.__nc.variables['latitude'][:]

        return np.array(res), np.array(latitude), np.array(longitude)


def demo1():
    fileIn = '../data/ERA_interim/TEMP_SHUM_interim_daily_20010101_00.nc'
    fileOut = '../result/SHUM_interim_20010101_00.ascii'
    nc = NCtoAscii().setPar(fileIn, fileOut, DataType.SHUM)
    nc.info()
    nc.convert()


def demo2():
    # file = '../data/PressureLevel/ERA5/PSFC_ERA5_monthly_197902.nc'
    # file = '../data/PressureLevel/ERA5/SHUM_ERA5_monthly_200201.nc'
    # file = '../data/ModelLevel/ERA5/SHUM_ERA5_yearly_2017.nc'
    file = '../data/ModelLevel/ERA5/PHISFC_ERA5_invariant.nc'
    rd = ReadNC().setPar(file, DataType.PHISFC)
    res, lat, lon = rd.read()
    pass


if __name__ == '__main__':
    demo2()
