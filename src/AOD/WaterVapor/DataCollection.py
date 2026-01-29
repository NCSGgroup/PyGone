"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/11/12 下午12:23
@Description:
"""
import sys

sys.path.append('../')

import cdsapi
from GeoMathKit import GeoMathKit
from Setting import DataType
import json
import calendar
from ecmwfapi import ECMWFDataServer


class ERA5_monthly:
    server = cdsapi.Client()

    def __init__(self):

        '''download path'''
        self.__path = "../data/ModelLevel/ERA5/"
        self.__ERA_parameters = {}
        self.__monthlist = None

        pass

    def configure(self, config: dict):

        for term in DataType:
            self.__ERA_parameters[term] = config[term.name]

        self.__path = config['download path']

        self.__monthlist = GeoMathKit.monthListByMonth(config['begin_date'], config['end_date'])

        return self

    def setDate(self, begin='2002-01', end='2002-01'):
        """
        Manually set the date of data to be downloaded.
        Notice: this func has to be used after 'configure', otherwise the date will be automatically selected
        from the configuration file.
        :param begin:
        :param end:
        :return:
        """

        self.__monthlist = GeoMathKit.monthListByMonth(begin, end)
        return self

    def download(self, kind: DataType):

        if kind == DataType.PSFC:
            self.__getPSFC()
        elif kind == DataType.SHUM or kind == DataType.TEMP:
            self.__getTempAndShum(kind=kind)
        elif kind == DataType.PHISFC:
            self.__getPHISFC()

        pass

    def __getPSFC(self):

        for mon in self.__monthlist:
            par = self.__ERA_parameters[DataType.PSFC]
            par['details']['year'] = "%04d" % mon.year
            par['details']['month'] = "%02d" % mon.month

            par['target'] = self.__path + DataType.PSFC.name + "_" + \
                            "ERA5_monthly_%04d%02d.nc" % (mon.year, mon.month)

            self.server.retrieve(par['class'], par['details'], par['target'])

        pass

    def __getTempAndShum(self, kind: DataType):

        year = self.__monthlist[0].year
        par = self.__ERA_parameters[kind]
        month = ['01', '02', '03',
                 '04', '05', '06',
                 '07', '08', '09',
                 '10', '11', '12']

        requestDates = ''

        for mon in self.__monthlist:

            if mon.year != year:

                if 'date' in par['details'].keys():
                    par['details']['date'] = requestDates[0:-1]
                    par['details']['decade'] = str(year)[:3] + '0'
                elif 'year' in par['details'].keys():
                    par['details']['year'] = '%04d' % year
                    par['details']['month'] = month

                par['target'] = self.__path + kind.name + "_" + "ERA5_yearly_%04d.nc" % (
                    year)
                print('Get data for : %s' % requestDates)
                self.server.retrieve(par['class'], par['details'], par['target'])

                year = mon.year
                requestDates = str(mon.year) + (str(mon.month)).zfill(2) + '01/'
            else:
                requestDates = requestDates + str(mon.year) + (str(mon.month)).zfill(2) + '01/'

        pass

    def __getPHISFC(self):
        par = self.__ERA_parameters[DataType.PHISFC]

        par['target'] = self.__path + DataType.PHISFC.name + "_ERA5_invariant.nc"

        self.server.retrieve(par['class'], par['details'], par['target'])
        pass

    @staticmethod
    def defaultConfig(isWrite=True):
        config = {'download path': "../data/ModelLevel/ERA5/",
                  'begin_date': '2002-01',
                  'end_date': '2002-01'}

        # config = {'download path': "../data/PressureLevel/ERA5/",
        #           'begin_date': '2002-01',
        #           'end_date': '2002-01'}

        grid = "0.5/0.5"

        PHI = {
            "class": 'reanalysis-era5-single-levels',
            "details": {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': 'orography',
                'year': '1989',
                'month': '01',
                'day': '01',
                'time': '00:00',
                'grid': grid,
            },
            "target": 'orography.nc'

        }

        PSFC = {
            'class': 'reanalysis-era5-single-levels-monthly-means',
            'details': {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': 'surface_pressure',
                'year': '1980',
                'month': '01',
                'time': '00:00',
                'grid': grid,
            },
            'target': 'download.nc'
        }

        PSFC_preliminary = {
            'class': 'reanalysis-era5-single-levels-monthly-means-preliminary-back-extension',
            'details': {
                'format': 'netcdf',
                'product_type': 'reanalysis-monthly-means-of-daily-means',
                'variable': 'surface_pressure',
                'year': '1980',
                'month': '01',
                'time': '00:00',
                'grid': grid,
            },
            'target': 'download.nc'
        }

        SHUM_PL = {
            'class': 'reanalysis-era5-pressure-levels-monthly-means',
            'details': {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': 'specific_humidity',
                'pressure_level': [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000',
                ],
                'year': '1979',
                'month': '01',
                'time': '00:00',
                'grid': grid,
            },
            'target': 'download.nc'
        }

        SHUM_PL_preliminary = {
            'class': 'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension',
            'details': {
                'format': 'netcdf',
                'product_type': 'reanalysis-monthly-means-of-daily-means',
                'variable': 'specific_humidity',
                'pressure_level': [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000',
                ],
                'year': '1950',
                'month': '01',
                'time': '00:00',
                'grid': grid,
            },
            'target': 'download.nc'
        }

        SHUM_ML = {
            'class': 'reanalysis-era5-complete',
            'details': {
                'date': '19500101',
                'decade': '1950',
                'levelist': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32'
                            '/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61'
                            '/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90'
                            '/91/92/93/94/95/96/97/98/99/100/101/102/103/104/105/106/107/108/109/110/111/112/113/114'
                            '/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136'
                            '/137',
                'levtype': 'ml',
                'param': '133',
                'stream': 'moda',
                'type': 'an',
                'grid': grid,
                'format': 'netcdf',
            },
            'target': 'q_ml.nc'
        }

        TEMP_PL = {
            'class': 'reanalysis-era5-pressure-levels-monthly-means',
            'details': {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': 'temperature',
                'pressure_level': [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000',
                ],
                'year': '1979',
                'month': '01',
                'time': '00:00',
                'grid': grid,
            },
            'target': 'download.nc'
        }

        TEMP_PL_preliminary = {
            'class': 'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension',
            'details': {
                'format': 'netcdf',
                'product_type': 'reanalysis-monthly-means-of-daily-means',
                'variable': 'temperature',
                'pressure_level': [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000',
                ],
                'year': '1950',
                'month': '01',
                'time': '00:00',
                'grid': grid,
            },
            'target': 'download.nc'
        }

        TEMP_ML = {
            'class': 'reanalysis-era5-complete',
            'details': {
                'date': '19500101',
                'decade': '1950',
                'levelist': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32'
                            '/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61'
                            '/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90'
                            '/91/92/93/94/95/96/97/98/99/100/101/102/103/104/105/106/107/108/109/110/111/112/113/114'
                            '/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136'
                            '/137',
                'levtype': 'ml',
                'param': '130',
                'stream': 'moda',
                'type': 'an',
                'grid': grid,
                'format': 'netcdf',
            },
            'target': 't_ml.nc'
        }

        config[DataType.PSFC.name] = PSFC
        config[DataType.TEMP.name] = TEMP_ML
        config[DataType.SHUM.name] = SHUM_ML
        # config[DataType.TEMP.name] = TEMP_PL
        # config[DataType.SHUM.name] = SHUM_PL
        config[DataType.PHISFC.name] = PHI

        if isWrite:
            with open('../settings/ERA5_monthly_ml.download.settings.json', 'w') as f:
                f.write(json.dumps(config))

        return config


class ERAinterim_monthly:
    """link to the server"""
    server = ECMWFDataServer()

    def __init__(self):

        '''download path'''
        self.__path = "../data/ModelLevel/ERAinterim/"
        self.__ERA_parameters = {}
        self.__monthlist = None

        pass

    def configure(self, config: dict):

        for term in DataType:
            self.__ERA_parameters[term] = config[term.name]

        self.__path = config['download path']

        self.__monthlist = GeoMathKit.monthListByMonth(config['begin_date'], config['end_date'])

        return self

    def setDate(self, begin='2002-01', end='2002-01'):
        """
        Manually set the date of data to be downloaded.
        Notice: this func has to be used after 'configure', otherwise the date will be automatically selected
        from the configuration file.
        :param begin:
        :param end:
        :return:
        """

        self.__monthlist = GeoMathKit.monthListByMonth(begin, end)
        return self

    @staticmethod
    def defaultConfig(isWrite=True):
        config = {'download path': "../data/PressureLevel/ERAinterim/",
                  'begin_date': '2002-01',
                  'end_date': '2002-01'}

        grid = "0.5/0.5"

        PSFC = {
            "class": "ei",
            "dataset": "interim",
            "date": "20190101",
            "expver": "1",
            "grid": grid,
            "levtype": "sfc",
            "param": "134.128",
            "stream": "moda",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }

        TEMP_ML = {
            "class": "ei",
            "dataset": "interim",
            "date": "20190101",
            "expver": "1",
            "grid": grid,
            "levelist": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34"
                        "/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60",
            "levtype": "ml",
            "param": "130.128",
            "stream": "moda",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }

        SHUM_ML = {
            "class": "ei",
            "dataset": "interim",
            "date": "20190101",
            "expver": "1",
            "grid": grid,
            "levelist": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34"
                        "/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60",
            "levtype": "ml",
            "param": "133.128",
            "stream": "moda",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }

        TEMP_PL = {
            "class": "ei",
            "dataset": "interim",
            "date": "20190101",
            "expver": "1",
            "grid": grid,
            "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750"
                        "/775/800/825/850/875/900/925/950/975/1000",
            "levtype": "pl",
            "param": "130.128",
            "stream": "moda",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }

        SHUM_PL = {
            "class": "ei",
            "dataset": "interim",
            "date": "20190101",
            "expver": "1",
            "grid": grid,
            "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750"
                        "/775/800/825/850/875/900/925/950/975/1000",
            "levtype": "pl",
            "param": "133.128",
            "stream": "moda",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }

        PHI = {
            "class": "ei",
            "dataset": "interim",
            "date": "1989-01-01",
            "expver": "1",
            "grid": "0.5/0.5",
            "levtype": "sfc",
            "param": "129.128",
            "step": "0",
            "stream": "oper",
            "time": "12:00:00",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }

        config[DataType.PSFC.name] = PSFC
        # config[DataType.TEMP.name] = TEMP_ML
        # config[DataType.SHUM.name] = SHUM_ML
        config[DataType.TEMP.name] = TEMP_PL
        config[DataType.SHUM.name] = SHUM_PL
        config[DataType.PHISFC.name] = PHI

        if isWrite:
            with open('../settings/ERA-interim_monthly_pl.download.settings.json', 'w') as f:
                f.write(json.dumps(config))

        return config

    def download(self, kind: DataType):

        if kind == DataType.PSFC:
            self.__getPSFC()
        elif kind == DataType.SHUM or kind == DataType.TEMP:
            self.__getTempAndShum(kind=kind)
        elif kind == DataType.PHISFC:
            self.__getPHISFC()

        pass

    def __getPSFC(self):

        for mon in self.__monthlist:
            par = self.__ERA_parameters[DataType.PSFC]
            par['date'] = "%04d%02d%02d" % (mon.year, mon.month, mon.day)
            par['target'] = self.__path + DataType.PSFC.name + "_" + \
                            "ERAinterim_monthly_%04d%02d.nc" % (mon.year, mon.month)

            self.server.retrieve(par)

        pass

    def __getTempAndShum(self, kind: DataType):

        year = self.__monthlist[0].year
        par = self.__ERA_parameters[kind]
        month = ['01', '02', '03',
                 '04', '05', '06',
                 '07', '08', '09',
                 '10', '11', '12']

        requestDates = ''

        for mon in self.__monthlist:

            if mon.year != year:

                par['date'] = requestDates[0:-1]
                par['target'] = self.__path + kind.name + "_" + "ERAinterim_yearly_%04d.nc" % (
                    year)
                print('Get data for : %s' % requestDates)
                self.server.retrieve(par)

                year = mon.year
                requestDates = str(mon.year) + (str(mon.month)).zfill(2) + '01/'
            else:
                requestDates = requestDates + str(mon.year) + (str(mon.month)).zfill(2) + '01/'

        pass

    def __getPHISFC(self):
        par = self.__ERA_parameters[DataType.PHISFC]

        par['target'] = self.__path + DataType.PHISFC.name + "_ERAinterim_invariant.nc"

        self.server.retrieve(par)
        pass


def demo1():
    """
    download ERA-interim
    :return:
    """
    dd = ERAinterim_monthly()
    # dd.configure(dd.defaultConfig(isWrite=True))
    with open('../settings/ERA-interim_monthly_ml.download.settings.json', 'r') as fp:
        config = json.load(fp)
    dd.configure(config)
    dd.setDate(begin='1979-01', end='2020-01')
    # dd.download(kind=DataType.PHISFC)
    dd.download(kind=DataType.PSFC)
    # dd.download(kind=DataType.SHUM)
    # dd.download(kind=DataType.TEMP)
    pass


def demo2():
    """
    download ERA5
    :return:
    """
    dd = ERA5_monthly()
    dd.configure(dd.defaultConfig(isWrite=True))
    # with open('../settings/ERA5_monthly_ml.download.settings.json', 'r') as fp:
    #     config = json.load(fp)
    # dd.configure(config)
    dd.setDate(begin='2000-01', end='2001-01')
    # dd.download(kind=DataType.PHISFC)
    # dd.download(kind=DataType.PSFC)
    dd.download(kind=DataType.SHUM)
    # dd.download(kind=DataType.TEMP)


if __name__ == '__main__':
    demo2()
