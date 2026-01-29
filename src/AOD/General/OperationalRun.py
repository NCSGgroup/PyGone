"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2023/2/8
@Description:
"""
import copy
import sys

import numpy as np

sys.path.append('../')
import numpy as np
import os

from GeoMathKit import GeoMathKit
from LoadSH import AOD_GFZ
from HConfig import HConfig
from Format import CnmSnm, FormatWrite
from Harmonic import Harmonic, LoveNumber, LoveNumberType, GeoMathKit
# from pysrc.IntegralMethods import innerIn, InnerIntegral, GeopotentialHeight, InterpOption
from LoadFields import LoadFields
from Setting import AODtype, Constants, EllipsoidType, DataType, HarAnalysisType
from GeopotentialHeight import GeopotentialHeight
from InnerIntegral import InnerIntegral
import time as ti
import json
from RefEllipsoid import RefEllipsoid
from GeoidUndulation import GeoidUndulation
from SurPres2CS import SurPres2CS
from IntegralMethods import innerIn
from IBcorrection import IBcorrection
from TideFit import Detide
import calendar

import warnings


def run_SP_v1(BeginDate, EndDate):
    """
    Run by batch days
    surface pressure integration
    :param BeginDate: '2009-01-01'
    :param EndDate: '2009-01-02'
    :return:
    """
    # from pysrc.ExtractNC import ReadNC
    # from pysrc.GeoidUndulation import GeoidUndulation
    # from pysrc.RefEllipsoid import RefEllipsoid
    # from pysrc.Setting import EllipsoidType, DataType
    # from pysrc.SurPres2CS import SurPres2CS
    # from pysrc.IBcorrection import IBcorrection
    import numpy as np
    # from pysrc.TideFit import Detide

    log_file = '../log/log_sp.txt'
    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]
    # TimeEpoch = ["06:00:00"]
    elltype = EllipsoidType.GRS80_IERS2010
    ell = RefEllipsoid(elltype)
    LN = LoveNumber('../data/Auxiliary/')

    ld = LoadFields(data_path='../data/CRA.grib2/')
    ld.setTime()
    orography = ld.getField(DataType.PHISFC) * Constants.g_wmo
    lat, lon = ld.getLatLon_v2()

    undulation = GeoidUndulation(elltype).getGeoid(lat, lon).flatten()
    #
    # PHISFC = '../data/ERA5/PHISFC_ERA5_invariant.nc'
    # orography = ReadNC().setPar(PHISFC, DataType.PHISFC).read()[0]
    #
    # '''reduce the spatial resolution: 0.25 X 0.25 ==> 0.5 X 0.5'''
    # orography = orography[0:721:2, 0:1440:2].flatten()
    #
    #
    # '''IBcorrection'''
    # ib = IBcorrection(lat, lon)
    #
    # '''Configure for loading atmosphere driving data'''
    # with open('../settings/loading-ERA5.setting.json', 'r') as fp:
    #     config = json.load(fp)
    # lf = LoadFields().configure(config=config)
    #
    '''Configure for the surface pressure integration'''
    sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
                                loveNumber=LN)

    '''Configure for output'''
    # config_integral = InnerIntegral.defaultConfig(isWrite=True)
    # Nmax = config_integral['MaxDegree']
    Nmax = 100

    # fm = FormatWrite().setRootDir('../result/ERA5sp')
    fm = FormatWrite().setRootDir('../result/sp')

    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        print('---------Date: %s-------' % date)
        cs = CnmSnm(date=date, Nmax=Nmax)

        for time in TimeEpoch:
            print('Computing: %s' % time)
            begin = ti.time()
            # ld.setTime(date, time, OnlyPressure=True)
            try:
                ld.setTime(date, time, OnlyPressure=True)
            except Exception as err:
                with open(log_file, 'a') as f:
                    f.write(date + ' ' + time + '  ' + err.args[0] + '\n')
                    f.close()
                    continue

            sp = ld.getField(DataType.PSFC)
            '''de-tide'''
            # dt.remove(pressure=sp, date=date, time=time)
            '''IB correction'''
            # sp_af = ib.correct(sp.copy())
            # sp_af = sp
            '''SP integration and harmonic analysis'''
            cnm, snm = sf2cs.setPressure(pressure=sp, maxDeg=Nmax)
            # cnm, snm = sf2cs.setPressure(pressure=sp_af.flatten(), maxDeg=Nmax)
            '''record the stokes coefficients'''
            cs.add(Cnm=GeoMathKit.CS_2dTo1d(cnm), Snm=GeoMathKit.CS_2dTo1d(snm),
                   epoch=time, date=date, attribute=AODtype.ATM.name)
            '''counting time'''
            print("Cost time: %s ms" % ((ti.time() - begin) * 1000))

        '''write results'''
        fm.setCS(cs).AODstyle()

    pass


def run_VI_v1(BeginDate, EndDate):
    """
    Run by batch days
    surface pressure integration
    :param BeginDate: '2009-01-01'
    :param EndDate: '2009-01-02'
    :return:
    """
    # from pysrc.ExtractNC import ReadNC
    # from pysrc.GeoidUndulation import GeoidUndulation
    #
    # from pysrc.Setting import EllipsoidType, DataType
    # from pysrc.SurPres2CS import SurPres2CS
    # from pysrc.IBcorrection import IBcorrection
    import numpy as np
    # from pysrc.TideFit import Detide

    warnings.filterwarnings('ignore')

    log_file = '../log/log_vi.txt'

    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]
    # TimeEpoch = ["06:00:00"]
    config_integral = InnerIntegral.defaultConfig(isWrite=True)
    Nmax = config_integral['MaxDegree']

    elltype = EllipsoidType[config_integral['Ellipsoid']]
    # ell = RefEllipsoid(elltype)
    LN = LoveNumber('../data/Auxiliary/')

    ld = LoadFields(data_path='../data/CRA.grib2/')
    ld.setTime()
    # orography = ld.getField(DataType.PHISFC) * Constants.g_wmo
    lat, lon = ld.getLatLon_v2()
    PnmMat = GeoMathKit.getPnmMatrix(lat, Nmax, 2)

    hm = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)

    undulation = GeoidUndulation(elltype).getGeoid(lat, lon).flatten()
    #
    # PHISFC = '../data/ERA5/PHISFC_ERA5_invariant.nc'
    # orography = ReadNC().setPar(PHISFC, DataType.PHISFC).read()[0]
    #
    # '''reduce the spatial resolution: 0.25 X 0.25 ==> 0.5 X 0.5'''
    # orography = orography[0:721:2, 0:1440:2].flatten()
    #
    # '''remove two main tides: S1 and S2'''
    # dt = Detide()
    # tides = {
    #     'P1': True,
    #     'S1': True,
    #     'K1': True,
    #     'N2': True,
    #     'M2': True,
    #     'L2': True,
    #     'T2': True,
    #     'S2': True,
    #     'R2': True,
    #     'T3': True,
    #     'S3': True,
    #     'R3': True
    #     }
    # dt.setTides(tides=tides, tideDir='../result/tides/ERA5/2007-2014/')
    # dt.setStartPoint(startpoint='2007-01-01,00:00:00')
    #
    # '''IBcorrection'''
    # ib = IBcorrection(lat, lon)
    #
    # '''Configure for loading atmosphere driving data'''
    # with open('../settings/loading-ERA5.setting.json', 'r') as fp:
    #     config = json.load(fp)
    # lf = LoadFields().configure(config=config)
    #

    '''Configure for output'''
    # fm = FormatWrite().setRootDir('../result/ERA5sp')
    fm = FormatWrite().setRootDir('../result/vi')

    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        print('---------Date: %s-------' % date)
        cs = CnmSnm(date=date, Nmax=Nmax)

        for time in TimeEpoch:
            print('Computing: %s' % time)
            begin = ti.time()

            # ld.setTime(date, time)
            try:
                ld.setTime(date, time)
            except Exception as err:
                with open(log_file, 'a') as f:
                    f.write(date + ' ' + time + '  ' + err.args[0] + '\n')
                    f.close()
                    continue

            gh = GeopotentialHeight(ld)
            gh.produce_z()
            print("Cost time: %s ms" % ((ti.time() - begin) * 1000))
            InnerI = innerIn(config_integral, gh, ld).setGeoid(Geoid=undulation).deltaI()
            print("Cost time: %s ms" % ((ti.time() - begin) * 1000))
            deltaI = np.array(InnerI).reshape((Nmax + 1, len(lat), len(lon)))
            cnm, snm = hm.analysis(Nmax=Nmax, Gqij=deltaI, lat=lat, lon=lon, PnmMat=PnmMat,
                                   kind=HarAnalysisType.InnerIntegral)

            """Option to save memory in a sacrifice of computation efficiency."""
            # adr, pdg = innerIn(config_integral, gh, ld).setGeoid(Geoid=undulation).deltaI2()
            # cnm, snm = hm.analysis_smaller_meomory_innerIntegral(Nmax=Nmax, adr = adr, pdg=pdg,
            #                                                      lat=lat, lon=lon, PnmMat=PnmMat)

            print("Cost time: %s ms" % ((ti.time() - begin) * 1000))
            '''de-tide'''
            # dt.remove(pressure=sp, date=date, time=time)
            '''IB correction'''
            # sp_af = ib.correct(sp.copy())
            # sp_af = sp
            '''SP integration and harmonic analysis'''
            # cnm, snm = sf2cs.setPressure(pressure=sp, maxDeg=Nmax)
            # cnm, snm = sf2cs.setPressure(pressure=sp_af.flatten(), maxDeg=Nmax)
            '''record the stokes coefficients'''
            cs.add(Cnm=GeoMathKit.CS_2dTo1d(cnm), Snm=GeoMathKit.CS_2dTo1d(snm),
                   epoch=time, date=date, attribute=AODtype.ATM.name)
            '''counting time'''
            print("Cost time: %s ms" % ((ti.time() - begin) * 1000))

        '''write results'''
        fm.setCS(cs).AODstyle()

    pass


def run_hybrid(BeginDate, EndDate):
    import numpy as np
    # from pysrc.TideFit import Detide

    warnings.filterwarnings('ignore')

    log_file = '../log/log_hybrid.txt'

    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]
    # TimeEpoch = ["06:00:00"]
    config_integral = InnerIntegral.defaultConfig(isWrite=True)
    Nmax = config_integral['MaxDegree']

    elltype = EllipsoidType[config_integral['Ellipsoid']]
    ell = RefEllipsoid(elltype)
    LN = LoveNumber('../data/Auxiliary/')

    loveNum = LN.getNumber(Nmax, LoveNumberType.Wang)

    ld = LoadFields(data_path='../data/CRA.grib2/')
    ld.setTime()
    orography = ld.getField(DataType.PHISFC) * Constants.g_wmo
    lat, lon = ld.getLatLon_v2()
    PnmMat = GeoMathKit.getPnmMatrix(lat, Nmax, 2)

    hm = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)

    '''IBcorrection'''
    ib = IBcorrection(lat, lon)

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
    dt.setTides(tides=tides, tideDir='../result/tide/2007_2014/TidePressure/')

    # dt.setTides(tides=tides, tideDir='../result/tide/2010/TidePressure/')

    dt.setRefPoint(refpoint='2007-01-01,00:00:00')

    undulation = GeoidUndulation(elltype).getGeoid(lat, lon).flatten()

    '''Configure for the surface pressure integration'''
    sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
                                loveNumber=LN)

    '''Configure for output'''
    # fm_sp = FormatWrite().setRootDir('../result/sp')
    # fm_upper = FormatWrite().setRootDir('../result/upperair')
    #
    # fm_Primsp = FormatWrite().setRootDir('../result/Primsp')
    fm_vi = FormatWrite().setRootDir('../result/vi180')

    '''2010 tides subtrack'''
    fm_sp = FormatWrite().setRootDir('../result/sp180')
    fm_upper = FormatWrite().setRootDir('../result/upperair180')



    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        print('---------Date: %s-------' % date)
        cs_upper = CnmSnm(date=date, Nmax=Nmax)
        cs_sp = CnmSnm(date=date, Nmax=Nmax)


        # cs_Primsp = CnmSnm(date=date,Nmax=Nmax)
        # cs_vi = CnmSnm(date=date,Nmax=Nmax)


        for time in TimeEpoch:
            print('\nComputing: %s' % time)
            begin = ti.time()

            print('Loading CRA data...')
            # ld.setTime(date, time)
            try:
                ld.setTime(date, time)
            except Exception as err:
                with open(log_file, 'a') as f:
                    f.write(date + ' ' + time + '  ' + err.args[0] + '\n')
                    f.close()
                    continue

            print('Vertical Integration...')
            gh = GeopotentialHeight(ld)
            gh.produce_z()
            delta_vi = innerIn(config_integral, gh, ld).setGeoid(Geoid=undulation).deltaI()

            # '''Cheung made '''
            #
            #
            # deltaI_vi = np.array(delta_vi).reshape((Nmax + 1, len(lat), len(lon)))
            # cnm_vi, snm_vi = hm.analysis(Nmax=Nmax, Gqij=deltaI_vi, lat=lat, lon=lon, PnmMat=PnmMat,
            #                              kind=HarAnalysisType.InnerIntegral)

            """Option to save memory in a sacrifice of computation efficiency."""
            # adr, pdg = innerIn(config_integral, gh, ld).setGeoid(Geoid=undulation).deltaI2()
            # cnm, snm = hm.analysis_smaller_meomory_innerIntegral(Nmax=Nmax, adr = adr, pdg=pdg,
            #                                                      lat=lat, lon=lon, PnmMat=PnmMat)

            # print("Cost time: %s ms" % ((ti.time() - begin) * 1000))

            print('Surface Integration...')
            sp = ld.getField(DataType.PSFC)
            '''SP integration'''
            deltaI_sp = sf2cs.setPressure_inner(pressure=sp, maxDeg=Nmax)

            # '''Cheung made'''
            # deltaI_Primsp = np.array(deltaI_sp).reshape((Nmax + 1, len(lat), len(lon)))
            # cnm_Primsp, snm_Primsp = hm.analysis(Nmax=Nmax,Gqij=deltaI_Primsp,lat=lat,lon=lon,PnmMat=PnmMat,
            #                                      kind=HarAnalysisType.InnerIntegral)


            '''Obtain the Upper Air Component'''
            print('Obtain the Upper Air Component...')
            delta_upper = (delta_vi - deltaI_sp) / (1 + loveNum[:, None])
            deltaI_upper = delta_upper.reshape((Nmax + 1, len(lat), len(lon)))
            cnm_upper, snm_upper = hm.analysis(Nmax=Nmax, Gqij=deltaI_upper, lat=lat, lon=lon, PnmMat=PnmMat,
                                               kind=HarAnalysisType.InnerIntegral)

            '''Obtain the Surface Component'''
            print('Obtain the SP Component...')
            '''de-tide'''
            sp_af = dt.remove(pressure=sp, date=date, time=time)
            '''IB correction'''
            sp_af = ib.correct(sp_af)
            # sp_af = sp

            deltaI_sp_removal = sf2cs.setPressure_inner(pressure=sp_af, maxDeg=Nmax)
            deltaI_sp_removal = deltaI_sp_removal.reshape((Nmax + 1, len(lat), len(lon)))
            cnm_sp, snm_sp = hm.analysis(Nmax=Nmax, Gqij=deltaI_sp_removal, lat=lat, lon=lon, PnmMat=PnmMat,
                                         kind=HarAnalysisType.InnerIntegral)

            '''record the stokes coefficients'''
            cs_upper.add(Cnm=cnm_upper, Snm=snm_upper,
                         epoch=time, date=date, attribute=AODtype.ATM.name)
            cs_sp.add(Cnm=cnm_sp, Snm=snm_sp,
                      epoch=time, date=date, attribute=AODtype.ATM.name)

            # '''Cheung made'''
            # cs_vi.add(Cnm=cnm_vi, Snm=snm_vi,
            #           epoch=time, date=date, attribute=AODtype.ATM.name)

            # cs_Primsp.add(Cnm=cnm_Primsp,Snm=snm_Primsp,
            #               epoch=time, date=date, attribute=AODtype.ATM.name)


            '''counting time'''
            print("Cost time: %s ms" % ((ti.time() - begin) * 1000))
            print('Finish!')



        '''write results'''
        fm_sp.setCS(cs_sp).AODstyle()
        fm_upper.setCS(cs_upper).AODstyle()

        # '''Cheung made'''
        # fm_vi.setCS(cs_vi).AODstyle()
        # fm_Primsp.setCS(cs_Primsp).AODstyle()

    pass

def get_product(BeginDate, EndDate):
    Nmax = 180

    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]


    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        y = date.split('-')[0]
        m = date.split('-')[1]
        num = 0
        for a in np.arange(0, Nmax + 1):
            for b in np.arange(0, a + 1):
                num += 1

        print('---------Date: %s-------' % date)
        # savepath = '../result/product/{}-{}/'.format(y,m)
        savepath = '../result/product180/{}-{}/'.format(y, m)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        with open(savepath+'{}.asc'.format(date),'w') as f:
            f.write(HConfig().Message(Nmax=Nmax))
            for time in TimeEpoch:
                print('\nComputing: %s' % time)
                if time == '00:00:00':
                    message = 'DATA SET  0:   {} COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(num,date,time)
                elif time == '06:00:00':
                    message = 'DATA SET  1:   {} COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(num,date, time)
                elif time == '12:00:00':
                    message = 'DATA SET  2:   {} COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(num,date, time)
                elif time == '18:00:00':
                    message = 'DATA SET  3:   {} COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(num,date, time)
                f.write(message)
                sp = AOD_GFZ().load('../result/sp180').setType(AODtype.ATM)
                upperair = AOD_GFZ().load('../result/upperair180').setType(AODtype.ATM)
                mean = HConfig().MeanCS()

                spCS = sp.setTime(date,time).getCS(Nmax=Nmax)
                spCS = np.array(spCS)
                upperairCS = upperair.setTime(date,time).getCS(Nmax=Nmax)
                upperairCS = np.array(upperairCS)
                mean = np.array(mean)
                product = spCS+upperairCS-mean
                count = 0
                for i in np.arange(0, Nmax + 1):
                    for j in np.arange(0, i + 1):
                        f.write(str(i).rjust(5) + str(j).rjust(5) + str(product[0,count]).rjust(28) + str(product[1,count]).rjust(
                            28) + '\n')
                        count += 1

if __name__ == '__main__':
    # run_SP_v1(BeginDate='2002-01-05', EndDate='2002-01-10')
    # run_VI_v1(BeginDate='2002-01-01', EndDate='2002-01-05')
    # run_VI_v1(BeginDate='2002-01-01', EndDate='2015-01-01')
    run_hybrid(BeginDate='2006-08-01', EndDate='2014-12-31')
    # get_product(BeginDate='2018-12-01',EndDate='2018-12-31')
