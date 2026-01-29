"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2023/2/8
@Description:
"""
import copy
import sys

sys.path.append('../')

from Format import CnmSnm, FormatWrite
from Harmonic import Harmonic, LoveNumber, LoveNumberType, GeoMathKit
# from IntegralMethods import innerIn, InnerIntegral, GeopotentialHeight, InterpOption
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
        'S2': True,
        'R2': True,
        'T3': True,
        'S3': True,
        'R3': True
    }
    dt.setTides(tides=tides, tideDir='../result/tide/2007_2014/TidePressure/')
    dt.setRefPoint(refpoint='2007-01-01,00:00:00')

    undulation = GeoidUndulation(elltype).getGeoid(lat, lon).flatten()

    '''Configure for the surface pressure integration'''
    sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
                                loveNumber=LN)

    '''Configure for output'''
    fm_sp = FormatWrite().setRootDir('../result/sp')
    fm_upper = FormatWrite().setRootDir('../result/upperair')
    fm_mean = FormatWrite().setRootDir('../result/hybrid/cra')

    '''made by Cheung:'''
    fm_spmean =FormatWrite().setRootDir('../result/mean/meansp')
    fm_uppermean = FormatWrite().setRootDir('../result/mean/meanupper')
    fm_product = FormatWrite().setRootDir('../result/product')


    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    '''calculate the mean of field'''


    count = 1
    Cspsum = 0
    Sspsum = 0
    Cuppersum = 0
    Suppersum = 0
    for day in GeoMathKit.dayListByDay(begin='2003-01-01',end='2003-01-05'):
        date = day.strftime("%Y-%m-%d")
        print('day {} is input for mean'.format(date))
        cs_upper = CnmSnm(date=date,Nmax=Nmax)
        cs_sp = CnmSnm(date=date,Nmax=Nmax)
        for time in TimeEpoch:
            gh = GeopotentialHeight(ld)
            gh.produce_z()
            delta_vi = innerIn(config_integral, gh, ld).setGeoid(Geoid=undulation).deltaI()

            sp = ld.getField(DataType.PSFC)
            deltaI_sp = sf2cs.setPressure_inner(pressure=sp, maxDeg=Nmax)

            delta_upper = (delta_vi - deltaI_sp) / (1 + loveNum[:, None])
            deltaI_upper = delta_upper.reshape((Nmax + 1, len(lat), len(lon)))
            cnm_upper, snm_upper = hm.analysis(Nmax=Nmax, Gqij=deltaI_upper, lat=lat, lon=lon, PnmMat=PnmMat,
                                               kind=HarAnalysisType.InnerIntegral)
            sp_af = dt.remove(pressure=sp, date=date, time=time)
            sp_af = ib.correct(sp_af)

            deltaI_sp_removal = sf2cs.setPressure_inner(pressure=sp_af, maxDeg=Nmax)
            deltaI_sp_removal = deltaI_sp_removal.reshape((Nmax + 1, len(lat), len(lon)))
            cnm_sp, snm_sp = hm.analysis(Nmax=Nmax, Gqij=deltaI_sp_removal, lat=lat, lon=lon, PnmMat=PnmMat,
                                         kind=HarAnalysisType.InnerIntegral)

            cs_upper.add(Cnm=cnm_upper, Snm=snm_upper,
                         epoch=time, date=date, attribute=AODtype.ATM.name)
            cs_sp.add(Cnm=cnm_sp, Snm=snm_sp,
                      epoch=time, date=date, attribute=AODtype.ATM.name)
        # Cspsum += np.array(list((cs_sp.Cnm.values())))
        # Sspsum += np.array(list((cs_sp.Snm.values())))
        # Cuppersum += np.array(list((cs_upper.Cnm.values())))
        # Suppersum += np.array(list((cs_upper.Snm.values())))

        Cspsum += np.mean(np.array(list(cs_sp.Cnm.values())), axis=0)
        Sspsum += np.mean(np.array(list(cs_sp.Snm.values())), axis=0)
        Cuppersum += np.mean(np.array(list(cs_upper.Cnm.values())), axis=0)
        Cuppersum += np.mean(np.array(list(cs_upper.Snm.values())), axis=0)
        count +=1
    cspmean = Cspsum/count
    sspmean = Sspsum/count
    cuppermean = Cuppersum/count
    suppermean = Suppersum/count
    print('day {} is calculated over'.format(date))



    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        print('---------Date: %s-------' % date)
        cs_upper = CnmSnm(date=date, Nmax=Nmax)
        cs_sp = CnmSnm(date=date, Nmax=Nmax)


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
            # deltaI_vi = np.array(InnerI).reshape((Nmax + 1, len(lat), len(lon)))
            # cnm_vi, snm_vi = hm.analysis(Nmax=Nmax, Gqij=deltaI, lat=lat, lon=lon, PnmMat=PnmMat,
            #                        kind=HarAnalysisType.InnerIntegral)

            """Option to save memory in a sacrifice of computation efficiency."""
            # adr, pdg = innerIn(config_integral, gh, ld).setGeoid(Geoid=undulation).deltaI2()
            # cnm, snm = hm.analysis_smaller_meomory_innerIntegral(Nmax=Nmax, adr = adr, pdg=pdg,
            #                                                      lat=lat, lon=lon, PnmMat=PnmMat)

            # print("Cost time: %s ms" % ((ti.time() - begin) * 1000))

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



            '''counting time'''
            print("Cost time: %s ms" % ((ti.time() - begin) * 1000))
            print('Finish!')


        delta_cs_sp = copy.deepcopy(cs_sp)
        delta_cs_upper = copy.deepcopy(cs_upper)
        delta_cs = copy.deepcopy(delta_cs_sp)

        # for key in delta_cs_sp.Cnm.keys():
        #     delta_cs_sp.Cnm[key] -= cmean
        #     delta_cs_sp.Snm[key] -= smean
        #     delta_cs_upper.Cnm[key] -= cmean
        #     delta_cs_upper.Snm[key] -= smean
        #     delta_cs.Cnm[key] = delta_cs_sp.Cnm[key] + delta_cs_upper.Cnm[key]
        #     delta_cs.Snm[key] = delta_cs_sp.Snm[key] + delta_cs_upper.Snm[key]

        for key in delta_cs_sp.Cnm.keys():
            delta_cs_sp.Cnm[key] -= cspmean
            delta_cs_sp.Snm[key] -= sspmean
            delta_cs_upper.Cnm[key] -= cuppermean
            delta_cs_upper.Snm[key] -= suppermean
            delta_cs.Cnm[key] = delta_cs_sp.Cnm[key] + delta_cs_upper.Cnm[key]
            delta_cs.Snm[key] = delta_cs_sp.Snm[key] + delta_cs_upper.Snm[key]


        '''write results'''
        fm_sp.setCS(cs_sp).AODstyle()
        fm_upper.setCS(cs_upper).AODstyle()
        # fm_mean.setCS(delta_cs).AODstyle()

        fm_spmean.setCS(delta_cs_sp).AODstyle()
        fm_uppermean.setCS(delta_cs_upper).AODstyle()
        fm_product.setCS(delta_cs).AODstyle()




    pass


if __name__ == '__main__':
    # run_SP_v1(BeginDate='2002-01-01', EndDate='2020-12-30')
    # run_VI_v1(BeginDate='2002-01-01', EndDate='2020-12-30')
    # run_VI_v1(BeginDate='2002-01-01', EndDate='2015-01-01')
    run_hybrid(BeginDate='2002-01-01', EndDate='2002-01-05')

