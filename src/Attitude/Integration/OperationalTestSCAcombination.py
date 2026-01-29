"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/12/23
@Description:
"""
"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/21
@Description:
"""
import sys

sys.path.append('../')
from SCA1Ato1B import SCA1Ato1B, SCAinterOption
from IMU1Ato1B import IMU1Ato1B
from IMU1Bprocess import IMU1Bprocess, IMUinterpOption
from kalman import NormalKalmanFilter, MEKF_7Vars_JPL, MEKF_7Vars_JPL_combination, GetFusionRes
from Setting import SatID, Mission, IMUID
from GetInstrument import GetInstrument_L1A, GetInstrument_L1B
from CRNfilter import ReSample
from YKF import YKF_NoCov, YKF_Cov
import numpy as np


def Run_L1AtoL1B_perday(date=None, sat=SatID.D, imu_disable=None):
    """
    L1a to L1b for a given day
    :return:
    """

    # IMU1Bprocess.ConfigDMforSF2Axis()
    # imu_disable = IMUID.No_4
    IMU1Bprocess.ConfigDMforSF2Axis(disable=imu_disable)
    SCA1Ato1B.ConfigSCF2SRF()

    '''Step 1: basic configuration'''
    # sat = SatID.D
    mission = Mission.GRACE_FO
    # date = '2019-01-03'
    L1a_dir = '../data/GRACE_FO/RL04/L1A'
    L1b_dir = '../data/GRACE_FO/RL04/L1B'
    IMUinterpOp = IMUinterpOption.EightHZ
    SCAinterOp = SCAinterOption.OneHZ

    '''Step 2: get relevant data'''
    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir(L1b_dir)
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir(L1a_dir)
    SCA1A = L1A.getSCA(sat)
    if SCA1A == None:
        return True
        # print('SCA-1A is an empty file, so that this day is skipped.')
        # exit()
    IMU1A = L1A.getIMU(sat)

    '''Step 3: convert IMU1A to 1B and generate the standard L1B product'''
    IMU1Ato1B_instance = IMU1Ato1B(satID=sat, date=date, IMU1A=IMU1A, TIM1B=TIM1B, CLK1B=CLK1B)
    IMU1Ato1B_instance.produce(mission=mission, disable=imu_disable)

    '''Step 4: A further process of IMU1B in preparation for the data fusion '''
    imu = IMU1Bprocess(sat=sat, date=date, mission=mission, disable=imu_disable)
    imu.angular_rate(interpOption=IMUinterpOp).record()

    '''Step 5: convert SCA1A to 1B and generate the non-/standard L1B product'''
    SCA = SCA1Ato1B(satID=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)
    SCA.configInterp(interpOption=SCAinterOp)

    isCombined = False
    # SCA.produce_combine_last(mission=Mission.GRACE_FO, isCombined=isCombined)
    # SCA.produce_combine_first(mission=Mission.GRACE_FO, isCombined=isCombined)
    SCA.produce_NoCombine_NoCoordinateTransform(mission=Mission.GRACE_FO, version='02')


    return False


def Run_fusion_perday(date=None, sat=SatID.D, isFirst=False):
    """
    Run the data fusion per day
    :return:
    """

    # IMU1Bprocess.ConfigDMforSF2Axis()

    '''Step 1: basic configuration'''
    # sat = SatID.C
    mission = Mission.GRACE_FO
    # date = '2019-01-02'
    L1a_dir = '../data/GRACE_FO/RL04/L1A'
    L1b_dir = '../data/GRACE_FO/RL04/L1B'
    IMUinterpOp = IMUinterpOption.EightHZ
    SCAinterOp = SCAinterOption.OneHZ

    # kf = NormalKalmanFilter(date=date, Mission=mission, sat=sat)
    # kf = MEKF_7Vars_JPL(date=date, Mission=mission, sat=sat)
    # kf = MEKF_7Vars_JPL_combination(date=date, Mission=mission, sat=sat)
    # kf = YKF_NoCov(date=date, sat=sat, dir='../result/product/Method/free/')
    kf = YKF_Cov(date=date, sat=sat, dir='../result/product/Method/cov/')
    kf.configInput(IMUoption=IMUinterpOp, SCAoption=SCAinterOp)
    SCA1Ato1B.ConfigSCF2SRF()
    kf.setQSA(QSA = SCA1Ato1B.SCF2SRF[sat])
    kf.load()

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir(L1b_dir)
    SCA1B = L1B.getSCA(sat=sat)
    if len(SCA1B) == 0:
        return True

    index = 1
    iniT = SCA1B[0][index]
    iniX = np.array(list(SCA1B[1:, index]) + [0, 0, 0])

    iniP = np.diagflat([1e-5, 1e-5, 1e-5, 1e-5, 1e-10, 1e-10, 1e-10])
    if sat == SatID.D:
        iniB = [-3.7298e-7, -1.3832e-6, 5.0151e-7]  # D
    else:
        iniB = [1.5062e-6, -4.1724e-7, 8.9792e-7]  # C

    if isFirst:
        kf.filter(iniQ=None, iniP=iniP, iniT=None, iniB=iniB, onlyPredict=False, isFirst=isFirst, version='HUST01')
    else:
        kf.filter(iniQ=None, iniP=None, iniT=None, iniB=None, onlyPredict=False, isFirst=isFirst, version='HUST01')

    # kf.filter(iniQ=None, iniP=None, iniT=602164802, iniB=None, onlyPredict=False, version='HUST01')
    # kf.filter(iniQ=None, iniP=None, iniT=None, iniB=iniB, onlyPredict=False, version='HUST01')
    return False


def Run_resample_perday(date=None, sat=SatID.D, fileDir = '../result/product/Method/free/'):
    # sat = SatID.C
    mission = Mission.GRACE_FO
    # date = '2019-01-02'
    rs = ReSample(mission=mission, date=date, sat=sat, fileDir=fileDir)
    rs.run()
    pass


def Run_reformat(date='2017-01-06'):
    fileDir = '../result/product/HUGG/01/'
    lr = GetFusionRes(mission=Mission.GRACE_FO, date=date).configDir('../result/product/GRACE_FO/RL04/L1B')
    for sat in [SatID.C, SatID.D]:
        try:
            SCA = lr.getFusion(sat=sat)
        except IOError as e:
            print(e.strerror + ': %s, Sat %s' % (date, sat.name))
            continue
        else:
            filename = 'SCA1B_%s_%s_HUGG_01.txt' % (date, sat.name)
            with open(fileDir + filename, 'w') as file:
                file.write('%-25s%1s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
                file.write('%-25s%1s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST and APM'))
                file.write('%-25s%1s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
                file.write('%-25s%1s%-31s \n' % ('Author', ':', 'Fan Yang, Lei Liang, ChangQing Wang, ZhiCai Luo'))
                file.write('%-25s%1s%-31s \n' % ('Contact', ':', 'lianglei@apm.ac.cn'))
                file.write('%-25s%1s%-31s \n' % ('Variables', ':', 'gps_time(second), GRACEFO_id, q0, q1, q2, q3'))
                file.write('# End of YAML header \n')

                for index in range(len(SCA[0,:])):
                    vv = SCA[1:, index]
                    vt = SCA[0, index]
                    file.write('%-11d%2s   % 20.16f   % 20.16f   % 20.16f  % 20.16f\n' %
                               (int(vt), sat.name, vv[0], vv[1], vv[2], vv[3]))
            pass

    pass


def Run_complete(begin='2019-01-01', end='2019-01-31', sat=SatID.D):
    from pysrc.LOS import LOS
    from pysrc.GeoMathKit import GeoMathKit
    from pysrc.Exp import PointingError

    daylist = GeoMathKit.dayListByDay(begin, end)

    # a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
    for day in daylist:
        day = day.strftime("%Y-%m-%d")
        print('\n****************************************************')
        print('GRACE %s attitude determination for %s ...' % (sat.name, day))
        print('****************************************************\n')
        # isEmpty = Run_L1AtoL1B_perday(date=day, sat=sat, imu_disable=None)
        # if isEmpty:
        #     print('SCA-1A is an empty file, so that this day is skipped.')
        #     continue
        isEmpty = Run_fusion_perday(date=day, sat=sat, isFirst=False)
        if isEmpty:
            print('Official SCA-1B is an empty file, so that this day is skipped.')
            continue
        Run_resample_perday(date=day, sat=sat, fileDir='../result/product/Method/cov/')


    pass



if __name__ == '__main__':
    Run_complete(begin='2019-01-01', end='2019-01-01', sat=SatID.C)
    # Run_reformat()
