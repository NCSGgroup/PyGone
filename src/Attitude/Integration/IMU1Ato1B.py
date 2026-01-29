"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/6
@Description:
"""
from Setting import SatID, IMUID, Mission
import numpy as np
import os


class IMU1Ato1B:
    """
    The smallest processing unit is ONE DAY.
    """

    def __init__(self, satID: SatID, date, IMU1A, TIM1B, CLK1B):
        self.sat = satID
        self.date = date
        self.__IMU1A = IMU1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B

        pass

    def __TimeTagTransformation(self):
        """
        Step 1: time tag transformation: OBC--->RCV, RCV--->GPS
        :return:
        """

        '''Step 1: OBC--->RCV'''
        imu = self.__IMU1A

        tim = self.__TIM1B

        for key in IMUID:
            imuValue = imu[key.name]
            # remove the first epoch to avoid too big number
            x = imuValue[0] - tim[0][0] + imuValue[1] * 1e-6
            xp = tim[0] - tim[0][0]
            yp = tim[1] - tim[0][0] + tim[2] * 1e-9
            # quality check for TIM1B
            valid = tim[3] < 1000
            xp = xp[valid]
            yp = yp[valid]
            V = np.interp(x=x, xp=xp, fp=yp)

            V_int = np.trunc(V) + tim[0][0]
            V_frac = V - np.trunc(V)
            imuValue[0] = V_int
            imuValue[1] = V_frac * 1e6
            imu[key.name] = imuValue

        '''Step 2: RCV ---> GPS'''
        clk = self.__CLK1B
        # make a judgement on the quality
        index = np.where(clk[2] > 0)[0]
        # TODO: this should be refined by dividing the arcs
        if index[0] == 0 and index[-1] == (len(clk[2]) - 1):
            pass
        else:
            raise Warning('Take care of CLK1B file')

        clk = clk[:, np.where(clk[2] < 0.1)[0]]
        for key in IMUID:
            imuValue = imu[key.name]
            x = imuValue[0] - clk[0][0] + imuValue[1] * 1e-6
            xp = clk[0] - clk[0][0]
            yp = clk[1]
            V = np.interp(x=x, xp=xp, fp=yp)
            V_int = np.trunc(V)
            V_frac = V - np.trunc(V)

            # V_int[V_frac < 0] -= 1
            # V_frac[V_frac < 0] += 1

            imuValue[0] += V_int
            imuValue[1] += V_frac * 1e6

            imuValue[0][imuValue[1] < 0] -= 1
            imuValue[1][imuValue[1] < 0] += 1e6

            imu[key.name] = imuValue
            pass

        return imu

    def produce(self, mission: Mission, disable: IMUID, version='HUST01'):
        """
        produce the HUST-IMU1B product following the format of the official one.
        :return:
        """

        imu = self.__TimeTagTransformation()

        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)

        print('Create IMU-L1B for date: %s' % self.date)

        filename = 'IMU1B_%s_%s_%s.txt' % (self.date, self.sat.name, version)

        with open(fileDir + filename, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'IMU Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            ss = max([len(imu[key.name][0]) for key in IMUID])

            for index in range(ss):

                for key in IMUID:
                    if key is disable:
                        continue
                    if index >= len(imu[key.name][0]):
                        continue
                    vv = imu[key.name]
                    if np.shape(vv)[1] == 0:
                        continue
                    file.write('%9i %7i %3s %3s %3s   %20.15f\n' %
                               (vv[0][index], vv[1][index], 'G', self.sat.name, key.name[-1], vv[2][index]))

        pass


def demo1():
    from pysrc.GetInstrument import GetInstrument_L1A, GetInstrument_L1B, Mission, SatID

    date = '2018-12-02'

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
    IMU1A = L1A.getIMU(SatID.C)
    # L1A.getSCA(SatID.C)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    # L1B.getIMU(SatID.C)
    TIM1B = L1B.getTIM(SatID.C)
    CLK1B = L1B.getCLK(SatID.C)

    # IMU1Ato1B.DMforSF2Axis()

    IMU = IMU1Ato1B(satID=SatID.C, date=date, IMU1A=IMU1A, TIM1B=TIM1B, CLK1B=CLK1B)

    IMU.produce(mission=Mission.GRACE_FO)

    pass


if __name__ == '__main__':
    demo1()
