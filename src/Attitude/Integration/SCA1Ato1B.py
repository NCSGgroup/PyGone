"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/6
@Description:
"""

from Setting import SatID, SCAID, Mission
from SCAweighting import SCAweighting
import numpy as np
import os
import quaternion
from GeoMathKit import GeoMathKit
from enum import Enum


class SCAinterOption(Enum):
    NoInterp = 0
    OneHZ = 1
    TwoHZ = 2


class SCA1Ato1B:
    SCF2SRF = None

    def __init__(self, satID: SatID, date, SCA1A, TIM1B, CLK1B):
        self.sat = satID
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B

        self.__interpOption = SCAinterOption.NoInterp
        pass

    def configInterp(self, interpOption: SCAinterOption):
        """

        :param interpOption: 0--> no interpolation,
        1--> interp at integer seconds, 2---> interp at every half sec
        :return:
        """
        self.__interpOption = interpOption
        pass

    @staticmethod
    def ConfigSCF2SRF():
        QSA = {}
        file = '../data/GRACE_FO/Auxiliary/QSA1B_2018-05-22_C_04.txt'
        skip = GeoMathKit.ReadEndOfHeader(file_name=file, end_flag='End of YAML header')
        q = np.loadtxt(file, skiprows=skip, usecols=[3, 4, 5, 6])
        QSA[SatID.C] = quaternion.as_quat_array(q)

        file = '../data/GRACE_FO/Auxiliary/QSA1B_2018-05-22_D_04.txt'
        skip = GeoMathKit.ReadEndOfHeader(file_name=file, end_flag='End of YAML header')
        q = np.loadtxt(file, skiprows=skip, usecols=[3, 4, 5, 6])
        QSA[SatID.D] = quaternion.as_quat_array(q)

        SCA1Ato1B.SCF2SRF = QSA

        pass

    def __TimeTagTransformation(self):
        """
        Step 1: time tag transformation: OBC--->RCV, RCV--->GPS
        :return:
        """

        '''Step 1: OBC--->RCV'''
        sca = self.__SCA1A

        tim = self.__TIM1B

        for key in SCAID:
            scaValue = sca[key.name]
            # remove the first epoch to avoid too big number
            x = scaValue[0] - tim[0][0] + scaValue[1] * 1e-6
            xp = tim[0] - tim[0][0]
            yp = tim[1] - tim[0][0] + tim[2] * 1e-9
            # quality check for TIM1B
            valid = tim[3] < 1000
            xp = xp[valid]
            yp = yp[valid]
            V = np.interp(x=x, xp=xp, fp=yp)

            V_int = np.trunc(V) + tim[0][0]
            V_frac = V - np.trunc(V)
            scaValue[0] = V_int
            scaValue[1] = V_frac * 1e6
            sca[key.name] = scaValue

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
        for key in SCAID:
            scaValue = sca[key.name]
            x = scaValue[0] - clk[0][0] + scaValue[1] * 1e-6
            xp = clk[0] - clk[0][0]
            yp = clk[1]
            V = np.interp(x=x, xp=xp, fp=yp)
            V_int = np.trunc(V)
            V_frac = V - np.trunc(V)

            # V_int[V_frac < 0] -= 1
            # V_frac[V_frac < 0] += 1

            scaValue[0] += V_int
            scaValue[1] += V_frac * 1e6

            scaValue[0][scaValue[1] < 0] -= 1
            scaValue[1][scaValue[1] < 0] += 1e6

            sca[key.name] = scaValue
            pass

        return sca

    def __interp(self, sca):

        if self.__interpOption == SCAinterOption.NoInterp:
            return sca

        sca_new = None

        rr1 = np.round(sca[SCAID.No_1.name][0][0])
        rr2 = np.round(sca[SCAID.No_1.name][0][-1]) + 1

        time = None
        if self.__interpOption == SCAinterOption.OneHZ:
            time = np.arange(rr1, rr2)
            pass
        elif self.__interpOption == SCAinterOption.TwoHZ:
            time = np.arange(rr1, rr2, 0.5)
            pass

        for ID in SCAID:
            aa = sca[ID.name]

            # delete the data of bad quality
            x = np.where(aa[6, :] < 6)[0]
            bb = aa[:, x]
            starttime = time[0]
            time1 = bb[0] - starttime + bb[1] * 1e-6
            time2 = time - starttime

            qb = quaternion.as_quat_array(bb[2:6, :].T)
            # interpolation
            new = quaternion.squad(R_in=qb, t_in=time1, t_out=time2)
            qc = quaternion.as_float_array(new).T

            # add the quality flag to the interpolated data
            if self.__interpOption == SCAinterOption.OneHZ:
                quality = np.arange(0, len(aa[0, :]), 2)[0:len(time2)]
            elif self.__interpOption == SCAinterOption.TwoHZ:
                quality = np.arange(0, len(aa[0, :]), 1)[0:len(time2)]
                # TODO: It has a problem if there is a gap!!

            sca_new = np.zeros((len(aa[:, 0]) - 1, len(time2)))
            sca_new[0] = time2 + starttime
            # if self.__interpOption == SCAinterOption.OneHZ:
            #     sca_new[1][1:len(time2):2] = 0
            # elif self.__interpOption == SCAinterOption.TwoHZ:
            #     sca_new[1][1:len(time2):2] = 500000
            # sca_new[1] = ID.name[-1]
            sca_new[1:5] = qc
            sca_new[5:] = aa[6:, quality]
            # replace the first row
            # TODO: the first epoch has a problem to be fixed in future.
            sca_new[1:, 0] = aa[2:, 0]
            sca[ID.name] = sca_new

            pass

        return sca

    def __interp2(self, sca: np.ndarray):

        if self.__interpOption == SCAinterOption.NoInterp:
            return sca

        sca_new = None

        rr1 = np.round(sca[0][0])
        rr2 = np.round(sca[-1][0]) + 1

        time = None
        if self.__interpOption == SCAinterOption.OneHZ:
            time = np.arange(rr1, rr2)
            pass
        elif self.__interpOption == SCAinterOption.TwoHZ:
            time = np.arange(rr1, rr2, 0.5)
            pass

        aa = sca

        starttime = time[0]
        time1 = aa[:, 0] - starttime + aa[:, 1] * 1e-6
        time2 = time - starttime

        qb = quaternion.as_quat_array(aa[:, 2:6])
        # interpolation
        new = quaternion.squad(R_in=qb, t_in=time1, t_out=time2)
        qc = quaternion.as_float_array(new)
        sca_new = np.zeros((5, len(time2)))
        sca_new[0] = time2 + starttime
        sca_new[1:5] = qc.T
        # TODO: the first and the last epoch has a problem to be fixed in future.
        #  (interpolation error is huge at this epoch)
        sca_new[1:, 0] = aa[0, 2:]
        # sca_new[1:, -1] = aa[-1, 2:]

        return sca_new

    def produce_combine_last(self, mission: Mission, version='HUST01', isCombined=False):
        """
        produce the HUST-SCA1B product following the format of the official one.
        Interpolation first and then combination
        :return:
        """

        '''Step 1: Time-tag transformation'''
        sca = self.__TimeTagTransformation()

        '''Step 2: SCF --> SRF'''
        QSA = SCA1Ato1B.SCF2SRF
        for i in range(3):
            # three star cameras

            a = sca['No_%s' % (i + 1)][2:6, :].T
            qa = quaternion.as_quat_array(a)
            qb = qa * QSA[self.sat][i]
            qc = quaternion.as_float_array(qb).T
            sca['No_%s' % (i + 1)][2:6] = qc
            pass

        '''Step 3: Flip signs'''
        for id in SCAID:
            '''judge by the vector of quaternion'''
            a = sca[id.name][2:6, :].T
            quality = sca[id.name][6, :]

            m = [a[0, :]]

            for i in range(len(a[:, 0]) - 1):
                if np.dot(a[i + 1, 1:], m[-1][1:]) < 0 and quality[i] <= 6:
                    m.append(-a[i + 1, :])
                else:
                    m.append(a[i + 1, :])

            a = np.array(m)

            sca[id.name][2:6, :] = a.T

        '''Step 4: interpolation'''
        sca = self.__interp(sca)

        print('Create SCA-L1B for date: %s' % self.date)

        '''Step 5: combination'''
        if isCombined:
            sca = SCAweighting(SCA=sca, QSA=QSA[self.sat])
            sca = sca.combine_1t()
            pass

        '''Step 6: format writing'''
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)

        '''record the combined result'''
        if isCombined:
            filename = 'SCA1B_%s_%s_%s_combined_%s.txt' % (self.date, self.sat.name, self.__interpOption.name, version)
            with open(fileDir + filename, 'w') as file:
                file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
                file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
                file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
                file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
                file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
                file.write('END OF HEADER \n')

                for x in sca:
                    file.write('%16.5f  %3s %3s   %20.16f  %20.16f  %20.16f  %20.16f\n' %
                               (x[0], 'G', self.sat.name, x[1], x[2], x[3], x[4]))

            return sca

        filename = 'SCA1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__interpOption.name, version)
        with open(fileDir + filename, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            for index in range(len(sca[SCAID.No_1.name][0])):

                for key in SCAID:
                    vv = sca[key.name]
                    file.write('%16.5f  %3s %3s %3s   %20.16f  %20.16f  %20.16f  %20.16f %5d %5d %5d\n' %
                               (vv[0][index], 'G', self.sat.name, key.name[-1], vv[1][index],
                                vv[2][index], vv[3][index], vv[4][index], vv[5][index], vv[6][index], vv[7][index]
                                ))

        return sca

    def produce_combine_first(self, mission: Mission, version='HUST01', isCombined=True):
        """
        produce the HUST-SCA1B product following the format of the official one.
        Combination first and then interpolation
        :return:
        :param mission:
        :param version:
        :param isCombined:
        :return:
        """
        assert isCombined == True
        print('Create SCA-L1B for date: %s' % self.date)

        '''Step 1: Time-tag transformation'''
        sca = self.__TimeTagTransformation()

        '''Step 2: SCF --> SRF'''
        QSA = SCA1Ato1B.SCF2SRF
        for i in range(3):
            # three star cameras

            a = sca['No_%s' % (i + 1)][2:6, :].T
            qa = quaternion.as_quat_array(a)
            qb = qa * QSA[self.sat][i]
            qc = quaternion.as_float_array(qb).T
            sca['No_%s' % (i + 1)][2:6] = qc
            pass

        '''Step 3: Flip signs'''
        for id in SCAID:
            '''judge by the vector of quaternion'''
            a = sca[id.name][2:6, :].T
            quality = sca[id.name][6, :]

            m = [a[0, :]]

            for i in range(len(a[:, 0]) - 1):
                if np.dot(a[i + 1, 1:], m[-1][1:]) < 0 and quality[i] <= 6:
                    m.append(-a[i + 1, :])
                else:
                    m.append(a[i + 1, :])

            a = np.array(m)

            sca[id.name][2:6, :] = a.T

        '''Step 4: Combination'''
        sw = SCAweighting(SCA=sca, QSA=QSA[self.sat])
        sca = sw.combine_2t()

        '''Step 5: Interpolation '''
        sca = self.__interp2(sca=np.array(sca))

        '''Step 6: format writing'''
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)

        '''record the combined result'''
        filename = 'SCA1B_%s_%s_%s_combined_%s.txt' % (self.date, self.sat.name, self.__interpOption.name, version)
        with open(fileDir + filename, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            for x in list(sca.T):
                file.write('%16.5f  %3s %3s   %20.16f  %20.16f  %20.16f  %20.16f\n' %
                           (x[0], 'G', self.sat.name, x[1], x[2], x[3], x[4]))

        return sca

    def produce_NoCombine_NoCoordinateTransform(self, mission: Mission, version='HUST01'):
        """
        produce the HUST-SCA1B product following the format of the official one.
        Interpolation first and then combination
        :return:
        """

        '''Step 1: Time-tag transformation'''
        sca = self.__TimeTagTransformation()

        '''Step 3: Flip signs'''
        for id in SCAID:
            '''judge by the vector of quaternion'''
            a = sca[id.name][2:6, :].T
            quality = sca[id.name][6, :]

            m = [a[0, :]]

            for i in range(len(a[:, 0]) - 1):
                if np.dot(a[i + 1, 1:], m[-1][1:]) < 0 and quality[i] <= 6:
                    m.append(-a[i + 1, :])
                else:
                    m.append(a[i + 1, :])

            a = np.array(m)

            sca[id.name][2:6, :] = a.T

        '''Step 4: interpolation'''
        sca = self.__interp(sca)

        print('Create SCA-L1B for date: %s' % self.date)

        '''Step 6: format writing'''
        fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, self.date)

        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)

        filename = 'SCA1B_%s_%s_%s_%s.txt' % (self.date, self.sat.name, self.__interpOption.name, version)
        with open(fileDir + filename, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            for index in range(len(sca[SCAID.No_1.name][0])):

                for key in SCAID:
                    vv = sca[key.name]
                    file.write('%16.5f  %3s %3s %3s   %20.16f  %20.16f  %20.16f  %20.16f %5d %5d %5d\n' %
                               (vv[0][index], 'G', self.sat.name, key.name[-1], vv[1][index],
                                vv[2][index], vv[3][index], vv[4][index], vv[5][index], vv[6][index], vv[7][index]
                                ))

        return sca


def demo1():
    from pysrc.GetInstrument import GetInstrument_L1A, GetInstrument_L1B, Mission, SatID

    date = '2018-12-02'
    sat = SatID.C
    SCA1Ato1B.ConfigSCF2SRF()

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
    # IMU1A = L1A.getIMU(SatID.C)
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    # L1B.getIMU(SatID.C)
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)

    SCA = SCA1Ato1B(satID=SatID.C, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)
    SCA.configInterp(interpOption=SCAinterOption.TwoHZ)
    SCA.produce_combine_last(mission=Mission.GRACE_FO, isCombined=True)

    pass


if __name__ == '__main__':
    demo1()
