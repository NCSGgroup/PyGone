"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/9/8
@Description:
"""
import numpy as np
from scipy import signal
from Setting import Mission, SatID
from GeoMathKit import GeoMathKit
import os


class ButterWorth:

    def __init__(self, SCA, cutoff, fs):
        wn = 2 * cutoff / fs
        assert wn <= 1
        b, a = signal.butter(N=3, Wn=wn, btype='low', fs=fs)
        w = signal.filtfilt(b, a, SCA[1])
        x = signal.filtfilt(b, a, SCA[2])
        y = signal.filtfilt(b, a, SCA[3])
        z = signal.filtfilt(b, a, SCA[4])

        self.__new = np.array([w, x, y, z])
        pass

    def filteredSeries(self):
        return self.__new


class ReSample:

    def __init__(self, mission: Mission, date: str, sat: SatID, version='HUST01', fileDir=None):
        if fileDir is None:
            fileDir = '../result/product/%s/RL04/L1B/%s/' % (mission.name, date)
        else:
            fileDir = fileDir+'%s/'%date
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'SCA1B_%s_%s_Fusion_Resample_%s.txt' % (date, sat.name, version)
        filename2 = 'SCA1B_%s_%s_Fusion_%s.txt' % (date, sat.name, version)
        self.__output = fileDir + filename
        self.__input = fileDir + filename2
        self.sat = sat
        pass

    def run(self):

        print('\nLow-pass filter and resample is running ...')

        skip = GeoMathKit.ReadEndOfHeader(self.__input, 'END OF HEADER')
        SCA = np.loadtxt(self.__input, comments='#', skiprows=skip, usecols=(0, 2, 3, 4, 5), unpack=True)

        '''low-pass filter'''
        bw = ButterWorth(SCA=SCA, cutoff=1, fs=8)
        new = bw.filteredSeries()
        if np.any(np.isnan(new)):
            new = SCA[1:]

        '''resample'''
        time = SCA[0]
        timeResample = np.arange(int(time[0]), int(time[-1]) + 1)

        a, b = [], []
        k = 0
        for i, time1 in zip(range(len(timeResample)), timeResample):
            for j in range(k, len(time)):
                if np.fabs(time1 - time[j]) < 0.001:
                    a.append(j)
                    b.append(i)
                    k = j
                    break

        new_time = timeResample[b]
        new_data = new[:, a]

        with open(self.__output, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', '01'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Yang Fan'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'yfan_cge@hust.edu.cn'))
            file.write('END OF HEADER \n')

            for index in range(len(new_time)):
                vv = new_data[:, index]
                vt = new_time[index]
                file.write('%16.5f %3s   % 20.16f   % 20.16f   % 20.16f  % 20.16f\n' %
                           (vt, self.sat.name, vv[0], vv[1], vv[2], vv[3]))
        pass
