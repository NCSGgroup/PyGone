"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/4 12:02
@Description:
"""
import calendar
import datetime
import gzip, tarfile, os
import math

import numpy as np

class GeoMathKit:
    def __init__(self):
        pass

    @staticmethod
    def dayListByDay(begin, end):
        """
        get the date of every day between the given 'begin' day and 'end' day

        :param begin: year, month, day. '2009-01-01'
        :param end: year,month,day. '2010-01-01'
        :return:
        """

        daylist = []
        begin_date = datetime.datetime.strptime(begin, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")

        while begin_date <= end_date:
            date_str = begin_date
            daylist.append(date_str)
            begin_date += datetime.timedelta(days=1)

        return daylist

    @staticmethod
    def un_gz(file_name):

        # aquire the filename and remove the postfix
        f_name = file_name.replace(".gz", "")
        # start uncompress
        g_file = gzip.GzipFile(file_name)
        # read uncompressed files and write down a copy without postfix
        open(f_name, "wb+").write(g_file.read())
        g_file.close()

    @staticmethod
    def un_tar(file_name):
        """
        Also applicable for .tgz
        :param file_name:
        :return:
        """
        tar = tarfile.open(file_name)
        names = tar.getnames()
        if os.path.isdir(file_name + '_files'):
            pass
        else:
            os.mkdir(file_name + '_files')

        for name in names:
            tar.extract(name, file_name + '_files')
        tar.close()

    @staticmethod
    def ReadEndOfHeader(file_name, end_flag):
        """
        return the number of 'end of header'
        :param file_name:
        :param end_flag:
        :return:
        """
        with open(file_name) as f:
            count = 0
            while True:
                count += 1
                aa = f.readline()
                if end_flag in aa:
                    break
        return count

    @staticmethod
    def CalcPolarAngles(xyz: np.ndarray):
        """
        see IERS2010 the annotation of Eq. 6.6 the detailed definition of each parameter.
        Calculate polar components from XYZ frame
        :param xyz: pos in itrf frame
        :return: geodetic coordinate: lon , lat and R
        """

        # Length of projection in x-y-plane:
        rhoSqr = np.linalg.norm(xyz[0:2])
        # Norm of vector
        r = np.linalg.norm(xyz[0:])
        # longitude
        if (xyz[0] == 0) and (xyz[1] == 0):
            phi = 0
        else:
            phi = np.arctan2(xyz[1], xyz[0])

        if phi < 0:
            phi += 2 * np.pi

        # latitude
        if (xyz[2] == 0) and (rhoSqr == 0):
            theta = 0
        else:
            theta = np.arctan2(xyz[2], rhoSqr)

        return phi, theta, r

if __name__ == '__main__':
    GeoMathKit.un_tar('../data/GRACE_FO/RL04/L1A/2021-01/gracefo_1A_2021-01-03_RL04.ascii.noLRI.tgz')
    pass