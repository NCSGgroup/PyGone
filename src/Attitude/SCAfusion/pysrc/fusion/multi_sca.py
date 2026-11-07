from pysrc.core.settings import SCAID, SatID, Coord
from pysrc.core.geometry import GeoMathKit
import numpy as np
import quaternion
from tqdm import trange
import warnings
from pysrc.analysis.hfn_analysis import configReadEuler3
from pysrc.analysis.hfn_correlation import CorrelationAnalysis as ca
import datetime
import h5py

# -*- coding: utf-8 -*-
class SCAweighting:
    """
    This class is designed to determine the error level (weight) of three star cameras, so as to facilitate a
    further combination of star cameras before or in the Kalman filtering.

    Notice: in the current, only combination of SCAs is enabled. Function of weight determination will be done
    in the future work.

    Reference:
    1. Analyzing and monitoring GRACE-FO star camera performance in a changing environment, 2020, master thesis
    2. Optimal combination of quaternions from multiple star cameras, May 2003, L.Romans (JPL).
    """

    def __init__(self, SCA: dict, QSA: np.ndarray, A=None, date=None, sat=None):
        self._SCA = SCA
        self._QSA = QSA
        # self._Lambda = self.getLambda(A)
        self._Lambda=None
        self._date = date
        self.A=A
        self._time = None
        self._sat = sat
        if date is not None:
            self.HFNangle_x = self.ConfigHFNangle(coord=Coord.x)
            self.HFNangle_y = self.ConfigHFNangle(coord=Coord.y)
            self.HFNangle_z = self.ConfigHFNangle(coord=Coord.z)
        pass

    def ConfigHFNangle(self,coord:Coord):
        Date = datetime.datetime.strptime(self._date, '%Y-%m-%d')
        file='../HFN_Euler/%s/%s/HFNeuler-%s_%s_%s.h5'\
             %(Date.strftime('%Y-%m'),Date.strftime('%Y-%m-%d'),Date.strftime('%Y-%m-%d'),coord.name,self._sat.name)
        data={}
        with h5py.File(file, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    # 对于组，创建一个嵌套字典
                    nested_dict = {}
                    for nested_key in f[key].keys():
                        nested_dict[nested_key] = f[key][nested_key][()]
                    data[key] = nested_dict
                else:
                    # 对于数据集，直接读取值
                    data[key] = f[key][()]

        return data
        pass

    def combine_1t(self):
        """
        The combination considers the sigma of all SCAs as the same, totally following the Ref. 1
        This is specified for the SCA with only one time tag
        :return:
        """
        SCA = self._SCA

        q_new = []
        for i in trange(len(SCA[SCAID.No_1.name][0, :])):
            if SCA[SCAID.No_1.name][0, i] == SCA[SCAID.No_2.name][0, i] == SCA[SCAID.No_3.name][0, i]:
                pass
            else:
                raise Exception('SCA does not align well !')

            self._time=SCA[SCAID.No_1.name][0, i]

            q = self.__selectStrategy(quality1=SCA[SCAID.No_1.name][5, i],
                                      quality2=SCA[SCAID.No_2.name][5, i],
                                      quality3=SCA[SCAID.No_3.name][5, i],
                                      q1=SCA[SCAID.No_1.name][1:5, i],
                                      q2=SCA[SCAID.No_2.name][1:5, i],
                                      q3=SCA[SCAID.No_3.name][1:5, i])

            '''normalize'''
            q = q.normalized()
            q_new.append([SCA[SCAID.No_1.name][0, i], q.w, q.x, q.y, q.z])

        '''sign flip'''
        m = [q_new[0]]
        for i in range(len(q_new) - 1):
            vv = q_new[i + 1]
            if np.dot(q_new[i + 1][2:], m[-1][2:]) < 0:
                m.append([vv[0], -vv[1], -vv[2], -vv[3], -vv[4]])
            else:
                m.append(vv)

        return m

    def combine_2t_n(self):
        """
        The combination considers the sigma of all SCAs as the same, totally following the Ref. 1
        This is specified for the SCA with two time tags
        :return:
        """
        SCA = self._SCA

        q_new = []
        for i in trange(len(SCA[SCAID.No_1.name][0, :]),desc="Processing"):
            time1 = SCA[SCAID.No_1.name][0, i] + SCA[SCAID.No_1.name][1, i] * 1e-6
            time2 = SCA[SCAID.No_1.name][0, i] + SCA[SCAID.No_1.name][1, i] * 1e-6
            time3 = SCA[SCAID.No_1.name][0, i] + SCA[SCAID.No_1.name][1, i] * 1e-6
            if np.fabs(time1 - time2) <= 0.3 and np.fabs(time1 - time3) <= 0.3 and np.fabs(time2 - time3) <= 0.3:
                pass
            else:
                raise Exception('SCA does not align well !')

            self._time=time1

            q, n = self.__selectStrategy(quality1=SCA[SCAID.No_1.name][6, i],
                                      quality2=SCA[SCAID.No_2.name][6, i],
                                      quality3=SCA[SCAID.No_3.name][6, i],
                                      q1=SCA[SCAID.No_1.name][2:6, i],
                                      q2=SCA[SCAID.No_2.name][2:6, i],
                                      q3=SCA[SCAID.No_3.name][2:6, i])
            if q is None:
                continue
            '''normalize'''
            q = q.normalized()
            q_new.append([SCA[SCAID.No_1.name][0, i], SCA[SCAID.No_1.name][1, i], q.w, q.x, q.y, q.z, n])

        '''sign flip'''
        m = [q_new[0]]
        for i in range(len(q_new) - 1):
            vv = q_new[i + 1]
            if np.dot(q_new[i + 1][3:], m[-1][3:]) < 0:
                m.append([vv[0], vv[1], -vv[2], -vv[3], -vv[4], -vv[5], vv[6]])
            else:
                m.append(vv)

        return m

    def combine_2t(self):
        """
        The combination considers the sigma of all SCAs as the same, totally following the Ref. 1
        This is specified for the SCA with two time tags
        :return:
        """
        SCA = self._SCA

        q_new = []
        for i in trange(len(SCA[SCAID.No_1.name][0, :]),desc="Processing"):
            time1 = SCA[SCAID.No_1.name][0, i] + SCA[SCAID.No_1.name][1, i] * 1e-6
            time2 = SCA[SCAID.No_1.name][0, i] + SCA[SCAID.No_1.name][1, i] * 1e-6
            time3 = SCA[SCAID.No_1.name][0, i] + SCA[SCAID.No_1.name][1, i] * 1e-6
            if np.fabs(time1 - time2) <= 0.3 and np.fabs(time1 - time3) <= 0.3 and np.fabs(time2 - time3) <= 0.3:
                pass
            else:
                raise Exception('SCA does not align well !')

            self._time=time1

            q, n = self.__selectStrategy(quality1=SCA[SCAID.No_1.name][6, i],
                                      quality2=SCA[SCAID.No_2.name][6, i],
                                      quality3=SCA[SCAID.No_3.name][6, i],
                                      q1=SCA[SCAID.No_1.name][2:6, i],
                                      q2=SCA[SCAID.No_2.name][2:6, i],
                                      q3=SCA[SCAID.No_3.name][2:6, i])
            if q is None:
                continue
            '''normalize'''
            q = q.normalized()
            q_new.append([SCA[SCAID.No_1.name][0, i], SCA[SCAID.No_1.name][1, i], q.w, q.x, q.y, q.z])

        '''sign flip'''
        m = [q_new[0]]
        for i in range(len(q_new) - 1):
            vv = q_new[i + 1]
            if np.dot(q_new[i + 1][3:], m[-1][3:]) < 0:
                m.append([vv[0], vv[1], -vv[2], -vv[3], -vv[4], -vv[5]])
            else:
                m.append(vv)

        return m

    def combine_with_sigma(self):
        """
        The combination involves sigma of every individual SCA.
        :return:
        """

    def getLambda(self, a):
        """
        get Lambda. See ref. 1
        :param scaNo:
        a:a is a narray that contains the noise of three star-cameras, such as the first row containing No_ 1's
        :return:
        """

        k = 10
        sigma = 3.  # sigma is useless in fact.
        if a is not None:
            Lambda = {}
            for no in SCAID:
                QSA = self._QSA[int(no.name[-1]) - 1]
                R = quaternion.as_rotation_matrix(q=QSA)
                Cov_inv = np.diagflat([1 / a[no.value-1][0] ** 2, 1 / a[no.value-1][1] ** 2, 1 / a[no.value-1][2] ** 2])
                # TAKE CARE!!
                Lambda[no] = R.T @ Cov_inv @ R
        else:
            Lambda = {}
            for no in SCAID:
                a1 = np.interp(self._time, self.HFNangle_x[no.name][0], self.HFNangle_x[no.name][1])
                a2 = np.interp(self._time, self.HFNangle_y[no.name][0], self.HFNangle_y[no.name][1])
                a3 = np.interp(self._time, self.HFNangle_z[no.name][0], self.HFNangle_z[no.name][1])
                QSA = self._QSA[int(no.name[-1]) - 1]
                R = quaternion.as_rotation_matrix(q=QSA)
                Cov_inv = np.diagflat([1 / a1 ** 2, 1 / a2 ** 2, 1 / a3 ** 2])
                # TAKE CARE!!
                Lambda[no] = R.T @ Cov_inv @ R

        return Lambda

    def __selectStrategy(self, quality1, quality2, quality3, q1, q2, q3):

        quality1 = int(quality1)
        quality2 = int(quality2)
        quality3 = int(quality3)

        q = None
        n = None

        # if quality1 == quality2 == quality3 and quality1 <= 6:
        #     q = self.__combineThree(q1, q2, q3)
        #     return q

        if quality1 == quality2 == quality3 and quality1 > 6:
            warnings.warn('Discontinuity in SCA data.')
            return q, n

        # if quality1 < quality2 and quality1 < quality3 and quality1 <= 6:
        #     q = self.__onlyOne(q1)
        # elif quality2 < quality1 and quality2 < quality3 and quality2 <= 6:
        #     q = self.__onlyOne(q2)
        # elif quality3 < quality1 and quality3 < quality2 and quality3 <= 6:
        #     q = self.__onlyOne(q3)
        # elif quality1 == quality2 and quality1 <= 6:
        #     q = self.__combineTwo(sca1=SCAID.No_1, sca2=SCAID.No_2, q1=q1, q2=q2)
        # elif quality1 == quality3 and quality1 <= 6:
        #     q = self.__combineTwo(sca1=SCAID.No_1, sca2=SCAID.No_3, q1=q1, q2=q3)
        # elif quality2 == quality3 and quality2 <= 6:
        #     q = self.__combineTwo(sca1=SCAID.No_2, sca2=SCAID.No_3, q1=q2, q2=q3)
        # else:
        #     raise Exception('Discontinuity in SCA data')

        if quality1 <= 6 and quality2 <= 6 and quality3 <= 6:
            q = self.__combineThree(q1, q2, q3)
            n = 3
            return q, n

        if quality1 <= 6 and quality2 <= 6:
            q = self.__combineTwo(sca1=SCAID.No_1, sca2=SCAID.No_2, q1=q1, q2=q2)
            n = 2
            return q, n

        if quality2 <= 6 and quality3 <= 6:
            q = self.__combineTwo(sca1=SCAID.No_2, sca2=SCAID.No_3, q1=q2, q2=q3)
            n = 2
            return q, n

        if quality1 <= 6 and quality3 <= 6:
            q = self.__combineTwo(sca1=SCAID.No_1, sca2=SCAID.No_3, q1=q1, q2=q3)
            n = 2
            return q, n

        if quality1 <= 6:
            q = self.__onlyOne(q1)
            n=1
        elif quality2 <= 6:
            q = self.__onlyOne(q2)
            n=1
        elif quality3 <= 6:
            q = self.__onlyOne(q3)
            n = 1

        return q, n

    def __combineTwo(self, sca1: SCAID, sca2: SCAID, q1, q2):
        """

        :param sca1:
        :param sca2:
        :param q1: quaternion in SRF
        :param q2: quaternion in SRF
        :return:
        """
        '''sign flip'''
        if np.dot(q1[1:], q2[1:]) < 0:
            q2 = -q2

        qq1 = quaternion.as_quat_array(q1)
        qq2 = quaternion.as_quat_array(q2)
        Lambda = self.getLambda(self.A)
        Delta12 = qq1.inverse() * qq2
        Delta12 = quaternion.as_float_array(Delta12)
        # Del12 = Delta12[1:] / Delta12[0]
        Del12 = Delta12[1:]

        Lambdatot = Lambda[sca1] + Lambda[sca2]
        qv = np.linalg.inv(Lambdatot) @ (Lambda[sca2] @ Del12[:, None])

        return qq1 * quaternion.as_quat_array([1, qv[0, 0], qv[1, 0], qv[2, 0]])

    def __combineThree(self, q1, q2, q3):
        """

        :param q1: quaternion in SRF
        :param q2: quaternion in SRF
        :param q3: quaternion in SRF
        :return:
        """

        Lambda = self.getLambda(self.A)

        '''sign flip'''
        if np.dot(q1[1:], q2[1:]) < 0:
            q2 = -q2
        if np.dot(q1[1:], q3[1:]) < 0:
            q3 = -q3

        qq1 = quaternion.as_quat_array(q1)
        qq2 = quaternion.as_quat_array(q2)
        qq3 = quaternion.as_quat_array(q3)

        Delta12 = qq1.inverse() * qq2
        Delta13 = qq1.inverse() * qq3

        Delta12 = quaternion.as_float_array(Delta12)
        Delta13 = quaternion.as_float_array(Delta13)

        # Del12 = Delta12[1:] / Delta12[0]
        # Del13 = Delta13[1:] / Delta13[0]

        Del12 = Delta12[1:]
        Del13 = Delta13[1:]

        Lambdatot = Lambda[SCAID.No_1] + Lambda[SCAID.No_2] + Lambda[SCAID.No_3]

        qv = np.linalg.inv(Lambdatot) @ (Lambda[SCAID.No_2] @ Del12[:, None] + Lambda[SCAID.No_3] @ Del13[:, None])

        return qq1 * quaternion.as_quat_array([1, qv[0, 0], qv[1, 0], qv[2, 0]])

    def __onlyOne(self, q):
        return quaternion.as_quat_array(a=q)

# if __name__ == '__main__':
#     a = SCAweighting()
#     a.getLambda()