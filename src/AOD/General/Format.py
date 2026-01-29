"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/7/1 下午6:55
@Description:
"""
import os
import numpy as np

from GeoMathKit import GeoMathKit



class CnmSnm:

    def __init__(self, date: str, Nmax: int):
        self.Cnm = {}
        self.Snm = {}
        self.maxDegree = Nmax
        self.date = date
        self._other()
        pass

    def add(self, Cnm, Snm, epoch: str, date: str, attribute: str):
        assert date == self.date

        self.Cnm[epoch+'/'+attribute] = Cnm

        self.Snm[epoch+'/'+attribute] = Snm

        pass

    def _other(self):
        self.producer = 'HUST'
        self.product_type = 'Atmosphere Dealiasing'
        self.product = 'CRA-HUST'
        self.version = '01'
        self.author='Yang F'+','+'Zhang W.H'+','+'Wu Y'
        self.start_year =''
        self.end_year =''




class FormatWrite:

    def __init__(self):
        self.__fileDir = None
        self.__CS = None
        self.__fileFullPath = None
        self.orderFirst = True
        pass

    def setRootDir(self, fileDir):
        self.__fileDir = fileDir
        assert os.path.exists(fileDir)
        return self

    def setCS(self, CS: CnmSnm):
        self.__CS = CS
        res = CS.date.split('-')

        try:
            subdir = res[0] + '-' + res[1]
        except:
            return self

        subdir = os.path.join(self.__fileDir, subdir)

        if not os.path.exists(subdir):
            os.makedirs(subdir)

        self.__fileFullPath = subdir + os.sep + CS.date + '.asc'

        return self

    def AODstyle(self):
        with open(self.__fileFullPath, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', self.__CS.product_type))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', self.__CS.producer))
            file.write('%-31s%3s%-31s \n' % ('PRODUCT', ':',self.__CS.product))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', self.__CS.version))
            file.write('%-31s%3s%-31s \n' % ('MAXIMUM DEGREE', ':', self.__CS.maxDegree))
            file.write('%-31s%3s%-31s \n' % ('PRODUCT AUTHOR', ':',self.__CS.author ))
            # file.write()
            # file.write()

            file.write('END OF HEADER \n')

            keys = list(self.__CS.Cnm.keys())
            keys.sort()

            for key in keys:
                Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
                Nmax = self.__CS.maxDegree
                file.write('DATA SET %2i:   %s COEFFICIENTS FOR %s %s OF TYPE %s \n'
                           % (keys.index(key), int((Nmax + 2) * (Nmax + 1) / 2), self.__CS.date, key.split('/')[0],
                              key.split('/')[1].lower()))
                self._mainContent(Cnm, Snm, Nmax, file)

        pass

    def TideStyle(self, tide:str):
        fileFullPath = self.__fileDir + os.sep + 'ATM_'+tide+'.asc'

        with open(fileFullPath, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'Tidal constitutions'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', self.__CS.producer))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', self.__CS.version))
            file.write('%-31s%3s%-31s \n' % ('MAXIMUM DEGREE', ':', self.__CS.maxDegree))
            file.write('%-31s%3s%-31s \n' % ('PRODUCT AUTHOR', ':', self.__CS.author))
            # file.write('END OF HEADER \n')

            keys = list(self.__CS.Cnm.keys())
            keys.sort()

            for key in keys:
                Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
                Nmax = self.__CS.maxDegree
                file.write('DATA SET %2i:   %s COEFFICIENTS OF TYPE %s \n'
                           % (keys.index(key), int((Nmax + 2) * (Nmax + 1) / 2), key.split('/')[1].lower()))
                self._mainContent(Cnm, Snm, Nmax, file)

        pass

    def _mainContent(self, Cnm, Snm, Nmax, file):

        if np.ndim(Cnm) == 1:
            Cnm = GeoMathKit.CS_1dTo2d(Cnm)
            Snm = GeoMathKit.CS_1dTo2d(Snm)

        if self.orderFirst:
            for i in range(Nmax + 1):
                for j in range(i + 1):
                    file.write('%5i %5i  %+15.10E  %+15.10E\n' % (i, j, Cnm[i, j], Snm[i, j]))
        else:
            for j in range(Nmax + 1):
                for i in range(j, Nmax + 1):
                    file.write('%5i %5i  %15.10g  %15.10g\n' % (i, j, Cnm[i, j], Snm[i, j]))

        pass


def demo1():
    from pysrc.LoadSH import AOD_GFZ, AODtype
    ad = AOD_GFZ().load('../data/Products/RL05').setType(AODtype.ATM).setTime('2005-01-01', '12:00:00')
    C, S = ad.getCS(ad.maxDegree)

    cs = CnmSnm(date='2005-01-01', Nmax=5)

    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.GLO.name)
    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.OCN.name)
    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.OBA.name)
    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.ATM.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.ATM.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.GLO.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.ATM.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.OCN.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.OBA.name)

    fm = FormatWrite().setRootDir('../result/products/')
    fm.setCS(cs).AODstyle()
    pass


if __name__ == '__main__':
    demo1()
