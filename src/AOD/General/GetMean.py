
import numpy as np
from GeoMathKit import GeoMathKit
import datetime
from Setting import FileType

class Mean:
    def __init__(self):
        self.__BeginTime = '2003-01-01'
        self.__EndTime = '2014-12-31'
        self.__Nmax = 180
        self.__path = '../result/mean180/'
        self.__TimeEpoch = ['00:00:00','06:00:00','12:00:00','18:00:00']
        pass

    def setTime(self,Begin,End):
        self.__BeginTime = Begin
        self.__EndTime = End
        return self

    def SetPath(self,path:str):

        self.__path = path
        return self

    def GetMeanCS(self):
        '''
        The most important part of this module.
        :return:
        '''
        daylist = GeoMathKit.dayListByDay(begin=self.__BeginTime,end=self.__EndTime)
        Cupper,Supper,Csp,Ssp = 0,0,0,0
        BDate = self.__BeginTime.split('-')
        EDate = self.__EndTime.split('-')
        d1 = datetime.datetime(int(BDate[0]), int(BDate[1]), int(BDate[2]))
        d2 = datetime.datetime(int(EDate[0]), int(EDate[1]), int(EDate[2]))
        D = (d2 - d1).days + 1
        daylistout = ['2004-12-22', '2008-12-16', '2008-12-18', '2008-12-24', '2008-12-26']
        for day in daylist:
            day = day.strftime('%Y-%m-%d')

            if day in daylistout:
                D -= 1
                continue
            else:
                SP = self.GetDayMean(kind=FileType.SP,day=day)
                UPPER = self.GetDayMean(kind=FileType.UPPER,day=day)
                Csp += SP[0]
                Ssp += SP[1]
                Cupper += UPPER[0]
                Supper += UPPER[1]
                print('{} has finished'.format(day))
        C = Csp+Cupper
        S = Ssp+Supper
        C = C/D
        S = S/D
        return C,S

    def GetMeanFile(self):
        count = 0
        CS = self.GetMeanCS()
        C = CS[0]
        S = CS[1]
        with open(self.File(kind=FileType.MEAN),'w') as f:
            f.write(self.HeadMessage())
            f.write('DATA SET  0:   5151 COEFFICIENTS FOR {}-{} mean OF TYPE atm\n'.format(self.__BeginTime,self.__EndTime))
            for i in np.arange(0,self.__Nmax+1):
                for j in np.arange(0,i+1):
                    f.write(
                        str(i).rjust(5) + str(j).rjust(5) + str(C[count]).rjust(28) + str(S[count]).rjust(28) + '\n')
                    count += 1



    def GetDayMean(self,kind:FileType=FileType.SP,day='2018-12-02'):
        C0,S0 = [],[]
        C1,S1 = [],[]
        C2,S2 = [],[]
        C3,S3 = [],[]
        with open(self.File(kind,day),'r') as f:
            for i in f.readlines()[self.Index(kind)[0]:self.Index(kind)[1]-1]:
                a = i.split()
                C0.append(a[2])
                S0.append(a[3])
            f.seek(0)
            for i in f.readlines()[self.Index(kind)[1]:self.Index(kind)[2]-1]:
                a = i.split()
                C1.append(a[2])
                S1.append(a[3])
            f.seek(0)
            for i in f.readlines()[self.Index(kind)[2]:self.Index(kind)[3]-1]:
                a = i.split()
                C2.append(a[2])
                S2.append(a[3])
            f.seek(0)
            for i in f.readlines()[self.Index(kind)[3]:]:
                a = i.split()
                C3.append(a[2])
                S3.append(a[3])
            C0 = np.array(C0).astype(np.float)
            S0 = np.array(S0).astype(np.float)
            C1 = np.array(C1).astype(np.float)
            S1 = np.array(S1).astype(np.float)
            C2 = np.array(C2).astype(np.float)
            S2 = np.array(S2).astype(np.float)
            C3 = np.array(C3).astype(np.float)
            S3 = np.array(S3).astype(np.float)

            C = (C0+C1+C2+C3)/4
            S = (S0+S1+S2+S3)/4
            return C,S


    def HeadMessage(self):
        nam = 'PRODUCT NAME'.ljust(30) + ':Atmosphere Dealiasing Mean Field\n'
        age = 'PRODUCER AGENCY'.ljust(30) + ':HUST\n'
        pro = 'PRODUCT'.ljust(30) + ':CRA-HUST\n'
        beg = 'BEGINDATE'.ljust(30) + ':{}\n'.format(self.__BeginTime)
        end = 'BEGINDATE'.ljust(30) + ':{}\n'.format(self.__EndTime)
        aut = 'PRODUCT AUTHOR'.ljust(30) + ':Yang F,Zhang W.H,Wu Y\n'
        tim = 'RPODUCE TIME'.ljust(30) + ':{}\n'.format(datetime.datetime.now())
        message = nam + age + pro + beg + end + aut + tim + 'END OF HEADER\n'

        return message




    def Index(self,kind:FileType=FileType.SP):
        count = 0
        Index = []
        with open(self.File(kind),'r') as f:
            for i in f.readlines():
                a = i.split()
                if a[-1] == 'atm':
                    count += 1
                    Index.append(count)
                else:
                    count += 1
        return Index


    def File(self,kind:FileType=FileType.SP,day='2018-12-02'):
        year = day.split('-')[0]
        month = day.split('-')[1]
        if kind == FileType.SP:
            file = '../result/sp180/{}-{}/{}.asc'.format(year,month,day)
        elif kind == FileType.UPPER:
            file = '../result/upperair180/{}-{}/{}.asc'.format(year,month,day)
        elif kind == FileType.MEAN:
            file = '../result/mean180/{}-{}.asc'.format(self.__BeginTime,self.__EndTime)
        return file


def demo():
    a = Mean()
    a.setTime(Begin='2018-12-01',End='2018-12-31')
    a.GetMeanFile()


if __name__ == '__main__':
    demo()
