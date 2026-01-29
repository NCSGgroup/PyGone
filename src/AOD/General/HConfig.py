import os
import numpy as np
import datetime



class HConfig:

    def __init__(self):
        # self.__Begin = '2003-01-01'
        # self.__End = '2014-12-31'
        self.__Begin = '2018-12-01'
        self.__End = '2018-12-31'
        self.__path = '../result/mean180/'
        pass

    def MeanCS(self):
        count = 0
        C,S = [],[]
        with open(self.__path+self.__Begin+'-'+self.__End+'.asc','r') as f:
            for i in f.readlines():
                a = i.split()
                if a[-1] == 'atm':
                    count += 1
                    break
                else:
                    count += 1
            f.seek(0)
            for i in f.readlines()[count:]:
                a = i.split()
                C.append(a[2])
                S.append(a[3])
        C = np.array(C).astype(np.float)
        S = np.array(S).astype(np.float)
        return C,S

    def MeanCS1(self):
        '''
        This mean field is from 2006-01-01 to 2017-12-31
        :return:
        '''
        count = 0
        C, S = [], []
        with open(self.__path + '2006-01-01' + '-' + '2017-12-31' + '.asc', 'r') as f:
            for i in f.readlines():
                a = i.split()
                if a[-1] == 'atm':
                    count += 1
                    break
                else:
                    count += 1
            f.seek(0)
            for i in f.readlines()[count:]:
                a = i.split()
                C.append(a[2])
                S.append(a[3])
        C = np.array(C).astype(np.float)
        S = np.array(S).astype(np.float)
        return C, S

    def MeanCS2(self):
        '''
        This mean field is from 2009-01-01 to 2020-12-30
        :return:
        '''
        count = 0
        C, S = [], []
        with open(self.__path + '2009-01-01' + '-' + '2020-12-30' + '.asc', 'r') as f:
            for i in f.readlines():
                a = i.split()
                if a[-1] == 'atm':
                    count += 1
                    break
                else:
                    count += 1
            f.seek(0)
            for i in f.readlines()[count:]:
                a = i.split()
                C.append(a[2])
                S.append(a[3])
        C = np.array(C).astype(np.float)
        S = np.array(S).astype(np.float)
        return C, S

    def Message(self,Nmax:int):
        '''
        HEADER MESSAGE WHEN WRITING
        :param day:
        :return:Message[0,1,2,3,4] = file/D1/D2/D3/D4
        '''
        nam = 'PRODUCT NAME'.ljust(30)+':Atmosphere Dealiasing\n'
        age = 'PRODUCER AGENCY'.ljust(30)+':HUST\n'
        pro = 'PRODUCT'.ljust(30)+':CRA-HUST\n'
        ver = 'SOFTWARE VERSION'.ljust(30)+':01\n'
        max = 'MAXIMUM DEGREE'.ljust(30)+':{}\n'.format(Nmax)
        aut = 'PRODUCT AUTHOR'.ljust(30)+':Yang F,Zhang W.H,Wu Y\n'
        tim = 'PRODUCE TIME'.ljust(30)+':{}\n'.format(datetime.datetime.now())
        file = nam+age+pro+ver+max+aut+tim+'END OF HEADER\n'
        return file






def demo():
    a = HConfig()
    # a.MeanCS()

if __name__ == '__main__':
    demo()