import numpy as np
import os

from Format import FormatWrite
from GeoMathKit import GeoMathKit
from LoadSH import AOD_GFZ
from Setting import AODtype
from HConfig import HConfig

def get_product(BeginDate, EndDate):
    Nmax = 100

    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]


    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    daylistout = ['2004-12-22', '2008-12-16', '2008-12-18', '2008-12-24', '2008-12-26',]

    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        y = date.split('-')[0]
        m = date.split('-')[1]

        if date in daylistout:
            continue
        else:

            print('---------Date: %s-------' % date)
            savepath = '../result/product/{}-{}/'.format(y,m)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            with open(savepath+'{}.asc'.format(date),'w') as f:
                f.write(HConfig().Message(Nmax=Nmax))
                for time in TimeEpoch:
                    print('\nComputing: %s' % time)
                    if time == '00:00:00':
                        message = 'DATA SET  0:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date,time)
                    elif time == '06:00:00':
                        message = 'DATA SET  1:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    elif time == '12:00:00':
                        message = 'DATA SET  2:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    elif time == '18:00:00':
                        message = 'DATA SET  3:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    else:
                        continue
                    f.write(message)
                    sp = AOD_GFZ().load('../result/sp').setType(AODtype.ATM)
                    upperair = AOD_GFZ().load('../result/upperair').setType(AODtype.ATM)
                    mean = HConfig().MeanCS()

                    spCS = sp.setTime(date,time).getCS(Nmax=Nmax)
                    spCS = np.array(spCS)
                    upperairCS = upperair.setTime(date,time).getCS(Nmax=Nmax)
                    upperairCS = np.array(upperairCS)
                    mean = np.array(mean)
                    product = spCS+upperairCS-mean
                    count = 0
                    for i in np.arange(0, Nmax + 1):
                        for j in np.arange(0, i + 1):
                            f.write(str(i).rjust(5) + str(j).rjust(5) + str(product[0,count]).rjust(28) + str(product[1,count]).rjust(
                                28) + '\n')
                            count += 1

def get_product1(BeginDate, EndDate):
    '''
    product1 is to get the product which mean field is to subtract 2006-2017
    :param BeginDate:
    :param EndDate:
    :return:
    '''
    Nmax = 100

    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]


    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    daylistout = ['2004-12-22', '2008-12-16', '2008-12-18', '2008-12-24', '2008-12-26',]

    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        y = date.split('-')[0]
        m = date.split('-')[1]

        if date in daylistout:
            continue
        else:

            print('---------Date: %s-------' % date)
            savepath = '../result/product1/{}-{}/'.format(y,m)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            with open(savepath+'{}.asc'.format(date),'w') as f:
                f.write(HConfig().Message(Nmax=Nmax))
                for time in TimeEpoch:
                    print('\nComputing: %s' % time)
                    if time == '00:00:00':
                        message = 'DATA SET  0:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date,time)
                    elif time == '06:00:00':
                        message = 'DATA SET  1:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    elif time == '12:00:00':
                        message = 'DATA SET  2:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    elif time == '18:00:00':
                        message = 'DATA SET  3:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    else:
                        continue
                    f.write(message)
                    sp = AOD_GFZ().load('../result/sp').setType(AODtype.ATM)
                    upperair = AOD_GFZ().load('../result/upperair').setType(AODtype.ATM)
                    mean = HConfig().MeanCS1()

                    spCS = sp.setTime(date,time).getCS(Nmax=Nmax)
                    spCS = np.array(spCS)
                    upperairCS = upperair.setTime(date,time).getCS(Nmax=Nmax)
                    upperairCS = np.array(upperairCS)
                    mean = np.array(mean)
                    product = spCS+upperairCS-mean
                    count = 0
                    for i in np.arange(0, Nmax + 1):
                        for j in np.arange(0, i + 1):
                            f.write(str(i).rjust(5) + str(j).rjust(5) + str(product[0,count]).rjust(28) + str(product[1,count]).rjust(
                                28) + '\n')
                            count += 1
def get_product2(BeginDate, EndDate):
    '''
    product1 is to get the product which mean field is to subtract 2009-2020
    :param BeginDate:
    :param EndDate:
    :return:
    '''
    Nmax = 100

    TimeEpoch = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]


    daylist = GeoMathKit.dayListByDay(BeginDate, EndDate)

    daylistout = ['2004-12-22', '2008-12-16', '2008-12-18', '2008-12-24', '2008-12-26',]

    for day in daylist:
        date = day.strftime("%Y-%m-%d")
        y = date.split('-')[0]
        m = date.split('-')[1]

        if date in daylistout:
            continue
        else:

            print('---------Date: %s-------' % date)
            savepath = '../result/product2/{}-{}/'.format(y,m)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            with open(savepath+'{}.asc'.format(date),'w') as f:
                f.write(HConfig().Message(Nmax=Nmax))
                for time in TimeEpoch:
                    print('\nComputing: %s' % time)
                    if time == '00:00:00':
                        message = 'DATA SET  0:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date,time)
                    elif time == '06:00:00':
                        message = 'DATA SET  1:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    elif time == '12:00:00':
                        message = 'DATA SET  2:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    elif time == '18:00:00':
                        message = 'DATA SET  3:   5151 COEFFICIENTS FOR {} {} OF TYPE atm\n'.format(date, time)
                    else:
                        continue
                    f.write(message)
                    sp = AOD_GFZ().load('../result/sp').setType(AODtype.ATM)
                    upperair = AOD_GFZ().load('../result/upperair').setType(AODtype.ATM)
                    mean = HConfig().MeanCS2()

                    spCS = sp.setTime(date,time).getCS(Nmax=Nmax)
                    spCS = np.array(spCS)
                    upperairCS = upperair.setTime(date,time).getCS(Nmax=Nmax)
                    upperairCS = np.array(upperairCS)
                    mean = np.array(mean)
                    product = spCS+upperairCS-mean
                    count = 0
                    for i in np.arange(0, Nmax + 1):
                        for j in np.arange(0, i + 1):
                            f.write(str(i).rjust(5) + str(j).rjust(5) + str(product[0,count]).rjust(28) + str(product[1,count]).rjust(
                                28) + '\n')
                            count += 1
if __name__ == '__main__':
    # get_product(BeginDate='2020-08-20',EndDate='2020-12-30')
    get_product1(BeginDate='2015-01-01',EndDate='2019-12-31')
    get_product2(BeginDate='2015-01-01',EndDate='2019-12-31')
