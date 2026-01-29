import os

import numpy as np
import datetime
import calendar
def getUppairMean(BeginDate='2003-01-01',EndDate='2004-01-01'):
    Begyear = int(BeginDate.split('-')[0])
    Endyear = int(EndDate.split('-')[0])
    Begmonth = int(BeginDate.split('-')[1])
    Endmonth = int(EndDate.split('-')[1])
    # Begday = int(BeginDate.split('-')[2])
    # Endday = int(EndDate.split('-')[2])
    # d1 = datetime.datetime(Begyear,Begmonth,Begday)
    # d2 = datetime.datetime(Endyear,Endmonth,Endday)
    # d = d2-d1
    # print(d.days)
    MCsum =0
    MSsum =0
    DCsum =0
    DSsum =0
    Csum = 0
    Ssum = 0
    num = 0
    for year in range(Begyear,Endyear+1):
        for month in range(Begmonth,Endmonth+1):
            for day in range(1,int(calendar.monthrange(year,month)[1])+1):
                num += 1
                count = 0
                Index = []
                uppairC = []
                uppairS = []
                with open('../result/upperair/{year}-{month}/{year}-{month}-{day}.asc'.format(year=year,month=str(month).rjust(2,'0'),day=str(day).rjust(2,'0')), 'r') as f:
                    for i in f.readlines():
                        a = i.split()
                        if a[0] == 'DATA' and a[-1] == 'atm':
                            Index.append(count + 1)
                            count += 1
                            continue
                        else:
                            count += 1
                    Index = np.array(Index)
                    # print(Index.dtype)
                    # print(Index[0])
                    f.seek(0)
                    for j in np.arange(len(Index)):
                        # print(j)
                        # print(type(j))
                        if j <= 2:
                            for i in f.readlines()[Index[j]:Index[j + 1] - 1]:
                                # print(i)
                                a = i.split()
                                uppairC.append(a[2])
                                uppairS.append(a[3])
                                # print(len(uppairC))
                                f.seek(0)

                        else:
                            # print(Index[j])
                            for i in f.readlines()[Index[j]:]:
                                # print(i)
                                a = i.split()
                                uppairC.append(a[2])
                                uppairS.append(a[3])
                                f.seek(0)
                    uppairC = np.array(uppairC).astype(np.float)
                    uppairS = np.array(uppairS).astype(np.float)
                # print(len(uppairC))
                # print(uppairC[0])
                DCsum += uppairC
                DSsum += uppairS
            # print('Csum:',len(DCsum))
            # print(DCsum[0])
            MCsum += DCsum
            MSsum += DSsum
        # print('Ssum:',len(MSsum))
        # print(MCsum[0])
        Csum += MCsum
        Ssum += MSsum
    # print(len(Csum))
    # print(Csum[0])

    l = len(Csum)
    n = 4
    step = int(l/n)
    C = [Csum[i:i+step] for i in np.arange(0,l,step)]
    S = [Ssum[i:i+step] for i in np.arange(0,l,step)]

    C = C[0]+C[1]+C[2]+C[3]
    S = S[0]+S[1]+S[2]+S[3]
    Cmean = C/(num*4)
    Smean = S/(num*4)
    # print(Cmean[0])

    return Cmean,Smean

def getMeanUpperFile(BeginDate='2002-01-01',EndDate='2002-01-02',Nmax=100):
    Begyear = int(BeginDate.split('-')[0])
    Endyear = int(EndDate.split('-')[0])
    Begmonth = int(BeginDate.split('-')[1])
    Endmonth = int(EndDate.split('-')[1])
    for year in range(Begyear, Endyear + 1):
        for month in range(Begmonth, Endmonth + 1):
            # for day in range(1, 2):
            for day in range(1,int(calendar.monthrange(year,month)[1])+1):
                path = '../result/meanupper/{year}-{month:02}/'.format(year=year, month=month)
                if not os.path.exists(path):
                    os.makedirs(path)
                meanC0 = 0
                meanS0 = 0
                meanC1 = 0
                meanS1 = 0
                meanC2 = 0
                meanS2 = 0
                meanC3 = 0
                meanS3 = 0
                C0, S0, C1, S1, C2, S2, C3, S3 = [], [], [], [], [], [], [], []
                with open('../result/upperair/{year}-{month}/{year}-{month}-{day}.asc'.format(year=year,
                                                                                              month=str(month).rjust(2,
                                                                                                                     '0'),
                                                                                              day=str(day).rjust(2,
                                                                                                                 '0')),
                          'r') as f:
                    for i in f.readlines()[GetIndex()[0]:GetIndex()[1] - 1]:
                        a = i.split()
                        C0.append(a[2])
                        S0.append(a[3])

                    C0 = np.array(C0).astype(np.float)
                    S0 = np.array(S0).astype(np.float)
                    C0 = C0 - getUppairMean()[0]
                    S0 = S0 - getUppairMean()[1]
                    meanC0 += C0
                    meanS0 += S0
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[1]:GetIndex()[2] - 1]:
                        a = i.split()
                        C1.append(a[2])
                        S1.append(a[3])
                    C1 = np.array(C1).astype(np.float)
                    S1 = np.array(S1).astype(np.float)
                    C1 = C1 - getUppairMean()[0]
                    S1 = S1 - getUppairMean()[1]
                    meanC1 += C1
                    meanS1 += S1
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[2]:GetIndex()[3] - 1]:
                        a = i.split()
                        C2.append(a[2])
                        S2.append(a[3])
                    C2 = np.array(C2).astype(np.float)
                    S2 = np.array(S2).astype(np.float)
                    C2 = C2 - getUppairMean()[0]
                    S2 = S2 - getUppairMean()[1]
                    meanC2 += C2
                    meanS2 += S2
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[3]:]:
                        a = i.split()
                        C3.append(a[2])
                        S3.append(a[3])
                    C3 = np.array(C3).astype(np.float)
                    S3 = np.array(S3).astype(np.float)
                    C3 = C3 - getUppairMean()[0]
                    S3 = S3 - getUppairMean()[1]
                    meanC3 += C3
                    meanS3 += S3
                    with open(path + '{year}-{month:02}-{day:02}.asc'.format(year=year, month=month, day=day),
                              'w') as file:
                        file.write('Substract Mean of Upperair \n')
                        file.write('END OF HEADER \n')
                        file.write(
                            'DATA SET 1:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 00:00:00 OF TYPE atm\n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C0[count]).rjust(30) + str(S0[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 2:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 06:00:00 OF TYPE atm \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C1[count]).rjust(30) + str(S1[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 3:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 12:00:00 OF TYPE atm \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C2[count]).rjust(30) + str(S2[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 4:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 18:00:00 OF TYPE atm \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C3[count]).rjust(30) + str(S3[count]).rjust(
                                        30) + '\n')
                                count += 1
                print('{}-{}-{}-upperair finished'.format(year,month,day))


def getSurPreMean(BeginDate='2003-01-01',EndDate='2004-01-01'):
    Begyear = int(BeginDate.split('-')[0])
    Endyear = int(EndDate.split('-')[0])
    Begmonth = int(BeginDate.split('-')[1])
    Endmonth = int(EndDate.split('-')[1])
    # Begday = int(BeginDate.split('-')[2])
    # Endday = int(EndDate.split('-')[2])
    # d1 = datetime.datetime(Begyear,Begmonth,Begday)
    # d2 = datetime.datetime(Endyear,Endmonth,Endday)
    # d = d2-d1
    # print(d.days)
    MCsum =0
    MSsum =0
    DCsum =0
    DSsum =0
    Csum = 0
    Ssum = 0
    num = 0
    for year in range(Begyear,Endyear+1):
        for month in range(Begmonth,Endmonth+1):
            for day in range(1,int(calendar.monthrange(year,month)[1])+1):
                num += 1
                count = 0
                Index = []
                uppairC = []
                uppairS = []
                with open('../result/sp/{year}-{month}/{year}-{month}-{day}.asc'.format(year=year,month=str(month).rjust(2,'0'),day=str(day).rjust(2,'0')), 'r') as f:
                    for i in f.readlines():
                        a = i.split()
                        if a[0] == 'DATA' and a[-1] == 'atm':
                            Index.append(count + 1)
                            count += 1
                            continue
                        else:
                            count += 1
                    Index = np.array(Index)
                    # print(Index.dtype)
                    # print(Index[0])
                    f.seek(0)
                    for j in np.arange(len(Index)):
                        # print(j)
                        # print(type(j))
                        if j <= 2:
                            for i in f.readlines()[Index[j]:Index[j + 1] - 1]:
                                # print(i)
                                a = i.split()
                                uppairC.append(a[2])
                                uppairS.append(a[3])
                                # print(len(uppairC))
                                f.seek(0)

                        else:
                            # print(Index[j])
                            for i in f.readlines()[Index[j]:]:
                                # print(i)
                                a = i.split()
                                uppairC.append(a[2])
                                uppairS.append(a[3])
                                f.seek(0)
                    uppairC = np.array(uppairC).astype(np.float)
                    uppairS = np.array(uppairS).astype(np.float)
                # print(len(uppairC))
                # print(uppairC[0])
                DCsum += uppairC
                DSsum += uppairS
            # print('Csum:',len(DCsum))
            # print(DCsum[0])
            MCsum += DCsum
            MSsum += DSsum
        # print('Ssum:',len(MSsum))
        # print(MCsum[0])
        Csum += MCsum
        Ssum += MSsum
    # print(len(Csum))
    # print(Csum[0])

    l = len(Csum)
    n = 4
    step = int(l/n)
    C = [Csum[i:i+step] for i in np.arange(0,l,step)]
    S = [Ssum[i:i+step] for i in np.arange(0,l,step)]

    C = C[0]+C[1]+C[2]+C[3]
    S = S[0]+S[1]+S[2]+S[3]
    Cmean = C/(num*4)
    Smean = S/(num*4)
    # print(Cmean[0])

    return Cmean,Smean

def getMeanSurPreFile(BeginDate='2002-01-01',EndDate='2002-01-02',Nmax=100):
    Begyear = int(BeginDate.split('-')[0])
    Endyear = int(EndDate.split('-')[0])
    Begmonth = int(BeginDate.split('-')[1])
    Endmonth = int(EndDate.split('-')[1])
    for year in range(Begyear, Endyear + 1):
        for month in range(Begmonth, Endmonth + 1):
            # for day in range(1, 2):
            for day in range(1,int(calendar.monthrange(year,month)[1])+1):
                path = '../result/meansp/{year}-{month:02}/'.format(year=year, month=month)
                if not os.path.exists(path):
                    os.makedirs(path)
                meanC0 = 0
                meanS0 = 0
                meanC1 = 0
                meanS1 = 0
                meanC2 = 0
                meanS2 = 0
                meanC3 = 0
                meanS3 = 0
                C0, S0, C1, S1, C2, S2, C3, S3 = [], [], [], [], [], [], [], []
                with open('../result/sp/{year}-{month}/{year}-{month}-{day}.asc'.format(year=year,
                                                                                              month=str(month).rjust(2,
                                                                                                                     '0'),
                                                                                              day=str(day).rjust(2,
                                                                                                                 '0')),
                          'r') as f:
                    for i in f.readlines()[GetIndex()[0]:GetIndex()[1] - 1]:
                        a = i.split()
                        C0.append(a[2])
                        S0.append(a[3])

                    C0 = np.array(C0).astype(np.float)
                    S0 = np.array(S0).astype(np.float)
                    C0 = C0 - getSurPreMean()[0]
                    S0 = S0 - getSurPreMean()[1]
                    meanC0 += C0
                    meanS0 += S0
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[1]:GetIndex()[2] - 1]:
                        a = i.split()
                        C1.append(a[2])
                        S1.append(a[3])
                    C1 = np.array(C1).astype(np.float)
                    S1 = np.array(S1).astype(np.float)
                    C1 = C1 - getSurPreMean()[0]
                    S1 = S1 - getSurPreMean()[1]
                    meanC1 += C1
                    meanS1 += S1
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[2]:GetIndex()[3] - 1]:
                        a = i.split()
                        C2.append(a[2])
                        S2.append(a[3])
                    C2 = np.array(C2).astype(np.float)
                    S2 = np.array(S2).astype(np.float)
                    C2 = C2 - getSurPreMean()[0]
                    S2 = S2 - getSurPreMean()[1]
                    meanC2 += C2
                    meanS2 += S2
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[3]:]:
                        a = i.split()
                        C3.append(a[2])
                        S3.append(a[3])
                    C3 = np.array(C3).astype(np.float)
                    S3 = np.array(S3).astype(np.float)
                    C3 = C3 - getSurPreMean()[0]
                    S3 = S3 - getSurPreMean()[1]
                    meanC3 += C3
                    meanS3 += S3
                    with open(path + '{year}-{month:02}-{day:02}.asc'.format(year=year, month=month, day=day),
                              'w') as file:
                        file.write('Substract Mean of Surface Pressure \n')
                        file.write('END OF HEADER \n')
                        file.write(
                            'DATA SET 1:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 00:00:00 OF TYPE atm\n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C0[count]).rjust(30) + str(S0[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 2:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 06:00:00 OF TYPE atm \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C1[count]).rjust(30) + str(S1[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 3:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 12:00:00 OF TYPE atm \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C2[count]).rjust(30) + str(S2[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 4:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 18:00:00 OF TYPE atm \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C3[count]).rjust(30) + str(S3[count]).rjust(
                                        30) + '\n')
                                count += 1

                print('{}-{}-{}-surface pressure finished'.format(year, month, day))

def getMeanValue(BeginDate='2002-01-01',EndDate='2002-01-31'):
    Cuppair = getUppairMean(BeginDate,EndDate)[0]
    Suppair = getUppairMean(BeginDate,EndDate)[1]
    Csp = getSurPreMean(BeginDate,EndDate)[0]
    Ssp = getSurPreMean(BeginDate,EndDate)[0]
    Cmean = Cuppair+Csp
    Smean = Suppair+Ssp
    print(Cmean.shape)

    return Cmean,Smean

def getSubMeanFile(BeginDate='2002-01-01',EndDate='2002-01-02',Nmax=100):
    '''has been trash already! forbiten to using!'''
    Begyear = int(BeginDate.split('-')[0])
    Endyear = int(EndDate.split('-')[0])
    Begmonth = int(BeginDate.split('-')[1])
    Endmonth = int(EndDate.split('-')[1])
    MC0,MS0,MC1,MS1,MC2,MS2,MC3,MS3 = [],[],[],[],[],[],[],[]

    for year in range(Begyear,Endyear+1):
        for month in range(Begmonth,Endmonth+1):
            for day in range(1,2):
            # for day in range(1,int(calendar.monthrange(year,month)[1])+1):
                meanC0=0
                meanS0=0
                meanC1=0
                meanS1=0
                meanC2=0
                meanS2=0
                meanC3=0
                meanS3=0
                path='../result/meanupper/{year}-{month:02}/'.format(year=year,month=month)
                if not os.path.exists(path):
                    os.makedirs(path)
                C = []
                S = []
                C1,S1,C2,S2,C3,S3 = [],[],[],[],[],[]
                with open('../result/upperair/{year}-{month}/{year}-{month}-{day}.asc'.format(year=year,month=str(month).rjust(2,'0'),day=str(day).rjust(2, '0')),'r') as f:
                    for i in f.readlines()[GetIndex()[0]:GetIndex()[1]-1]:
                        a = i.split()
                        C.append(a[2])
                        # print(C)
                        S.append(a[3])

                    C = np.array(C).astype(np.float)
                    S = np.array(S).astype(np.float)
                    C0 = C - getUppairMean()[0]
                    S0 = S - getUppairMean()[1]
                    meanC0 += C0
                    meanS0 += S0
                    # with open(path+'{year}-{month:02}-{day:02}-00.asc'.format(year=year,month=month,day=day), 'w') as file:
                    #     file.write('END OF HEADER \n')
                    #     file.write('DATA SET :   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 00:00:00 OF TYPE atm(remove average) \n'.format(year=year,month=month,day=day))
                    #     count = 0
                    #     for i in range(0,Nmax+1):
                    #         for j in range(0,i+1):
                    #             file.write(str(i).rjust(5)+str(j).rjust(5)+str(C0[count]).rjust(30)+str(S0[count]).rjust(30)+'\n')
                    #             count += 1
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[1]:GetIndex()[2]-1]:
                        a = i.split()
                        C1.append(a[2])
                        S1.append(a[3])
                    C1 = np.array(C1).astype(np.float)
                    S1 = np.array(S1).astype(np.float)
                    C1 = C1 - getUppairMean()[0]
                    S1 = S1 - getUppairMean()[1]
                    meanC1 += C1
                    meanS1 += S1
                    # with open(path+'{year}-{month:02}-{day:02}-06.asc'.format(year=year,month=month,day=day), 'w') as file:
                    #     file.write('END OF HEADER \n')
                    #     file.write('DATA SET :   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 06:00:00 OF TYPE atm(remove average) \n'.format(year=year,month=month,day=day))
                    #     count = 0
                    #     for i in range(0,Nmax+1):
                    #         for j in range(0,i+1):
                    #             file.write(str(i).rjust(5)+str(j).rjust(5)+str(C1[count]).rjust(30)+str(S1[count]).rjust(30)+'\n')
                    #             count += 1

                    f.seek(0)
                    for i in f.readlines()[GetIndex()[2]:GetIndex()[3]-1]:
                        a = i.split()
                        C2.append(a[2])
                        S2.append(a[3])
                    C2 = np.array(C2).astype(np.float)
                    S2 = np.array(S2).astype(np.float)
                    C2 = C2 - getUppairMean()[0]
                    S2 = S2 - getUppairMean()[1]
                    meanC2 += C2
                    meanS2 += S2
                    # with open(path+'{year}-{month:02}-{day:02}-12.asc'.format(year=year,month=month,day=day), 'w') as file:
                    #     file.write('END OF HEADER \n')
                    #     file.write('DATA SET :   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 12:00:00 OF TYPE atm(remove average) \n'.format(year=year,month=month,day=day))
                    #     count = 0
                    #     for i in range(0,Nmax+1):
                    #         for j in range(0,i+1):
                    #             file.write(str(i).rjust(5)+str(j).rjust(5)+str(C2[count]).rjust(30)+str(S2[count]).rjust(30)+'\n')
                    #             count += 1
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[3]:]:
                        a = i.split()
                        C3.append(a[2])
                        S3.append(a[3])
                    C3 = np.array(C3).astype(np.float)
                    S3 = np.array(S3).astype(np.float)
                    C3 = C3 - getUppairMean()[0]
                    S3 = S3 - getUppairMean()[1]
                    meanC3 += C3
                    meanS3 += S3
                    # with open(path + '{year}-{month:02}-{day:02}-18.asc'.format(year=year, month=month, day=day),
                    #           'w') as file:
                    #     file.write('END OF HEADER \n')
                    #     file.write(
                    #         'DATA SET :   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 18:00:00 OF TYPE atm(remove average) \n'.format(
                    #             year=year, month=month, day=day))
                    #     count = 0
                    #     for i in range(0, Nmax+1):
                    #         for j in range(0, i + 1):
                    #             file.write(
                    #                 str(i).rjust(5) + str(j).rjust(5) + str(C3[count]).rjust(30) + str(S3[count]).rjust(
                    #                     30) + '\n')
                    #             count += 1
                    with open(path + '{year}-{month:02}-{day:02}.asc'.format(year=year,month=month,day=day),'w') as file:
                        file.write('END OF HEADER \n')
                        file.write('DATA SET 1:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 00:00:00 OF TYPE atm(remove average of uppair) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C0[count]).rjust(30) + str(S0[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 2:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 06:00:00 OF TYPE atm(remove average of upperair) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C1[count]).rjust(30) + str(S1[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 3:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 12:00:00 OF TYPE atm(remove average of upperair) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C2[count]).rjust(30) + str(S2[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 4:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 18:00:00 OF TYPE atm(remove average of upperair) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C3[count]).rjust(30) + str(S3[count]).rjust(
                                        30) + '\n')
                                count += 1

                print(meanC0.shape)
                print(type(meanC0))


                path = '../result/meansp/{year}-{month:02}/'.format(year=year, month=month)
                if not os.path.exists(path):
                    os.makedirs(path)
                C = []
                S = []
                C1, S1, C2, S2, C3, S3 = [], [], [], [], [], []
                with open('../result/sp/{year}-{month}/{year}-{month}-{day}.asc'.format(year=year,
                                                                                              month=str(month).rjust(2,
                                                                                                                     '0'),
                                                                                              day=str(day).rjust(2,
                                                                                                                 '0')),
                          'r') as f:
                    for i in f.readlines()[GetIndex()[0]:GetIndex()[1] - 1]:
                        a = i.split()
                        C.append(a[2])
                        S.append(a[3])

                    C = np.array(C).astype(np.float)
                    S = np.array(S).astype(np.float)
                    C0 = C - getUppairMean()[0]
                    S0 = S - getUppairMean()[1]
                    meanC0 += C0
                    meanS0 += S0
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[1]:GetIndex()[2] - 1]:
                        a = i.split()
                        C1.append(a[2])
                        S1.append(a[3])
                    C1 = np.array(C1).astype(np.float)
                    S1 = np.array(S1).astype(np.float)
                    C1 = C1 - getUppairMean()[0]
                    S1 = S1 - getUppairMean()[1]
                    meanC1 += C1
                    meanS1 += S1
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[2]:GetIndex()[3] - 1]:
                        a = i.split()
                        C2.append(a[2])
                        S2.append(a[3])
                    C2 = np.array(C2).astype(np.float)
                    S2 = np.array(S2).astype(np.float)
                    C2 = C2 - getUppairMean()[0]
                    S2 = S2 - getUppairMean()[1]
                    meanC2 += C2
                    meanS2 += S2
                    f.seek(0)
                    for i in f.readlines()[GetIndex()[3]:]:
                        a = i.split()
                        C3.append(a[2])
                        S3.append(a[3])
                    C3 = np.array(C3).astype(np.float)
                    S3 = np.array(S3).astype(np.float)
                    C3 = C3 - getUppairMean()[0]
                    S3 = S3 - getUppairMean()[1]
                    meanC3 += C3
                    meanS3 += S3

                    with open(path + '{year}-{month:02}-{day:02}.asc'.format(year=year, month=month, day=day),
                              'w') as file:
                        file.write('END OF HEADER \n')
                        file.write(
                            'DATA SET 1:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 00:00:00 OF TYPE atm(remove average of mean sp) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C0[count]).rjust(30) + str(S0[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 2:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 06:00:00 OF TYPE atm(remove average of mean sp) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C1[count]).rjust(30) + str(S1[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 3:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 12:00:00 OF TYPE atm(remove average of mean sp) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C2[count]).rjust(30) + str(S2[count]).rjust(
                                        30) + '\n')
                                count += 1
                        file.write(
                            'DATA SET 4:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 18:00:00 OF TYPE atm(remove average of mean sp) \n'.format(
                                year=year, month=month, day=day))
                        count = 0
                        for i in range(0, Nmax + 1):
                            for j in range(0, i + 1):
                                file.write(
                                    str(i).rjust(5) + str(j).rjust(5) + str(C3[count]).rjust(30) + str(S3[count]).rjust(
                                        30) + '\n')
                                count += 1
            MC0.append(meanC0)
            MC1.append(meanC1)
            MC2.append(meanC2)
            MC3.append(meanC3)
            MS0.append(meanS0)
            MS1.append(meanS1)
            MS2.append(meanS2)
            MS3.append(meanS3)
    MC0 = np.array(MC0).astype(np.float)
    MC1 = np.array(MC1).astype(np.float)
    MC2 = np.array(MC2).astype(np.float)
    MC3 = np.array(MC3).astype(np.float)
    MS0 = np.array(MS0).astype(np.float)
    MS1 = np.array(MS1).astype(np.float)
    MS2 = np.array(MS2).astype(np.float)
    MS3 = np.array(MS3).astype(np.float)

    return MC0,MC1,MC2,MC3,MS0,MS1,MS2,MS3

def getAODfile(BeginDate='2002-01-01',EndDate='2002-01-02',Nmax=100):
    Begyear = int(BeginDate.split('-')[0])
    Endyear = int(EndDate.split('-')[0])
    Begmonth = int(BeginDate.split('-')[1])
    Endmonth = int(EndDate.split('-')[1])
    for year in range(Begyear, Endyear + 1):
        for month in range(Begmonth, Endmonth + 1):
            for day in range(1, 2):
            # for day in range(1,int(calendar.monthrange(year,month)[1])+1):
                Csp,Ssp = [],[]
                Cupper,Supper = [],[]
                with open('../result/meansp/{year}-{month:02}/{year}-{month:02}-{day:02}.asc'.format(year=year,month=month,day=day),'r') as f:
                    for i in f.readlines()[getAODIndex()[0]:getAODIndex()[1]-1]:
                        a = i.split()
                        Csp.append(a[2])
                        Ssp.append(a[3])
                    f.seek(0)
                    for i in f.readlines()[getAODIndex()[1]:getAODIndex()[2]-1]:
                        a = i.split()
                        Csp.append(a[2])
                        Ssp.append(a[3])
                    f.seek(0)
                    for i in f.readlines()[getAODIndex()[2]:getAODIndex()[3]-1]:
                        a = i.split()
                        Csp.append(a[2])
                        Ssp.append(a[3])
                    f.seek(0)
                    for i in f.readlines()[getAODIndex()[3]:]:
                        a = i.split()
                        Csp.append(a[2])
                        Ssp.append(a[3])
                Csp = np.array(Csp).astype(np.float)
                Ssp = np.array(Ssp).astype(np.float)


                with open('../result/meanupper/{year}-{month:02}/{year}-{month:02}-{day:02}.asc'.format(year=year,month=month,day=day),'r') as f:
                    for i in f.readlines()[getAODIndex()[0]:getAODIndex()[1]-1]:
                        a = i.split()
                        Cupper.append(a[2])
                        Supper.append(a[3])
                    f.seek(0)
                    for i in f.readlines()[getAODIndex()[1]:getAODIndex()[2]-1]:
                        a = i.split()
                        Cupper.append(a[2])
                        Supper.append(a[3])
                        # print(Cupper)
                    f.seek(0)
                    for i in f.readlines()[getAODIndex()[2]:getAODIndex()[3]-1]:
                        a = i.split()
                        Cupper.append(a[2])
                        Supper.append(a[3])
                    f.seek(0)
                    for i in f.readlines()[getAODIndex()[3]:]:
                        a = i.split()
                        Cupper.append(a[2])
                        Supper.append(a[3])
                Cupper = np.array(Cupper).astype(np.float)
                Supper = np.array(Supper).astype(np.float)
                print('Csp,shape:', Csp.shape)

                print('Cupper,shape:',Cupper.shape)
                C = Csp+Cupper
                S = Ssp+Supper
                path = '../result/product/{year}-{month:02}/'.format(year=year,month=month)
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(path+'{year}-{month:02}-{day:02}.asc'.format(year=year,month=month,day=day),'w') as f:
                    f.write('Remove Averager of Product \n')
                    f.write('END OF HEADER \n')
                    f.write(
                        'DATA SET 1:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 00:00:00 OF TYPE atm\n'.format(
                            year=year, month=month, day=day))
                    count = 0
                    for i in range(0, Nmax + 1):
                        for j in range(0, i + 1):
                            f.write(
                                str(i).rjust(5) + str(j).rjust(5) + str(C[count]).rjust(30) + str(S[count]).rjust(
                                    30) + '\n')
                            count += 1
                    f.write(
                        'DATA SET 2:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 06:00:00 OF TYPE atm \n'.format(
                            year=year, month=month, day=day))
                    for i in range(0, Nmax + 1):
                        for j in range(0, i + 1):
                            f.write(
                                str(i).rjust(5) + str(j).rjust(5) + str(C[count]).rjust(30) + str(S[count]).rjust(
                                    30) + '\n')
                            count += 1
                    f.write(
                        'DATA SET 3:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 12:00:00 OF TYPE atm \n'.format(
                            year=year, month=month, day=day))
                    for i in range(0, Nmax + 1):
                        for j in range(0, i + 1):
                            f.write(
                                str(i).rjust(5) + str(j).rjust(5) + str(C[count]).rjust(30) + str(S[count]).rjust(
                                    30) + '\n')
                            count += 1
                    f.write(
                        'DATA SET 4:   5151 COEFFICIENTS FOR {year}-{month:02}-{day:02} 18:00:00 OF TYPE atm \n'.format(
                            year=year, month=month, day=day))
                    for i in range(0, Nmax + 1):
                        for j in range(0, i + 1):
                            f.write(
                                str(i).rjust(5) + str(j).rjust(5) + str(C[count]).rjust(30) + str(S[count]).rjust(
                                    30) + '\n')
                            count += 1
                print('{year}-{month:02}-{day:02} product has produced!'.format(year=year,month=month,day=day))




    pass

def getAODIndex():
    '''get the number beiging location of the product,remove str messeage'''
    Index = []
    count = 0
    with open('../result/meansp/2002-01/2002-01-01.asc','r') as f:
        for i in f.readlines():
            a = i.split()
            if a[0] == 'DATA' and a[-1] == 'atm':
                count += 1
                Index.append(count)

            else:
                count += 1
        # Index = np.array(Index).astype(np.int)
        # print(Index)
        # f.seek(0)
        # for i in f.readlines()[Index[0]:Index[1]-1]:
        #     a = i.split()
        #     print(a[2])

    Index = np.array(Index).astype(np.int)
    print(Index)

    return Index


def GetIndex():
    '''get the start location of the C and S'''
    with open('../result/upperair/2002-01/2002-01-01.asc','r') as f:
        count = 0
        Index = []
        for i in f.readlines():
            a = i.split()
            if a[-1] == 'atm' and a[0] == 'DATA':
                count += 1
                Index.append(count)

            else:
                count += 1
        Index = np.array(Index).astype(np.int)
        # print(Index)
        # print(type(Index))
        return Index


def Test():
    a=getAODIndex()[0]
    b=getAODIndex()[1]
    with open('../result/meansp/2002-01/2002-01-01.asc','r') as f:
        for i in f.readlines()[a:b-1]:
            print(i)




if __name__ == '__main__':
    # getMeanUpperFile()
    # getMeanSurPreFile()
    # getAODIndex()
    getAODfile()
    # Test()