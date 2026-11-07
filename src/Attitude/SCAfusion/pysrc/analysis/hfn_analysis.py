import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from matplotlib.ticker import MultipleLocator

from pysrc.core.geometry import GeoMathKit
from pysrc.core.settings import SatID, SCAID, Mission, Coord
from enum import Enum
import datetime
from scipy import signal
import h5py
import json
import copy
from matplotlib import rcParams
from scipy.stats import norm


config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)
def configReadEuler(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28'):
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    Resx={}
    Resy={}
    Resz={}
    Res_temp={}
    Res_std={}
    outliers_index={}
    outliers_index_time={}

    for day in daylist:
        try:

            if day.strftime('%Y-%m-%d')==begindate:
                file='%s/%s-%s.csv'%(day.strftime('%Y-%m'),day.strftime('%Y-%m-%d'),sat.name)
                path='../Produce/'+file
                res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
                Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
                Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
                Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)

                # Calculate the average value as of a day
                for key in SCAID:
                    # res_filter1=Res_temp[key.name][1]
                    res_filter1 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][1])
                    # res_filter1=rollingWindow(data=Res_temp[key.name][[0,1],:],size=10)
                    resx1=np.vstack((Res_temp[key.name][0],res_filter1))
                    resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]
                    o1 = resx1
                    while len(resx1[1])!=len(resx2[1]):
                        resx1=resx2[:,np.abs(resx2[1])<3*np.std(resx2[1])]
                        resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]

                    outliers_index_x = np.isin(o1[-1], resx2[-1])
                    x_std =np.std(resx2[1])

                    # res_filter2 = Res_temp[key.name][2]
                    res_filter2 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][2])
                    # res_filter2 = rollingWindow(data=Res_temp[key.name][[0, 2], :],size=10)
                    resy1=np.vstack((Res_temp[key.name][0],res_filter2))
                    resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]
                    o2 = resy1
                    while len(resy1[1])!=len(resy2[1]):
                        resy1=resy2[:,np.abs(resy2[1])<3*np.std(resy2[1])]
                        resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]

                    outliers_index_y = np.isin(o2[-1], resy2[-1])
                    y_std =np.std(resy2[1])

                    # res_filter3 = Res_temp[key.name][3]
                    res_filter3 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][3])
                    # res_filter3 = rollingWindow(data=Res_temp[key.name][[0, 3], :],size=10)
                    resz1=np.vstack((Res_temp[key.name][0],res_filter3))
                    resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]
                    o3= resz1
                    while len(resz1[1])!=len(resz2[1]):
                        resz1=resz2[:,np.abs(resz2[1])<3*np.std(resz2[1])]
                        resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]

                    outliers_index_z = np.isin(o3[-1],resz2)
                    z_std =np.std(resz2[1])

                    print(x_std,y_std,z_std,'%s'%key.name)

                    Resx[key.name]=resx2
                    Resy[key.name]=resy2
                    Resz[key.name]=resz2
                    outliers_index[key.name]=np.vstack((outliers_index_x,outliers_index_y,outliers_index_z))
                    #obtaining the time points of outliers
                    index_time=np.all(outliers_index[key.name],axis=0)
                    outliers_index_time[key.name]=o1[0,~index_time]
                    Res_std[key.name]=np.array([x_std,y_std,z_std])


                print('%s data has processed'%(day.strftime('%Y-%m-%d')))


            else:
                file = '%s/%s-%s.csv' % (day.strftime('%Y-%m'), day.strftime('%Y-%m-%d'),sat.name)
                path = '../Produce/' + file
                res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
                Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
                Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
                Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)
                # Calculate the average value as of a day
                for key in SCAID:
                    res_filter1 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][1])
                    # res_filter1 = rollingWindow(data=Res_temp[key.name][[0, 1], :],size=20)
                    resx1 = np.vstack((Res_temp[key.name][0], res_filter1))
                    o1= resx1
                    resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]
                    while len(resx1[1]) != len(resx2[1]):
                        resx1 = resx2[:, np.abs(resx2[1]) < 3 * np.std(resx2[1])]
                        resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]

                    outliers_index_x = np.isin(o1[-1], resx2[-1])
                    x_std = np.std(resx2[1])

                    res_filter2 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][2])
                    # res_filter2 = rollingWindow(data=Res_temp[key.name][[0, 2], :],size=20)
                    resy1 = np.vstack((Res_temp[key.name][0], res_filter2))
                    resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]
                    o2 = resy1
                    while len(resy1[1]) != len(resy2[1]):
                        resy1 = resy2[:, np.abs(resy2[1]) < 3 * np.std(resy2[1])]
                        resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]

                    outliers_index_y = np.isin(o2[-1], resy2[-1])
                    y_std = np.std(resy2[1])

                    res_filter3 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][3])
                    # res_filter3 = rollingWindow(data=Res_temp[key.name][[0, 3], :], size=20)
                    resz1 = np.vstack((Res_temp[key.name][0], res_filter3))
                    resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]
                    o3= resz1
                    while len(resz1[1]) != len(resz2[1]):
                        resz1 = resz2[:, np.abs(resz2[1]) < 3 * np.std(resz2[1])]
                        resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]

                    outliers_index_z = np.isin(o3[-1],resz2)
                    z_std = np.std(resz2[1])

                    index=np.vstack((outliers_index_x,outliers_index_y,outliers_index_z))
                    #obtaining the time points of outliers
                    index_time=np.all(index,axis=0)
                    outliers_index_time_temp=o1[0,~index_time]


                    print(x_std,y_std,z_std,'%s'%key.name)



                    Resx[key.name]=np.hstack((Resx[key.name],resx2))
                    Resy[key.name]=np.hstack((Resy[key.name],resy2))
                    Resz[key.name]=np.hstack((Resz[key.name],resz2))
                    outliers_index[key.name]=np.hstack((outliers_index[key.name],index))
                    outliers_index_time[key.name]=np.hstack((outliers_index_time[key.name],outliers_index_time_temp))
                    Res_std[key.name]=np.vstack((Res_std[key.name],np.array([x_std,y_std,z_std])))



                print('%s data has read' % (day.strftime('%Y-%m-%d')))

        except:
            print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))



    return Resx,Resy,Resz,Res_std,outliers_index,outliers_index_time

    pass


def configReadEuler2(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28'):
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    Resx={}
    Resy={}
    Resz={}
    Res_temp={}
    Res_std={}
    outliers_index={}
    outliers_index_time={}
    days=[]

    for day in daylist:
        try:

            if day.strftime('%Y-%m-%d')==begindate:
                file='%s/%s-%s.csv'%(day.strftime('%Y-%m'),day.strftime('%Y-%m-%d'),sat.name)
                path='../Produce/'+file
                res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
                Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
                Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
                Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)

                # Calculate the average value as of a day
                for key in SCAID:
                    # res_filter1=Res_temp[key.name][1]
                    res_filter1 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][1])
                    # res_filter1=rollingWindow(data=Res_temp[key.name][[0,1],:],size=10)
                    resx1=np.vstack((Res_temp[key.name][0],res_filter1))
                    resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]
                    o1 = resx1
                    while len(resx1[1])!=len(resx2[1]):
                        resx1=resx2[:,np.abs(resx2[1])<3*np.std(resx2[1])]
                        resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]

                    outliers_index_x = np.isin(o1[-1], resx2[-1])
                    x_std =np.std(resx2[1])

                    # res_filter2 = Res_temp[key.name][2]
                    res_filter2 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][2])
                    # res_filter2 = rollingWindow(data=Res_temp[key.name][[0, 2], :],size=10)
                    resy1=np.vstack((Res_temp[key.name][0],res_filter2))
                    resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]
                    o2 = resy1
                    while len(resy1[1])!=len(resy2[1]):
                        resy1=resy2[:,np.abs(resy2[1])<3*np.std(resy2[1])]
                        resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]

                    outliers_index_y = np.isin(o2[-1], resy2[-1])
                    y_std =np.std(resy2[1])

                    # res_filter3 = Res_temp[key.name][3]
                    res_filter3 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][3])
                    # res_filter3 = rollingWindow(data=Res_temp[key.name][[0, 3], :],size=10)
                    resz1=np.vstack((Res_temp[key.name][0],res_filter3))
                    resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]
                    o3= resz1
                    while len(resz1[1])!=len(resz2[1]):
                        resz1=resz2[:,np.abs(resz2[1])<3*np.std(resz2[1])]
                        resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]

                    outliers_index_z = np.isin(o3[-1],resz2)
                    z_std =np.std(resz2[1])

                    print(x_std,y_std,z_std,'%s'%key.name)

                    Resx[key.name]=resx2
                    Resy[key.name]=resy2
                    Resz[key.name]=resz2
                    outliers_index[key.name]=np.vstack((outliers_index_x,outliers_index_y,outliers_index_z))
                    #obtaining the time points of outliers
                    index_time=np.all(outliers_index[key.name],axis=0)
                    outliers_index_time[key.name]=o1[0,~index_time]
                    Res_std[key.name]=np.array([x_std,y_std,z_std])


                print('%s data has processed'%(day.strftime('%Y-%m-%d')))


            else:
                file = '%s/%s-%s.csv' % (day.strftime('%Y-%m'), day.strftime('%Y-%m-%d'),sat.name)
                path = '../Produce/' + file
                res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
                Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
                Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
                Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)
                # Calculate the average value as of a day
                for key in SCAID:
                    res_filter1 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][1])
                    # res_filter1 = rollingWindow(data=Res_temp[key.name][[0, 1], :],size=20)
                    resx1 = np.vstack((Res_temp[key.name][0], res_filter1))
                    o1= resx1
                    resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]
                    while len(resx1[1]) != len(resx2[1]):
                        resx1 = resx2[:, np.abs(resx2[1]) < 3 * np.std(resx2[1])]
                        resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]

                    outliers_index_x = np.isin(o1[-1], resx2[-1])
                    x_std = np.std(resx2[1])

                    res_filter2 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][2])
                    # res_filter2 = rollingWindow(data=Res_temp[key.name][[0, 2], :],size=20)
                    resy1 = np.vstack((Res_temp[key.name][0], res_filter2))
                    resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]
                    o2 = resy1
                    while len(resy1[1]) != len(resy2[1]):
                        resy1 = resy2[:, np.abs(resy2[1]) < 3 * np.std(resy2[1])]
                        resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]

                    outliers_index_y = np.isin(o2[-1], resy2[-1])
                    y_std = np.std(resy2[1])

                    res_filter3 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][3])
                    # res_filter3 = rollingWindow(data=Res_temp[key.name][[0, 3], :], size=20)
                    resz1 = np.vstack((Res_temp[key.name][0], res_filter3))
                    resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]
                    o3= resz1
                    while len(resz1[1]) != len(resz2[1]):
                        resz1 = resz2[:, np.abs(resz2[1]) < 3 * np.std(resz2[1])]
                        resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]

                    outliers_index_z = np.isin(o3[-1],resz2)
                    z_std = np.std(resz2[1])

                    index=np.vstack((outliers_index_x,outliers_index_y,outliers_index_z))
                    #obtaining the time points of outliers
                    index_time=np.all(index,axis=0)
                    outliers_index_time_temp=o1[0,~index_time]


                    print(x_std,y_std,z_std,'%s'%key.name)



                    Resx[key.name]=np.hstack((Resx[key.name],resx2))
                    Resy[key.name]=np.hstack((Resy[key.name],resy2))
                    Resz[key.name]=np.hstack((Resz[key.name],resz2))
                    outliers_index[key.name]=np.hstack((outliers_index[key.name],index))
                    outliers_index_time[key.name]=np.hstack((outliers_index_time[key.name],outliers_index_time_temp))
                    Res_std[key.name]=np.vstack((Res_std[key.name],np.array([x_std,y_std,z_std])))



                print('%s data has read' % (day.strftime('%Y-%m-%d')))
            days.append(day)

        except:
            print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))



    return Resx,Resy,Resz,Res_std,outliers_index,outliers_index_time,days

    pass


def configReadEuler3(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28'):
    '''obtain the STD of HFN-angle'''
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    Res_temp={}
    Res_std={}
    days=[]

    for day in daylist:
        try:

            if day.strftime('%Y-%m-%d')==begindate:
                file='%s/%s-%s.csv'%(day.strftime('%Y-%m'),day.strftime('%Y-%m-%d'),sat.name)
                path='../Produce/'+file
                res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
                Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
                Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
                Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)

                # Calculate the average value as of a day
                for key in SCAID:
                    # res_filter1=Res_temp[key.name][1]
                    res_filter1 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][1])
                    # res_filter1=rollingWindow(data=Res_temp[key.name][[0,1],:],size=10)
                    resx1=np.vstack((Res_temp[key.name][0],res_filter1))
                    resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]
                    o1 = resx1
                    while len(resx1[1])!=len(resx2[1]):
                        resx1=resx2[:,np.abs(resx2[1])<3*np.std(resx2[1])]
                        resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]

                    outliers_index_x = np.isin(o1[-1], resx2[-1])
                    x_std =np.std(resx2[1])

                    # res_filter2 = Res_temp[key.name][2]
                    res_filter2 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][2])
                    # res_filter2 = rollingWindow(data=Res_temp[key.name][[0, 2], :],size=10)
                    resy1=np.vstack((Res_temp[key.name][0],res_filter2))
                    resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]
                    o2 = resy1
                    while len(resy1[1])!=len(resy2[1]):
                        resy1=resy2[:,np.abs(resy2[1])<3*np.std(resy2[1])]
                        resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]

                    outliers_index_y = np.isin(o2[-1], resy2[-1])
                    y_std =np.std(resy2[1])

                    # res_filter3 = Res_temp[key.name][3]
                    res_filter3 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][3])
                    # res_filter3 = rollingWindow(data=Res_temp[key.name][[0, 3], :],size=10)
                    resz1=np.vstack((Res_temp[key.name][0],res_filter3))
                    resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]
                    o3= resz1
                    while len(resz1[1])!=len(resz2[1]):
                        resz1=resz2[:,np.abs(resz2[1])<3*np.std(resz2[1])]
                        resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]

                    outliers_index_z = np.isin(o3[-1],resz2)
                    z_std =np.std(resz2[1])

                    # print(x_std,y_std,z_std,'%s'%key.name)

                    Res_std[key.name]=np.array([x_std,y_std,z_std])


                # print('%s data has processed'%(day.strftime('%Y-%m-%d')))


            else:
                file = '%s/%s-%s.csv' % (day.strftime('%Y-%m'), day.strftime('%Y-%m-%d'),sat.name)
                path = '../Produce/' + file
                res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
                Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
                Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
                Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)
                # Calculate the average value as of a day
                for key in SCAID:
                    res_filter1 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][1])
                    # res_filter1 = rollingWindow(data=Res_temp[key.name][[0, 1], :],size=20)
                    resx1 = np.vstack((Res_temp[key.name][0], res_filter1))
                    o1= resx1
                    resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]
                    while len(resx1[1]) != len(resx2[1]):
                        resx1 = resx2[:, np.abs(resx2[1]) < 3 * np.std(resx2[1])]
                        resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]

                    outliers_index_x = np.isin(o1[-1], resx2[-1])
                    x_std = np.std(resx2[1])

                    res_filter2 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][2])
                    # res_filter2 = rollingWindow(data=Res_temp[key.name][[0, 2], :],size=20)
                    resy1 = np.vstack((Res_temp[key.name][0], res_filter2))
                    resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]
                    o2 = resy1
                    while len(resy1[1]) != len(resy2[1]):
                        resy1 = resy2[:, np.abs(resy2[1]) < 3 * np.std(resy2[1])]
                        resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]

                    outliers_index_y = np.isin(o2[-1], resy2[-1])
                    y_std = np.std(resy2[1])

                    res_filter3 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][3])
                    # res_filter3 = rollingWindow(data=Res_temp[key.name][[0, 3], :], size=20)
                    resz1 = np.vstack((Res_temp[key.name][0], res_filter3))
                    resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]
                    o3= resz1
                    while len(resz1[1]) != len(resz2[1]):
                        resz1 = resz2[:, np.abs(resz2[1]) < 3 * np.std(resz2[1])]
                        resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]

                    outliers_index_z = np.isin(o3[-1],resz2)
                    z_std = np.std(resz2[1])

                    index=np.vstack((outliers_index_x,outliers_index_y,outliers_index_z))
                    #obtaining the time points of outliers
                    index_time=np.all(index,axis=0)
                    outliers_index_time_temp=o1[0,~index_time]


                    # print(x_std,y_std,z_std,'%s'%key.name)


                    Res_std[key.name]=np.vstack((Res_std[key.name],np.array([x_std,y_std,z_std])))



                # print('%s data has read' % (day.strftime('%Y-%m-%d')))
            days.append(day)

        except:
            print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))



    return Res_std,days

    pass



def saveHFNflie(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28'):
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    Resx={}
    Resy={}
    Resz={}
    Res_temp={}
    Res_std={}
    outliers_index={}
    outliers_index_time={}
    days=[]
    for day in daylist:
        try:
            days.append(day)
            file='%s/%s-%s.csv'%(day.strftime('%Y-%m'),day.strftime('%Y-%m-%d'),sat.name)
            path='../Produce/'+file
            res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
            Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
            Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
            Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)

            # Calculate the average value as of a day
            for key in SCAID:
                # res_filter1=Res_temp[key.name][1]
                res_filter1 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][1])
                # res_filter1=rollingWindow(data=Res_temp[key.name][[0,1],:],size=10)
                resx1=np.vstack((Res_temp[key.name][0],res_filter1))
                resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]
                o1 = copy.deepcopy(resx1)
                while len(resx1[1])!=len(resx2[1]):
                    resx1=resx2[:,np.abs(resx2[1])<3*np.std(resx2[1])]
                    resx2=resx1[:,np.abs(resx1[1])<3*np.std(resx1[1])]

                outliers_index_x = np.isin(o1[-1], resx2[-1])
                x_std =np.std(resx2[1])

                # res_filter2 = Res_temp[key.name][2]
                res_filter2 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][2])
                # res_filter2 = rollingWindow(data=Res_temp[key.name][[0, 2], :],size=10)
                resy1=np.vstack((Res_temp[key.name][0],res_filter2))
                resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]
                o2 = copy.deepcopy(resy1)
                while len(resy1[1])!=len(resy2[1]):
                    resy1=resy2[:,np.abs(resy2[1])<3*np.std(resy2[1])]
                    resy2=resy1[:,np.abs(resy1[1])<3*np.std(resy1[1])]

                outliers_index_y = np.isin(o2[-1], resy2[-1])
                y_std =np.std(resy2[1])

                # res_filter3 = Res_temp[key.name][3]
                res_filter3 = HPF(Fs=1, cutoff_freq=0.02, data=Res_temp[key.name][3])
                # res_filter3 = rollingWindow(data=Res_temp[key.name][[0, 3], :],size=10)
                resz1=np.vstack((Res_temp[key.name][0],res_filter3))
                resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]
                o3=copy.deepcopy(resz1)
                while len(resz1[1])!=len(resz2[1]):
                    resz1=resz2[:,np.abs(resz2[1])<3*np.std(resz2[1])]
                    resz2=resz1[:,np.abs(resz1[1])<3*np.std(resz1[1])]

                outliers_index_z = np.isin(o3[-1],resz2)
                z_std =np.std(resz2[1])

                print(x_std,y_std,z_std,'%s'%key.name)


                Resx[key.name]=resx2
                Resy[key.name]=resy2
                Resz[key.name]=resz2
                outliers_index[key.name]=np.vstack((outliers_index_x,outliers_index_y,outliers_index_z))
                #obtaining the time points of outliers
                index_time=np.all(outliers_index[key.name],axis=0)
                outliers_index_time[key.name]=o1[0,~index_time]
                Res_std[key.name]=np.array([x_std,y_std,z_std])

            fileDir = os.path.join('../HFN_Euler',
                                   '%s/%s/' % (day.strftime('%Y-%m'), day.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            file1='HFNeuler-%s_x_%s.h5'%((day.strftime('%Y-%m-%d'),sat.name))
            file2='HFNeuler-%s_y_%s.h5'%((day.strftime('%Y-%m-%d'),sat.name))
            file3='HFNeuler-%s_z_%s.h5'%((day.strftime('%Y-%m-%d'),sat.name))
            file4='HFNeuler-outliers_index-%s_x_%s.h5'%((day.strftime('%Y-%m-%d'),sat.name))

            with h5py.File(fileDir+'/'+file1,'w') as f:
                for key,value in Resx.items():
                    if isinstance(value,dict):
                        group=f.create_group(key)
                        for nested_key,nested_value in value.items():
                            group.create_dataset(nested_key,data=nested_value)

                    else:
                        f.create_dataset(key,data=value)

            with h5py.File(fileDir+'/'+file2,'w') as f:
                for key,value in Resy.items():
                    if isinstance(value,dict):
                        group=f.create_group(key)
                        for nested_key,nested_value in value.items():
                            group.create_dataset(nested_key,data=nested_value)

                    else:
                        f.create_dataset(key,data=value)

            with h5py.File(fileDir+'/'+file3,'w') as f:
                for key,value in Resz.items():
                    if isinstance(value,dict):
                        group=f.create_group(key)
                        for nested_key,nested_value in value.items():
                            group.create_dataset(nested_key,data=nested_value)

                    else:
                        f.create_dataset(key,data=value)

            with h5py.File(fileDir+'/'+file4,'w') as f:
                for key,value in outliers_index.items():
                    if isinstance(value,dict):
                        group=f.create_group(key)
                        for nested_key,nested_value in value.items():
                            group.create_dataset(nested_key,data=nested_value)

                    else:
                        f.create_dataset(key,data=value)


            print('%s_%s HFN-data has saved'%(day.strftime('%Y-%m-%d'),sat.name))
            pass
        except:
            print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))


def ConfigAnglefile(date, sat:SatID):
    Date = datetime.datetime.strptime(date, '%Y-%m-%d')
    file='../SunMoonAngle/%s/%s/Angle_Sun-Moon_%s_%s.h5'\
         %(Date.strftime('%Y-%m'),Date.strftime('%Y-%m-%d'),Date.strftime('%Y-%m-%d'),sat.name)
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


def PlotEuler_sunangle(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28',scaID=SCAID.No_1):
    Resx,Resy,Resz,Res_std,outliers_index,outliers_index_time= configReadEuler(sat=sat,begindate=begindate, enddate=enddate)

    Sundata=ConfigAnglefile(date=begindate,sat=sat)

    if scaID.value==1:

        seconds_x=Resx[scaID.name][0,:]
        x=Resx[scaID.name][1,:]
        # avg_x=Res[scaID.name][4,:]
        label_x='roll'

        seconds_y=Resy[scaID.name][0,:]
        y=Resy[scaID.name][1,:]
        # avg_y=Res[scaID.name][5,:]
        label_y='pitch'

        seconds_z=Resz[scaID.name][0,:]
        z=Resz[scaID.name][1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z='yaw'

        suntime=Sundata[scaID.name][0]
        sunangle=Sundata[scaID.name][1]

        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time=[]
        time_x = []
        time_y = []
        time_z = []
        for sec in suntime:
            dt = base_time + datetime.timedelta(seconds=sec)
            time.append(dt)

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        # fig, ax = plt.subplots(figsize=(12.8, 4.8))
        fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)

        ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5,  zorder=2)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5, zorder=2)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5, zorder=2)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        # ax.set_xlabel('Time(days)')
        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(-75, 75)

        ax2 = ax.twinx()
        ax2.set_ylabel('SunAngle(deg)')
        ax2.plot(time,sunangle, color='gray',label='SunAngle',linestyle='--')
        ax2.set_ylim(0, 180)
        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%m-%d')
        # ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in',  top=True, bottom=True, left=True,  width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        ax2.tick_params(direction='in', right=True)
        ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
        ax2.tick_params(axis='x', direction='in', right=True)


        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2)


        fig.savefig('../paper_figure/Fig13.png', dpi=300)


        plt.show()

    elif scaID.value==2:

        seconds_x=Resx[scaID.name][0,:]
        x=Resx[scaID.name][1,:]
        # avg_x=Res[scaID.name][4,:]
        label_x='roll'

        seconds_y=Resy[scaID.name][0,:]
        y=Resy[scaID.name][1,:]
        # avg_y=Res[scaID.name][5,:]
        label_y='pitch'

        seconds_z=Resz[scaID.name][0,:]
        z=Resz[scaID.name][1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z='yaw'

        suntime=Sundata[scaID.name][0]
        sunangle=Sundata[scaID.name][1]

            # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time=[]
        time_x = []
        time_y = []
        time_z = []
        for sec in suntime:
            dt = base_time + datetime.timedelta(seconds=sec)
            time.append(dt)

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        # fig, ax = plt.subplots(figsize=(12.8, 4.8))
        fig, ax = plt.subplots()
        # default_size = plt.rcParams['figure.figsize']
        # print("默认尺寸值:", default_size)
        # ax.plot(time, a)
        color_x=(31/255, 119/255, 180/255)
        color_y=(229/255, 114/255, 12/255)
        color_z=(35/255, 142/255, 35/255)
        ax.scatter(time_z, z, s=0.1, color=color_z,label=label_z, alpha=0.5)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.1, color='blue',label=label_x, alpha=0.5)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y,alpha=0.5)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        # ax.set_xlabel('Time(day)')
        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(-75, 75)


        ax2 = ax.twinx()
        ax2.set_ylabel('SunAngle(deg)')
        ax2.plot(time,sunangle, color='gray', label='SunAngle',linestyle='--')
        ax2.set_ylim(0, 180)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%m-%d')
        # ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in',  top=True, bottom=True, left=True,  width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        ax2.tick_params(direction='in', right=True)
        ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
        ax2.tick_params(axis='x', direction='in', right=True)


        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/Fig14.png', dpi=300)

        plt.show()

    elif scaID.value==3:

        seconds_x=Resx[scaID.name][0,:]
        x=Resx[scaID.name][1,:]
        # avg_x=Res[scaID.name][4,:]
        label_x='roll'

        seconds_y=Resy[scaID.name][0,:]
        y=Resy[scaID.name][1,:]
        # avg_y=Res[scaID.name][5,:]
        label_y='pitch'

        seconds_z=Resz[scaID.name][0,:]
        z=Resz[scaID.name][1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z='yaw'

        suntime=Sundata[scaID.name][0]
        sunangle=Sundata[scaID.name][1]

        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time=[]
        time_x = []
        time_y = []
        time_z = []

        for sec in suntime:
            dt = base_time + datetime.timedelta(seconds=sec)
            time.append(dt)

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x=(31/255, 119/255, 180/255)
        color_y=(229/255, 114/255, 12/255)
        color_z=(35/255, 142/255, 35/255)


        ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5,)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5,)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5,)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)



        # ax.set_xlabel('Date')
        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')

        ax2 = ax.twinx()
        ax2.set_ylabel('SunAngle(deg)')
        ax2.plot(time,sunangle, color='gray', label='SunAngle',linestyle='--')
        ax2.set_ylim(0, 180)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%y-%m-%d')
        # ax.xaxis.set_major_formatter(date_format)
        #
        # # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))


        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in',  top=True, bottom=True, left=True,  width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        ax2.tick_params(direction='in', right=True)
        ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
        ax2.tick_params(axis='x', direction='in', right=True)


        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/Fig15.png', dpi=300)

        plt.show()

    else:
        pass



    pass



def PlotEuler(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28',scaID=SCAID.No_1):
    Resx,Resy,Resz,Res_std,outliers_index,outliers_index_time= configReadEuler(sat=sat,begindate=begindate, enddate=enddate)
    #plot angle figure
    if scaID.value==1:

        seconds_x=Resx[scaID.name][0,:]
        x=Resx[scaID.name][1,:]
        # avg_x=Res[scaID.name][4,:]
        label_x='roll'

        seconds_y=Resy[scaID.name][0,:]
        y=Resy[scaID.name][1,:]
        # avg_y=Res[scaID.name][5,:]
        label_y='pitch'

        seconds_z=Resz[scaID.name][0,:]
        z=Resz[scaID.name][1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z='yaw'


        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time_x = []
        time_y = []
        time_z = []

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        # fig, ax = plt.subplots(figsize=(12.8, 4.8))
        fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)

        ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        # ax.set_xlabel('Time(days)')
        ax.set_xlabel('date')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(-75, 75)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        plt.show()

    elif scaID.value==2:

        seconds_x=Resx[scaID.name][0,:]
        x=Resx[scaID.name][1,:]
        # avg_x=Res[scaID.name][4,:]
        label_x='roll'

        seconds_y=Resy[scaID.name][0,:]
        y=Resy[scaID.name][1,:]
        # avg_y=Res[scaID.name][5,:]
        label_y='pitch'

        seconds_z=Resz[scaID.name][0,:]
        z=Resz[scaID.name][1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z='yaw'


        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time=[]
        time_x = []
        time_y = []
        time_z = []

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        # fig, ax = plt.subplots()
        # default_size = plt.rcParams['figure.figsize']
        # print("默认尺寸值:", default_size)
        # ax.plot(time, a)
        color_x=(31/255, 119/255, 180/255)
        color_y=(229/255, 114/255, 12/255)
        color_z=(35/255, 142/255, 35/255)
        ax.scatter(time_z, z, s=0.1, color=color_z,label=label_z, alpha=0.5)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.1, color='blue',label=label_x, alpha=0.5)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y,alpha=0.5)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        # ax.set_xlabel('Time(days)')
        ax.set_xlabel('date')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(-120, 120)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        plt.show()

    elif scaID.value==3:

        seconds_x=Resx[scaID.name][0,:]
        x=Resx[scaID.name][1,:]
        # avg_x=Res[scaID.name][4,:]
        label_x='roll'

        seconds_y=Resy[scaID.name][0,:]
        y=Resy[scaID.name][1,:]
        # avg_y=Res[scaID.name][5,:]
        label_y='pitch'

        seconds_z=Resz[scaID.name][0,:]
        z=Resz[scaID.name][1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z='yaw'

        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time=[]
        time_x = []
        time_y = []
        time_z = []

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        # fig, ax = plt.subplots(figsize=(12.8, 4.8))
        fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x=(31/255, 119/255, 180/255)
        color_y=(229/255, 114/255, 12/255)
        color_z=(35/255, 142/255, 35/255)


        ax.scatter(time_z, z, s=0.5, color=color_z,label=label_z, alpha=0.2)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.5, color=color_x,label=label_x, alpha=0.2)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.5, color=color_y, label=label_y,alpha=0.2)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)


        ax.set_xlabel('Date')
        ax.set_ylabel('res_angle(arcsec)')

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%y-%m-%d')
        # ax.xaxis.set_major_formatter(date_format)
        #
        # # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')

        ax.legend(loc='upper left')
        plt.show()

    else:
        pass



    pass

def PlotEuler2(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28',scaID=SCAID.No_1):
    """
    Plot the std of residual euler
    """
    Res_std,days= configReadEuler3(sat=sat,begindate=begindate, enddate=enddate)
    # plot STD figure
    if scaID.value==1:

        # seconds_x=Resx[scaID.name][0,:]
        # x=Resx[scaID.name][1,:]
        # avg_x=Res[scaID.name][4,:]
        label_x='roll'

        # seconds_y=Resy[scaID.name][0,:]
        # y=Resy[scaID.name][1,:]
        # avg_y=Res[scaID.name][5,:]
        label_y='pitch'

        # seconds_z=Resz[scaID.name][0,:]
        # z=Resz[scaID.name][1,:]
        # avg_z=Res[scaID.name][6,:]
        label_z='yaw'


        # plt.subplot(2, 1, 1)
        # base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        # time_x = []
        # time_y = []
        # time_z = []

        # for sec_x in seconds_x:
        #     dt = base_time + datetime.timedelta(seconds=sec_x)
        #     time_x.append(dt)
        #
        # for sec_y in seconds_y:
        #     dt = base_time + datetime.timedelta(seconds=sec_y)
        #     time_y.append(dt)
        #
        # for sec_z in seconds_z:
        #     dt = base_time + datetime.timedelta(seconds=sec_z)
        #     time_z.append(dt)

        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        # fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)

        # ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.2)
        # # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)
        #
        # ax.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.2)
        # # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)
        #
        # ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.2)
        # # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        ax.plot(days,Res_std[SCAID.No_1.name][:,0].T,color='blue',label=label_x)
        ax.plot(days, Res_std[SCAID.No_1.name][:,1].T, color=color_y, label=label_y)
        ax.plot(days, Res_std[SCAID.No_2.name][:,2].T, color=color_z, label=label_z)

        # ax.set_xlabel('Time(days)')
        ax.set_xlabel('date')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(0, 35)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax.tick_params(axis='y', direction='in', which='both', top=True, bottom=True, left=True, right=True, width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper right')
        plt.show()

    elif scaID.value==2:

        # seconds_x=Resx[scaID.name][0,:]
        # x=Resx[scaID.name][1,:]
        # # avg_x=Res[scaID.name][4,:]
        label_x='roll'
        #
        # seconds_y=Resy[scaID.name][0,:]
        # y=Resy[scaID.name][1,:]
        # # avg_y=Res[scaID.name][5,:]
        label_y='pitch'
        #
        # seconds_z=Resz[scaID.name][0,:]
        # z=Resz[scaID.name][1,:]
        # # avg_z=Res[scaID.name][6,:]
        label_z='yaw'
        #
        #
        # # plt.subplot(2, 1, 1)
        # base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # # 转换为日期并打印
        # time=[]
        # time_x = []
        # time_y = []
        # time_z = []
        #
        # for sec_x in seconds_x:
        #     dt = base_time + datetime.timedelta(seconds=sec_x)
        #     time_x.append(dt)
        #
        # for sec_y in seconds_y:
        #     dt = base_time + datetime.timedelta(seconds=sec_y)
        #     time_y.append(dt)
        #
        # for sec_z in seconds_z:
        #     dt = base_time + datetime.timedelta(seconds=sec_z)
        #     time_z.append(dt)

        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        # fig, ax = plt.subplots()
        # default_size = plt.rcParams['figure.figsize']
        # print("默认尺寸值:", default_size)
        # ax.plot(time, a)
        color_x=(31/255, 119/255, 180/255)
        color_y=(229/255, 114/255, 12/255)
        color_z=(35/255, 142/255, 35/255)
        # ax.scatter(time_z, z, s=0.1, color=color_z,label=label_z, alpha=0.2)
        # # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)
        #
        # ax.scatter(time_x, x, s=0.1, color='blue',label=label_x, alpha=0.2)
        # # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)
        #
        # ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y,alpha=0.2)
        # # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        ax.plot(days,Res_std[SCAID.No_2.name][:,0].T,color=color_x,label=label_x)
        ax.plot(days, Res_std[SCAID.No_2.name][:,1].T, color=color_y, label=label_y)
        ax.plot(days, Res_std[SCAID.No_2.name][:,2].T, color=color_z, label=label_z)

        # ax.set_xlabel('Time(days)')
        ax.set_xlabel('date')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(0, 35)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax.tick_params(axis='y', direction='in', which='both', top=True, bottom=True, left=True, right=True, width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)
        # ax.tick_params(axis='both', which='major', width=2, length=6)
        # ax.tick_params(axis='both', which='minor', width=1, length=3)
        ax.tick_params(axis='both', labelsize=10)

        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper right')

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/Fig9.png', dpi=300)
        plt.show()

    elif scaID.value==3:

        # seconds_x=Resx[scaID.name][0,:]
        # x=Resx[scaID.name][1,:]
        # # avg_x=Res[scaID.name][4,:]
        label_x='roll'
        #
        # seconds_y=Resy[scaID.name][0,:]
        # y=Resy[scaID.name][1,:]
        # # avg_y=Res[scaID.name][5,:]
        label_y='pitch'
        #
        # seconds_z=Resz[scaID.name][0,:]
        # z=Resz[scaID.name][1,:]
        # # avg_z=Res[scaID.name][6,:]
        label_z='yaw'
        #
        # # plt.subplot(2, 1, 1)
        # base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # # 转换为日期并打印
        # time=[]
        # time_x = []
        # time_y = []
        # time_z = []
        #
        # for sec_x in seconds_x:
        #     dt = base_time + datetime.timedelta(seconds=sec_x)
        #     time_x.append(dt)
        #
        # for sec_y in seconds_y:
        #     dt = base_time + datetime.timedelta(seconds=sec_y)
        #     time_y.append(dt)
        #
        # for sec_z in seconds_z:
        #     dt = base_time + datetime.timedelta(seconds=sec_z)
        #     time_z.append(dt)

        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        # ax.plot(time, a)
        color_x=(31/255, 119/255, 180/255)
        color_y=(229/255, 114/255, 12/255)
        color_z=(35/255, 142/255, 35/255)


        # ax.scatter(time_z, z, s=0.5, color=color_z,label=label_z, alpha=0.2)
        # # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)
        #
        # ax.scatter(time_x, x, s=0.5, color=color_x,label=label_x, alpha=0.2)
        # # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)
        #
        # ax.scatter(time_y, y, s=0.5, color=color_y, label=label_y,alpha=0.2)
        # # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        ax.plot(days, Res_std[SCAID.No_3.name][:, 0].T, color=color_x, label=label_x)
        ax.plot(days, Res_std[SCAID.No_3.name][:, 1].T, color=color_y, label=label_y)
        ax.plot(days, Res_std[SCAID.No_3.name][:, 2].T, color=color_z, label=label_z)


        ax.set_xlabel('Date')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(0, 35)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        #
        # # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax.tick_params(axis='y', direction='in', which='both', top=True, bottom=True, left=True, right=True, width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper right')
        ax.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/Fig10.png', dpi=300)
        plt.show()

    else:
        pass



    pass


def HPF(Fs,cutoff_freq,data):
    """
    High Pass Filter
    """
    wn=2*cutoff_freq/Fs
    order = 4  # filter order
    b, a = signal.butter(order, wn, btype='high', fs=1)
    filtered_data = signal.lfilter(b, a, data)

    return filtered_data


def rollingWindow(data,size:int):
    # raw data
    x = np.array(data[0,:])
    y = np.array(data[1,:])

    # Scroll Window Size
    window_size = size

    # Fitting the rolling window quadratic function
    for i in range(window_size, len(x) + 1):

        x_window = x[i - window_size:i]
        y_window = y[i - window_size:i]

        # Polynomial fitting
        poly_coeffs = np.polyfit(x_window, y_window, 2)

        y_fit = np.polyval(poly_coeffs, x_window)
        y[i-window_size:i]=y_fit

    new_data = np.vstack((x, y))

    return new_data

    pass

def PSD(date='2020-02-12', sat=SatID.C, scaID=SCAID.No_1):
    Date=datetime.datetime.strptime(date, '%Y-%m-%d')
    file = '%s/%s-%s.csv'%(Date.strftime('%Y-%m'),Date.strftime('%Y-%m-%d'),sat.name)
    path = '../Produce/' + file
    Res_temp={}
    res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)

    Res_temp[SCAID.No_1.name] = res[0:4, res[-1, :] == 'No_1'].astype(float)
    Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
    Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)

    Resx, Resy, Resz, Res_std, days,outliers_index_time = configReadEuler(sat=sat, begindate=date, enddate=date)

    fig = plt.figure(figsize=(23.8, 6.4))

    for key in Coord:

        if key.value==0:
            resx1 = Res_temp[scaID.name][[0,1]]
            resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]
            while len(resx1[1]) != len(resx2[1]):
                resx1 = resx2[:, np.abs(resx2[1]) < 3 * np.std(resx2[1])]
                resx2 = resx1[:, np.abs(resx1[1]) < 3 * np.std(resx1[1])]



            x_interp1 = np.interp(Res_temp[scaID.name][0]-Res_temp[scaID.name][0,0],
                                  resx2[0]-Res_temp[scaID.name][0,0], resx2[1])
            x_interp1 = np.array(x_interp1)


            x_interp2 = np.interp(Res_temp[scaID.name][0]-Res_temp[scaID.name][0,0],
                                  Resx[scaID.name][0]-Res_temp[scaID.name][0,0], Resx[scaID.name][1])
            x_interp2 = np.array(x_interp2)

            fig2 = plt.figure()

            axx = fig2.add_subplot(1, 1, 1)

            psd1, freq1 = plt.psd(x_interp1, Fs=1, NFFT=len(x_interp1))
            psd2, freq2 = plt.psd(x_interp2, Fs=1, NFFT=len(x_interp2))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD')

            psd1 = np.sqrt(2 * psd1)
            psd2 = np.sqrt(2 * psd2)
            ax=fig.add_axes([0.05+0.32*key.value, 0.12, 0.28, 0.8])
            ax.set_xscale('log')
            ax.set_yscale('log')

            # ax.plot(freq1, psd1, label='2020-02-12_No_1_x', lw=0.5,
            #         alpha=0.8)
            ax.plot(freq2, psd2, label='2020-02-12_HP_No_1_x', lw=0.5,
                    alpha=0.9, color='green')

            ax.legend(ncol=3)
            ax.grid(True, linestyle='dashed')
            font={'size': 20}
            ax.set_xlabel('Frequency [Hz]', fontdict=font)
            ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
            ax.legend(loc='upper right', ncol=1, fontsize=18, framealpha=0.8)
            ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
            ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
            ax.grid(True, linestyle='dashed')

            for spine in ax.spines.values():
                spine.set_linewidth(2)

            fig.align_ylabels()
            fig.align_xlabels()


        elif key.value==1:
            resy1 = Res_temp[scaID.name][[0, 2]]
            resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]
            while len(resy1[1]) != len(resy2[1]):
                resy1 = resy2[:, np.abs(resy2[1]) < 3 * np.std(resy2[1])]
                resy2 = resy1[:, np.abs(resy1[1]) < 3 * np.std(resy1[1])]

            y_interp1 = np.interp(Res_temp[scaID.name][0] - Res_temp[scaID.name][0, 0],
                                  resy2[0] - Res_temp[scaID.name][0, 0], resy2[1])
            y_interp1 = np.array(y_interp1)

            y_interp2 = np.interp(Res_temp[scaID.name][0] - Res_temp[scaID.name][0, 0],
                                  Resy[scaID.name][0] - Res_temp[scaID.name][0, 0], Resy[scaID.name][1])
            y_interp2 = np.array(y_interp2)

            fig2 = plt.figure()

            axx = fig2.add_subplot(1, 1, 1)

            psd1, freq1 = plt.psd(y_interp1, Fs=1, NFFT=len(y_interp1))
            psd2, freq2 = plt.psd(y_interp2, Fs=1, NFFT=len(y_interp2))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD')

            psd1 = np.sqrt(2 * psd1)
            psd2 = np.sqrt(2 * psd2)
            ax = fig.add_axes([0.05+0.32*key.value, 0.12, 0.28, 0.8])
            ax.set_xscale('log')
            ax.set_yscale('log')

            # ax.plot(freq1, psd1, label='%s_%s_y'%(date,scaID.name), lw=0.5,
            #         alpha=0.8)
            ax.plot(freq2, psd2, label='%s_HP_%s_y'%(date,scaID.name), lw=0.5,
                    alpha=0.9, color='orange')

            ax.legend(ncol=3)
            ax.grid(True, linestyle='dashed')
            font={'size': 20}
            ax.set_xlabel('Frequency [Hz]', fontdict=font)
            # ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
            ax.legend(loc='upper right', ncol=1, fontsize=18, framealpha=0.8)
            ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
            ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
            ax.grid(True, linestyle='dashed')

            for spine in ax.spines.values():
                spine.set_linewidth(2)
            fig.align_ylabels()
            fig.align_xlabels()


        elif key.value==2:
            resz1 = Res_temp[scaID.name][[0, 3]]
            resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]
            while len(resz1[1]) != len(resz2[1]):
                resz1 = resz2[:, np.abs(resz2[1]) < 3 * np.std(resz2[1])]
                resz2 = resz1[:, np.abs(resz1[1]) < 3 * np.std(resz1[1])]

            z_interp1 = np.interp(Res_temp[scaID.name][0] - Res_temp[scaID.name][0, 0],
                                  resz2[0] - Res_temp[scaID.name][0, 0], resz2[1])
            z_interp1 = np.array(z_interp1)

            z_interp2 = np.interp(Res_temp[scaID.name][0] - Res_temp[scaID.name][0, 0],
                                  Resz[scaID.name][0] - Res_temp[scaID.name][0, 0], Resz[scaID.name][1])
            z_interp2 = np.array(z_interp2)

            fig2 = plt.figure()

            axx = fig2.add_subplot(1, 1, 1)

            psd1, freq1 = plt.psd(z_interp1, Fs=1, NFFT=len(z_interp1))
            psd2, freq2 = plt.psd(z_interp2, Fs=1, NFFT=len(z_interp2))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD')

            psd1 = np.sqrt(2 * psd1)
            psd2 = np.sqrt(2 * psd2)
            ax = fig.add_axes([0.05+0.32*key.value, 0.12, 0.28, 0.8])
            ax.set_xscale('log')
            ax.set_yscale('log')

            # ax.plot(freq1, psd1, label='%s_%s_z'%(date,scaID.name), lw=0.5,
            #         alpha=0.8)
            ax.plot(freq2, psd2, label='%s_HP_%s_z'%(date,scaID.name), lw=0.5,
                    alpha=0.9, color='blue')

            ax.legend(ncol=3)
            ax.grid(True, linestyle='dashed')
            font={'size': 20}
            ax.set_xlabel('Frequency [Hz]', fontdict=font)
            # ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
            ax.legend(loc='upper right', ncol=1, fontsize=18, framealpha=0.8)
            ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
            ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
            ax.grid(True, linestyle='dashed')

            for spine in ax.spines.values():
                spine.set_linewidth(2)
            fig.align_ylabels()
            fig.align_xlabels()


            pass

        else:
            print("the input is wrong, please input again")
            pass

    fig.savefig('../paper_figure/figure10.png', dpi=300)
    plt.show()


    pass


def outliers_index(sat=SatID.C,begindate='2020-04-01',enddate='2020-04-01'):

    Resx, Resy, Resz, Res_std, outliers_index,outliers_index_time = configReadEuler(sat=sat, begindate=begindate,enddate=enddate)

    return outliers_index

def outliers_index_time(sat=SatID.C,begindate='2020-04-01',enddate='2020-04-01'):

    Resx, Resy, Resz, Res_std, outliers_index,outliers_index_time = configReadEuler(sat=sat, begindate=begindate,enddate=enddate)

    return outliers_index_time

def sunangle_and_noise(sat=SatID.C,begindate='2020-02-01',enddate='2020-02-28',scaID=SCAID.No_1, win_size=2):
    """
    Using Sunangle to Model Noise
    """
    Resx, Resy, Resz, Res_std, outliers_index, outliers_index_time = configReadEuler(sat=sat, begindate=begindate,
                                                                                     enddate=enddate)

    Sundata = ConfigAnglefile(date=begindate, sat=sat)

    if scaID.value == 1:

        window_size = win_size * 60 * 60
        num_windows = len() // window_size

        seconds_x = Resx[scaID.name][0, :]
        x = Resx[scaID.name][1, :]
        label_x = 'roll'

        seconds_y = Resy[scaID.name][0, :]
        y = Resy[scaID.name][1, :]
        label_y = 'pitch'

        seconds_z = Resz[scaID.name][0, :]
        z = Resz[scaID.name][1, :]
        label_z = 'yaw'

        suntime = Sundata[scaID.name][0]
        sunangle = Sundata[scaID.name][1]

        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time = []
        time_x = []
        time_y = []
        time_z = []
        for sec in suntime:
            dt = base_time + datetime.timedelta(seconds=sec)
            time.append(dt)

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        fig, ax = plt.subplots()
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)

        ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5, zorder=2)
        ax.scatter(time_x, x, s=0.1, color=color_x, label=label_x, alpha=0.5, zorder=2)
        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5, zorder=2)

        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(-75, 75)

        ax2 = ax.twinx()
        ax2.set_ylabel('SunAngle(deg)')
        ax2.plot(time, sunangle, color='gray', label='SunAngle', linestyle='--')
        ax2.set_ylim(0, 180)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M'))

        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        ax2.tick_params(direction='in', right=True)
        ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
        ax2.tick_params(axis='x', direction='in', right=True)

        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/Fig13.png', dpi=300)

        plt.show()

    elif scaID.value == 2:

        seconds_x = Resx[scaID.name][0, :]
        x = Resx[scaID.name][1, :]
        # avg_x=Res[scaID.name][4,:]
        label_x = 'roll'

        seconds_y = Resy[scaID.name][0, :]
        y = Resy[scaID.name][1, :]
        # avg_y=Res[scaID.name][5,:]
        label_y = 'pitch'

        seconds_z = Resz[scaID.name][0, :]
        z = Resz[scaID.name][1, :]
        # avg_z=Res[scaID.name][6,:]
        label_z = 'yaw'

        suntime = Sundata[scaID.name][0]
        sunangle = Sundata[scaID.name][1]

        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time = []
        time_x = []
        time_y = []
        time_z = []
        for sec in suntime:
            dt = base_time + datetime.timedelta(seconds=sec)
            time.append(dt)

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        # fig, ax = plt.subplots(figsize=(12.8, 4.8))
        fig, ax = plt.subplots()
        # default_size = plt.rcParams['figure.figsize']
        # print("默认尺寸值:", default_size)
        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)
        ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5)
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5)
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5)
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        # ax.set_xlabel('Time(day)')
        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        ax.set_ylim(-75, 75)

        ax2 = ax.twinx()
        ax2.set_ylabel('SunAngle(deg)')
        ax2.plot(time, sunangle, color='gray', label='SunAngle', linestyle='--')
        ax2.set_ylim(0, 180)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%m-%d')
        # ax.xaxis.set_major_formatter(date_format)

        # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        ax2.tick_params(direction='in', right=True)
        ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
        ax2.tick_params(axis='x', direction='in', right=True)

        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/Fig14.png', dpi=300)

        plt.show()

    elif scaID.value == 3:

        seconds_x = Resx[scaID.name][0, :]
        x = Resx[scaID.name][1, :]
        # avg_x=Res[scaID.name][4,:]
        label_x = 'roll'

        seconds_y = Resy[scaID.name][0, :]
        y = Resy[scaID.name][1, :]
        # avg_y=Res[scaID.name][5,:]
        label_y = 'pitch'

        seconds_z = Resz[scaID.name][0, :]
        z = Resz[scaID.name][1, :]
        # avg_z=Res[scaID.name][6,:]
        label_z = 'yaw'

        suntime = Sundata[scaID.name][0]
        sunangle = Sundata[scaID.name][1]

        # plt.subplot(2, 1, 1)
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time = []
        time_x = []
        time_y = []
        time_z = []

        for sec in suntime:
            dt = base_time + datetime.timedelta(seconds=sec)
            time.append(dt)

        for sec_x in seconds_x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x.append(dt)

        for sec_y in seconds_y:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y.append(dt)

        for sec_z in seconds_z:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z.append(dt)

        fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)

        ax.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5, )
        # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

        ax.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5, )
        # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

        ax.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5, )
        # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

        # ax.set_xlabel('Date')
        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')

        ax2 = ax.twinx()
        ax2.set_ylabel('SunAngle(deg)')
        ax2.plot(time, sunangle, color='gray', label='SunAngle', linestyle='--')
        ax2.set_ylim(0, 180)

        # ax.set_title('The relative error of roll-No2')
        # 设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%y-%m-%d')
        # ax.xaxis.set_major_formatter(date_format)
        #
        # # 设置横坐标的是时间刻度间隔
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # 顺时针旋转横坐标标签
        # plt.xticks(rotation=45)

        # plt.subplot(2, 1, 2)
        #
        # plt.figure(days,Res_std[scaID.name][:,1],label='x_std')
        # plt.figure(days, Res_std[scaID.name][:, 2], label='y_std')
        # plt.figure(days, Res_std[scaID.name][:, 3], label='z_std')
        plt.grid(True, axis='y', linestyle='dashed')

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True)

        ax2.tick_params(direction='in', right=True)
        ax2.tick_params(axis='y', direction='in', right=True, width=1.5)
        ax2.tick_params(axis='x', direction='in', right=True)

        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        fig.savefig('../paper_figure/Fig15.png', dpi=300)

        plt.show()

    else:
        pass

    pass


    pass


def plot_paper_Euler_sunangle():
    fig=plt.figure(figsize=(13.6,10.4))

    sat=SatID.C
    scaID=SCAID.No_1
    date='2020-05-01'
    Resx, Resy, Resz, Res_std, outliers_index, outliers_index_time = configReadEuler(sat=sat, begindate=date,
                                                                                     enddate=date)

    Sundata = ConfigAnglefile(date=date, sat=sat)

    seconds_x = Resx[scaID.name][0, :]
    x = Resx[scaID.name][1, :]
    # avg_x=Res[scaID.name][4,:]
    label_x = 'roll'

    seconds_y = Resy[scaID.name][0, :]
    y = Resy[scaID.name][1, :]
    # avg_y=Res[scaID.name][5,:]
    label_y = 'pitch'

    seconds_z = Resz[scaID.name][0, :]
    z = Resz[scaID.name][1, :]
    # avg_z=Res[scaID.name][6,:]
    label_z = 'yaw'

    suntime = Sundata[scaID.name][0]
    sunangle = Sundata[scaID.name][1]

    # plt.subplot(2, 1, 1)
    base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    # 转换为日期并打印
    time = []
    time_x = []
    time_y = []
    time_z = []
    for sec in suntime:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)

    for sec_x in seconds_x:
        dt = base_time + datetime.timedelta(seconds=sec_x)
        time_x.append(dt)

    for sec_y in seconds_y:
        dt = base_time + datetime.timedelta(seconds=sec_y)
        time_y.append(dt)

    for sec_z in seconds_z:
        dt = base_time + datetime.timedelta(seconds=sec_z)
        time_z.append(dt)

    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    ax1=fig.add_axes([0.085,0.6,0.35,0.35])
    ax1.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

    ax1.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

    ax1.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

    # ax.set_xlabel('Time(days)')
    font={'size':18}
    ax1.set_xlabel('Time (hours)', fontdict=font)
    ax1.set_ylabel('Res_angle (arcsec)', fontdict=font)
    ax1.set_ylim(-75, 75)

    ax2 = ax1.twinx()
    ax2.set_ylabel('SunAngle (deg)',fontdict=font)
    ax2.plot(time, sunangle, color='gray', label='SunAngle', linestyle='--')
    ax2.set_ylim(0, 180)

    # 设置横坐标的是时间刻度间隔
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.grid(True, axis='y', linestyle='dashed')

    ax1.legend(loc='upper left', ncol=1, fontsize='x-large', framealpha=0.8)
    ax2.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)

    ax1.tick_params(direction='in', top=True, bottom=True, left=True)
    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20)

    ax2.tick_params(direction='in', right=True)
    ax2.tick_params(axis='y', direction='in', right=True, width=1.5, labelsize=20)
    ax2.tick_params(axis='x', direction='in', right=True)

    y_major_locator=MultipleLocator(20)
    ax2.yaxis.set_major_locator(y_major_locator)

    ax1.text(0,-0.27,s='(a) The residual angle for %s-%s on %s'%(sat.name,scaID.name[-1],date),fontdict={'size':20}, transform=ax1.transAxes)


    for spine in ax1.spines.values():
        spine.set_linewidth(2)



    sat=SatID.C
    scaID=SCAID.No_2
    date='2020-05-01'
    Resx, Resy, Resz, Res_std, outliers_index, outliers_index_time = configReadEuler(sat=sat, begindate=date,
                                                                                     enddate=date)

    Sundata = ConfigAnglefile(date=date, sat=sat)

    seconds_x = Resx[scaID.name][0, :]
    x = Resx[scaID.name][1, :]
    # avg_x=Res[scaID.name][4,:]
    label_x = 'roll'

    seconds_y = Resy[scaID.name][0, :]
    y = Resy[scaID.name][1, :]
    # avg_y=Res[scaID.name][5,:]
    label_y = 'pitch'

    seconds_z = Resz[scaID.name][0, :]
    z = Resz[scaID.name][1, :]
    # avg_z=Res[scaID.name][6,:]
    label_z = 'yaw'

    suntime = Sundata[scaID.name][0]
    sunangle = Sundata[scaID.name][1]

    # plt.subplot(2, 1, 1)
    base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    # 转换为日期并打印
    time = []
    time_x = []
    time_y = []
    time_z = []
    for sec in suntime:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)

    for sec_x in seconds_x:
        dt = base_time + datetime.timedelta(seconds=sec_x)
        time_x.append(dt)

    for sec_y in seconds_y:
        dt = base_time + datetime.timedelta(seconds=sec_y)
        time_y.append(dt)

    for sec_z in seconds_z:
        dt = base_time + datetime.timedelta(seconds=sec_z)
        time_z.append(dt)

    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    ax1=fig.add_axes([0.57,0.6,0.35,0.35])
    ax1.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

    ax1.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

    ax1.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

    # ax.set_xlabel('Time(days)')
    font={'size':18}
    ax1.set_xlabel('Time (hours)', fontdict=font)
    ax1.set_ylabel('Res_angle (arcsec)', fontdict=font)
    ax1.set_ylim(-75, 75)

    ax2 = ax1.twinx()
    ax2.set_ylabel('SunAngle (deg)',fontdict=font)
    ax2.plot(time, sunangle, color='gray', label='SunAngle', linestyle='--')
    ax2.set_ylim(0, 180)

    # 设置横坐标的是时间刻度间隔
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.grid(True, axis='y', linestyle='dashed')

    ax1.legend(loc='upper left', ncol=1, fontsize='x-large', framealpha=0.8)
    ax2.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)

    ax1.tick_params(direction='in', top=True, bottom=True, left=True)
    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20)

    ax2.tick_params(direction='in', right=True)
    ax2.tick_params(axis='y', direction='in', right=True, width=1.5, labelsize=20)
    ax2.tick_params(axis='x', direction='in', right=True)

    y_major_locator=MultipleLocator(20)
    ax2.yaxis.set_major_locator(y_major_locator)

    ax1.text(0,-0.27,s='(b) The residual angle for %s-%s on %s'%(sat.name,scaID.name[-1],date),fontdict={'size':20}, transform=ax1.transAxes)


    for spine in ax1.spines.values():
        spine.set_linewidth(2)



    sat=SatID.C
    scaID=SCAID.No_3
    date='2020-05-01'
    Resx, Resy, Resz, Res_std, outliers_index, outliers_index_time = configReadEuler(sat=sat, begindate=date,
                                                                                     enddate=date)

    Sundata = ConfigAnglefile(date=date, sat=sat)

    seconds_x = Resx[scaID.name][0, :]
    x = Resx[scaID.name][1, :]
    # avg_x=Res[scaID.name][4,:]
    label_x = 'roll'

    seconds_y = Resy[scaID.name][0, :]
    y = Resy[scaID.name][1, :]
    # avg_y=Res[scaID.name][5,:]
    label_y = 'pitch'

    seconds_z = Resz[scaID.name][0, :]
    z = Resz[scaID.name][1, :]
    # avg_z=Res[scaID.name][6,:]
    label_z = 'yaw'

    suntime = Sundata[scaID.name][0]
    sunangle = Sundata[scaID.name][1]

    # plt.subplot(2, 1, 1)
    base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    # 转换为日期并打印
    time = []
    time_x = []
    time_y = []
    time_z = []
    for sec in suntime:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)

    for sec_x in seconds_x:
        dt = base_time + datetime.timedelta(seconds=sec_x)
        time_x.append(dt)

    for sec_y in seconds_y:
        dt = base_time + datetime.timedelta(seconds=sec_y)
        time_y.append(dt)

    for sec_z in seconds_z:
        dt = base_time + datetime.timedelta(seconds=sec_z)
        time_z.append(dt)

    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    ax1=fig.add_axes([0.085,0.12,0.35,0.35])
    ax1.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

    ax1.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

    ax1.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

    # ax.set_xlabel('Time(days)')
    font={'size':18}
    ax1.set_xlabel('Time (hours)', fontdict=font)
    ax1.set_ylabel('Res_angle (arcsec)', fontdict=font)
    ax1.set_ylim(-75, 75)

    ax2 = ax1.twinx()
    ax2.set_ylabel('SunAngle (deg)',fontdict=font)
    ax2.plot(time, sunangle, color='gray', label='SunAngle', linestyle='--')
    ax2.set_ylim(0, 180)

    # 设置横坐标的是时间刻度间隔
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.grid(True, axis='y', linestyle='dashed')

    ax1.legend(loc='lower left', ncol=1, fontsize='x-large', framealpha=0.8)
    ax2.legend(loc='lower right', ncol=1, fontsize='x-large', framealpha=0.8)

    ax1.tick_params(direction='in', top=True, bottom=True, left=True)
    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20)

    ax2.tick_params(direction='in', right=True)
    ax2.tick_params(axis='y', direction='in', right=True, width=1.5, labelsize=20)
    ax2.tick_params(axis='x', direction='in', right=True)

    y_major_locator=MultipleLocator(20)
    ax2.yaxis.set_major_locator(y_major_locator)

    ax1.text(0,-0.27,s='(c) The residual angle for %s-%s on %s'%(sat.name,scaID.name[-1],date),fontdict={'size':20}, transform=ax1.transAxes)


    for spine in ax1.spines.values():
        spine.set_linewidth(2)




    sat=SatID.C
    scaID=SCAID.No_2
    date='2020-08-30'
    Resx, Resy, Resz, Res_std, outliers_index, outliers_index_time = configReadEuler(sat=sat, begindate=date,
                                                                                     enddate=date)

    Sundata = ConfigAnglefile(date=date, sat=sat)

    seconds_x = Resx[scaID.name][0, :]
    x = Resx[scaID.name][1, :]
    # avg_x=Res[scaID.name][4,:]
    label_x = 'roll'

    seconds_y = Resy[scaID.name][0, :]
    y = Resy[scaID.name][1, :]
    # avg_y=Res[scaID.name][5,:]
    label_y = 'pitch'

    seconds_z = Resz[scaID.name][0, :]
    z = Resz[scaID.name][1, :]
    # avg_z=Res[scaID.name][6,:]
    label_z = 'yaw'

    suntime = Sundata[scaID.name][0]
    sunangle = Sundata[scaID.name][1]

    # plt.subplot(2, 1, 1)
    base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    # 转换为日期并打印
    time = []
    time_x = []
    time_y = []
    time_z = []
    for sec in suntime:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)

    for sec_x in seconds_x:
        dt = base_time + datetime.timedelta(seconds=sec_x)
        time_x.append(dt)

    for sec_y in seconds_y:
        dt = base_time + datetime.timedelta(seconds=sec_y)
        time_y.append(dt)

    for sec_z in seconds_z:
        dt = base_time + datetime.timedelta(seconds=sec_z)
        time_z.append(dt)

    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    ax1=fig.add_axes([0.57,0.12,0.35,0.35])
    ax1.scatter(time_z, z, s=0.1, color=color_z, label=label_z, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_z, s=0.1, color=color_z,label=label_z)

    ax1.scatter(time_x, x, s=0.1, color='blue', label=label_x, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_x, s=0.1, color=color_x,label=label_x)

    ax1.scatter(time_y, y, s=0.1, color=color_y, label=label_y, alpha=0.5, zorder=2)
    # ax.scatter(time, avg_y, s=0.1, color=color_y,label=label_y)

    # ax.set_xlabel('Time(days)')
    font={'size':18}
    ax1.set_xlabel('Time (hours)', fontdict=font)
    ax1.set_ylabel('Res_angle (arcsec)', fontdict=font)
    ax1.set_ylim(-75, 75)

    ax2 = ax1.twinx()
    ax2.set_ylabel('SunAngle (deg)',fontdict=font)
    ax2.plot(time, sunangle, color='gray', label='SunAngle', linestyle='--')
    ax2.set_ylim(0, 180)

    # 设置横坐标的是时间刻度间隔
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.grid(True, axis='y', linestyle='dashed')

    ax1.legend(loc='lower left', ncol=1, fontsize='x-large', framealpha=0.8)
    ax2.legend(loc='lower right', ncol=1, fontsize='x-large', framealpha=0.8)

    ax1.tick_params(direction='in', top=True, bottom=True, left=True)
    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20)

    ax2.tick_params(direction='in', right=True)
    ax2.tick_params(axis='y', direction='in', right=True, width=1.5, labelsize=20)
    ax2.tick_params(axis='x', direction='in', right=True)

    y_major_locator=MultipleLocator(20)
    ax2.yaxis.set_major_locator(y_major_locator)

    ax1.text(0,-0.27,s='(d) The residual angle for %s-%s on %s'%(sat.name,scaID.name[-1],date),fontdict={'size':20}, transform=ax1.transAxes)


    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    fig.savefig('../paper_figure/figure3.png', dpi=300)

    plt.show()

    pass

def plot_paper_STD_Euler():
    fig = plt.figure(figsize=(13.6, 10.4))
    sat = SatID.C
    begindate = '2020-01-01'
    enddate = '2020-12-31'
    Res_std, days = configReadEuler3(sat=sat, begindate=begindate, enddate=enddate)

    ax1 = fig.add_axes([0.085, 0.53, 0.9, 0.43])

    scaID = SCAID.No_1

    label_x = 'roll'
    label_y = 'pitch'
    label_z = 'yaw'
    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    # ax1.plot(days, Res_std[scaID.name][:, 0].T, color=color_x, label=label_x)
    # ax1.plot(days, Res_std[scaID.name][:, 1].T, color=color_y, label=label_y)
    ax1.plot(days, Res_std[scaID.name][:, 2].T, color=color_z, label=label_z)

    font = {'size': 20}
    ax1.set_ylabel('Standard deviations of noise (arcsec)', fontdict=font)
    ax1.set_ylim(15, 32)
    date_format = mdates.DateFormatter('%y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    y_major_locator=MultipleLocator(2)
    ax1.yaxis.set_major_locator(y_major_locator)

    ax1.legend(loc='upper right', ncol=1, fontsize=18, framealpha=0.8)

    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20, labelbottom=False)
    plt.grid(True, axis='y', linestyle='dashed')

    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.text(0.02, 0.90, s='(a)', fontdict={'size': 25}, transform=ax1.transAxes)


    sat = SatID.D
    begindate = '2020-01-01'
    enddate = '2020-12-31'
    Res_std2, days2 = configReadEuler3(sat=sat, begindate=begindate, enddate=enddate)

    ax1 = fig.add_axes([0.085, 0.075, 0.9, 0.43])
    scaID = SCAID.No_1
    label_x = 'roll'
    label_y = 'pitch'
    label_z = 'yaw'
    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    # ax1.plot(days2, Res_std2[scaID.name][:, 0].T, color=color_x, label=label_x)
    # ax1.plot(days2, Res_std2[scaID.name][:, 1].T, color=color_y, label=label_y)
    ax1.plot(days2, Res_std2[scaID.name][:, 2].T, color=color_z, label=label_z)


    font = {'size': 20}
    ax1.set_xlabel('date', fontdict=font)
    ax1.set_ylabel('Standard deviations of noise (arcsec)', fontdict=font)
    ax1.set_ylim(17, 30)
    date_format = mdates.DateFormatter('%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    y_major_locator=MultipleLocator(2)
    ax1.yaxis.set_major_locator(y_major_locator)

    ax1.legend(loc='upper right', ncol=1, fontsize=18, framealpha=0.8)

    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True,width=1.5, labelsize=20, pad=4)
    plt.grid(True, axis='y', linestyle='dashed')

    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.text(0.02, 0.90, s='(b)', fontdict={'size': 25}, transform=ax1.transAxes)


    fig.savefig('../paper_figure/figure4.png', dpi=300)

    plt.show()

    pass


def plot_PDF(date='2020-04-10',sat=SatID.C, scaID=SCAID.No_1):
    Resx, Resy, Resz, Res_std, outliers_index, outliers_index_time = configReadEuler(sat=sat, begindate=date,
                                                                                     enddate=date)
    fig=plt.figure(figsize=(13.6, 15.6))

    color_x = 'green'
    color_y = 'orange'
    color_z = 'blue'

    for key in Coord:

        ax1 = fig.add_axes([0.1, 0.05+0.32*key.value, 0.8,0.28])
        if key.value==0:
            data=Resx
            color=color_x
        elif key.value==1:
            data=Resy
            color=color_y
        elif key.value==2:
            data=Resz
            color=color_z
            pass

        ax1.hist(data[scaID.name][1], bins=100, density=True, alpha=0.7, color=color,
                 label='HP-C-%s-%s'%(scaID.name,key.name))
        xmin=np.min(data[scaID.name][1])
        xmax=np.max(data[scaID.name][1])
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(data[scaID.name][1]), np.std(data[scaID.name][1]))
        ax1.plot(x, p, 'grey', linewidth=5)
        font = {'size': 20}
        if key.value==0:
            ax1.set_xlabel('Value (arcsec)', fontdict=font)
        else:
            pass
        if key.value==2:
            ax1.set_title('Probability Density Distribution', fontdict=font)
        else:
            pass
        ax1.set_ylabel('Probability Density', fontdict=font)
        if key.value==2:
            ax1.set_ylabel('Probability Density', fontdict=font, labelpad=14)
        else:
            pass
        ax1.legend(loc='upper right', ncol=1, fontsize=18, framealpha=0.8)
        ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
        ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        plt.grid(True, linestyle='dashed')

        for spine in ax1.spines.values():
            spine.set_linewidth(2)

    fig.savefig('../paper_figure/figure9.png', dpi=300)
    plt.show()

    pass

def plot_PDF1(date='2020-04-10', sat=SatID.C, scaID=SCAID.No_1):
    Resx, Resy, Resz, Res_std, outliers_index, outliers_index_time = configReadEuler(sat=sat, begindate=date,
                                                                                     enddate=date)
    fig = plt.figure(figsize=(13.6, 6.4))

    color_x = 'green'
    color_y = 'orange'
    color_z = 'blue'

    for key in Coord:

        ax1 = fig.add_axes([0.1+ 0.32 * key.value, 0.1 , 0.24, 0.8])
        if key.value == 0:
            data = Resx
            color = color_x
        elif key.value == 1:
            data = Resy
            color = color_y
        elif key.value == 2:
            data = Resz
            color = color_z
            pass

        ax1.hist(data[scaID.name][1], bins=100, density=True, alpha=0.7, color=color,
                 label='HP-C-%s-%s' % (scaID.name, key.name))
        xmin = np.min(data[scaID.name][1])
        xmax = np.max(data[scaID.name][1])
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(data[scaID.name][1]), np.std(data[scaID.name][1]))
        ax1.plot(x, p, 'grey', linewidth=2)
        font = {'size': 20}
        ax1.set_xlabel('Value (arcsec)', fontdict=font)
        if key.value == 0:
            ax1.set_ylabel('Probability Density', fontdict=font)
        else:
            pass
        ax1.legend(loc='upper right', ncol=1, fontsize=14, framealpha=0.8)
        ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
        ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5,
                        labelsize=20)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 100:.2f}%'))
        plt.grid(True, linestyle='dashed')

        for spine in ax1.spines.values():
            spine.set_linewidth(2)

    fig.savefig('../paper_figure/figure9.png', dpi=300)
    plt.show()

    pass


if __name__ == '__main__':
    # daylist = GeoMathKit.dayListByDay('2020-08-01', '2020-08-31')
    # for day in daylist:
    #     try:
    #         PlotEuler_sunangle(sat=SatID.C, begindate=day.strftime('%Y-%m-%d'),
    #                            enddate=day.strftime('%Y-%m-%d'), scaID=SCAID.No_3)
    #     except:
    #         print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))


    PlotEuler_sunangle(sat=SatID.C,begindate='2020-05-01', enddate='2020-05-01', scaID=SCAID.No_2)
    # PSD()
    # PlotEuler(sat=SatID.C,begindate='2020-03-04', enddate='2020-03-04', scaID=SCAID.No_1)
    # PlotEuler2(sat=SatID.C,begindate='2020-01-01', enddate='2020-12-31', scaID=SCAID.No_2)
    # PlotEuler2(sat=SatID.D, begindate='2020-01-01', enddate='2020-12-31', scaID=SCAID.No_3)
    # Resx,Resy,Resz,Res_std,outliers_index,outliers_index_time=configReadEuler(sat=SatID.C,begindate='2020-04-01', enddate='2020-04-01')
    saveHFNflie(sat=SatID.C,begindate='2020-04-02',enddate='2020-04-02')
    # std,day=configReadEuler3(sat=SatID.C,begindate='2020-03-04',enddate='2020-03-04')

    # plot_paper_Euler_sunangle()
    # plot_paper_STD_Euler()

    PSD(date='2020-03-28', sat=SatID.C, scaID=SCAID.No_3)
    # plot_PDF1(date='2020-04-10',sat=SatID.C, scaID=SCAID.No_3)
    pass