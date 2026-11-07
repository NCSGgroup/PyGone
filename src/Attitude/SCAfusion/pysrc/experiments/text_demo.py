
from pysrc.core.settings import SatID, SCAID, Mission
# from pysrc.fusion.multi_sca import SCAweighting
import numpy as np
import os
import quaternion
from pysrc.core.geometry import GeoMathKit
from enum import Enum
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B

import datetime
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt




class SCAinterOption(Enum):
    NoInterp = 0
    OneHZ = 1
    TwoHZ = 2

class Demone:
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

        Demone.SCF2SRF = QSA

        pass

    def TimeTagTransformation(self):
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
            print(valid)
            xp = xp[valid]
            yp = yp[valid]
            print(xp)
            V = np.interp(x=x, xp=xp, fp=yp)

            V_int = np.trunc(V) + tim[0][0]
            V_frac = V - np.trunc(V)
            scaValue[0] = V_int
            scaValue[1] = V_frac * 1e6
            sca[key.name] = scaValue

        '''Step 2: RCV ---> GPS'''
        clk = self.__CLK1B
        # make a judgement on the quality
        index = np.where(clk[2] > 0)[0] #返回的是数据位置信息
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

    def InterpSCA(self,sca1B):
        #Unify the time series of Sca1A and Sca1B
        sat = SatID.C

        sca1A=self.TimeTagTransformation()
        Sca1B={}
        Sca1A={}

        for key in SCAID:
            scaValue = sca1B
            SCA1A_ID = sca1A[key.name]
            starttime=SCA1A_ID[0][0]
            time1=SCA1A_ID[0]-starttime+SCA1A_ID[1] * 1e-6
            time2=scaValue[0]-starttime
            Demone.ConfigSCF2SRF()
            QSA=Demone.SCF2SRF
            qb=quaternion.as_quat_array(SCA1A_ID[2:6,:].T)
            if key.name=='No_1':
                qa = qb * QSA[SatID.C][0]
            elif key.name=='No_2':
                qa = qb * QSA[SatID.C][1]
            elif key.name=='No_3':
                qa = qb * QSA[SatID.C][2]
            else:
                pass

            qc = quaternion.as_float_array(qa).T
            SCA1A_ID[2:6,:] = qc

            #interpolation
            new=quaternion.squad(R_in=qa,t_in=time1,t_out=time2)
            qc=quaternion.as_float_array(new).T
            sca=[[],[],[],[],[]]
            sca[0]=scaValue[0]
            sca[1:5]=-qc
            Sca1A[key.name]=np.array(sca)
            Sca1B[key.name]=scaValue

        return Sca1A,Sca1B,sat


    @staticmethod
    def demo2Euler(SCA1A,SCA1B):

        Euler = {}

        for key in SCAID:
            sca1aValue=SCA1A[key.name]
            sca1bValue=SCA1B[key.name]
            # # 将SCA数据的JPL转化成Hamilton
            # sca1aValue[2:5, :] = -sca1aValue[2:5, :]
            # sca1bValue[2:5, :] = -sca1bValue[2:5, :]
            q1=quaternion.as_quat_array(sca1aValue[1:5,:].T)
            q2=quaternion.as_quat_array(sca1bValue[1:5,:].T)
            # 对 q1 中的每个四元数进行取逆操作
            num_quaternions = q1.size
            for i in range(num_quaternions):
                q1[i] = q1[i].inverse()

            q=q1*q2
            detal=quaternion.as_float_array(q)
            tem_c=detal[1:,1:]
            #化角秒
            Euler[key.name]=detal[1:,1:]*2*3600


            Euler[key.name]=Euler[key.name].T
        return Euler
        pass

def EulerFile(begindate='2019-01-02',enddate='2019-01-02'):
    """
    saving Euler date:../Produce
    """
    daylist=GeoMathKit.dayListByDay(begindate,enddate)
    for day in daylist:
        day = day.strftime('%Y-%m-%d')
        date = day
        sat = SatID.C

        L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
        SCA1A = L1A.getSCA(sat)

        L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
        TIM1B = L1B.getTIM(sat)
        CLK1B = L1B.getCLK(sat)
        sca1B = L1B.getSCA(sat)
        a = Demone(satID=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

        SCA1A,SCA1B,sat=a.InterpSCA(sca1B)
        Euler=a.demo2Euler(SCA1A,SCA1B)
        file_name = day+'-'+sat.name+'.csv'
        fname=datetime.datetime.strptime(day, '%Y-%m-%d')
        fname=fname.strftime('%Y-%m')
        path=os.path.join('../Produce',fname)
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, file_name)
        #input the Euler Date
        if os.path.exists(file_path):
            print("%s文件存在"%(file_path))
        else:
            for key in SCAID:
                array = SCA1B[key.name][:, 1:]
                suffix_array = np.full((1, array.shape[1]), key.name, dtype=object)
                merged_euler = np.vstack((SCA1B[key.name][0, 1:], Euler[key.name], suffix_array))
                merged_euler = merged_euler.T
                with open(file_path, 'a') as f:
                    np.savetxt(f, merged_euler, fmt='%s', delimiter=' ', newline='\n')

            print("Saving %s-%s Euler Date completed"%(day, sat.name))

def ReadFile(begindate='2019-01-02',enddate='2019-01-02'):
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    Res={}
    Res_temp={}
    for day in daylist:
        if day.strftime('%Y-%m-%d')==begindate:
            file='%s/%s-C.csv'%(day.strftime('%Y-%m'),day.strftime('%Y-%m-%d'))
            path='../Produce/'+file
            res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)

            Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
            Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
            Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)

            # Calculate the average value as of day
            for key in SCAID:
                Res_avg=np.mean(Res_temp[key.name][1:4,np.abs(Res_temp[key.name][1,:]<5)],axis=1)
                x_avg=np.ones(len(Res_temp[key.name][0,:]))*Res_avg[0]
                y_avg = np.ones(len(Res_temp[key.name][0,:])) * Res_avg[1]
                z_avg = np.ones(len(Res_temp[key.name][0,:])) * Res_avg[2]
                Res[key.name]=np.vstack([Res_temp[key.name],x_avg,y_avg,z_avg])


            print('%s data has read'%(day.strftime('%Y-%m-%d')))
        else:
            file = '%s/%s-C.csv' % (day.strftime('%Y-%m'), day.strftime('%Y-%m-%d'))
            path = '../Produce/' + file
            res = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), dtype=str, unpack=True)
            Res_temp[SCAID.No_1.name] = res[0:4,res[-1,:] == 'No_1'].astype(float)
            Res_temp[SCAID.No_2.name] = res[0:4, res[-1, :] == 'No_2'].astype(float)
            Res_temp[SCAID.No_3.name] = res[0:4, res[-1, :] == 'No_3'].astype(float)
            # Calculate the average value as of day
            for key in SCAID:
                Res_avg=np.mean(Res_temp[key.name][1:4,np.abs(Res_temp[key.name][1,:]<5)],axis=1)
                x_avg=np.ones(len(Res_temp[key.name][0,:]))*Res_avg[0]
                y_avg = np.ones(len(Res_temp[key.name][0,:])) * Res_avg[1]
                z_avg = np.ones(len(Res_temp[key.name][0,:])) * Res_avg[2]
                Res_temp[key.name]=np.vstack([Res_temp[key.name],x_avg,y_avg,z_avg])

            Res[SCAID.No_1.name] = np.hstack((Res[SCAID.No_1.name],Res_temp[SCAID.No_1.name]))
            Res[SCAID.No_2.name] = np.hstack((Res[SCAID.No_2.name],Res_temp[SCAID.No_2.name]))
            Res[SCAID.No_3.name] = np.hstack((Res[SCAID.No_3.name],Res_temp[SCAID.No_3.name]))

            print('%s data has read' % (day.strftime('%Y-%m-%d')))

    return Res

    pass


# def Daymean(time,date):
#     data={}
#     data['time']=time
#     data['value']=date
#     # 转换为DataFrame格式
#     df = pd.DataFrame(data)
#     base_time = pd.to_datetime('2000-01-01 12:00:00')
#     df['time']=pd.to_timedelta(df['time'])+(base_time - pd.to_datetime('1970-01-01'))
#     df['date'] = df['time'].astype(str).str.split().str[0]
#     daily_avg = df.groupby('date')['value'].mean()
#     return daily_avg
#
#     pass

def PlotEuler(begindate='2019-01-02',enddate='2019-01-02'):
    Res = ReadFile(begindate=begindate, enddate=enddate)
    x=Res['No_1'][1,:]
    seconds=Res['No_1'][0,:]
    avg_x=Res['No_1'][4,:]
    i=0
    untime=[]
    unx=[]
    Time=[]
    X=[]
    AVG_x=[]
    for j in range(len(x)):
        if np.abs(float(x[j])) > 5:
            i+=1
            untime.append(float(seconds[j]))
            unx.append(float(x[j]))
        else:
            i=i
            Time.append(float(seconds[j]))
            X.append(float(x[j]))
            AVG_x.append(float(avg_x[j]))
            pass

    # daily_avg=Daymean(Time,X)
    # print(daily_avg)

    base_time = datetime.datetime(2000, 1, 1,12,0,0)
    # 转换为日期并打印
    time=[]
    for sec in Time:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)
    fig, ax = plt.subplots()
    # ax.plot(time, X)
    ax.scatter(time, X, s=1, c='gray', alpha=0.5)
    ax.scatter(time, AVG_x, s=1, c='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('roll')
    ax.set_title('The relative error of roll-No1')
    # 设置横坐标的日期格式为 "%y-%m-%d"
    date_format = mdates.DateFormatter('%y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)

    # 设置横坐标的刻度间隔为 1 天
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))

    # 顺时针旋转横坐标标签
    # plt.xticks(rotation=45)

    plt.show()
    return untime,unx






if __name__ == '__main__':

    # EulerFile(begindate='2019-01-02',enddate='2019-01-30')
    untime,unx=PlotEuler(begindate='2019-01-02', enddate='2019-01-30')







    pass
