import matplotlib.pyplot as plt
import numpy as np
from pysrc.core.geometry import GeoMathKit
from pysrc.core.settings import SatID, SCAID, Mission
import os
import datetime
from pysrc.analysis import hfn_analysis as HFNAnalysis
from pysrc.fusion.multi_sca import SCAweighting
from tqdm import trange
import quaternion
import copy
import matplotlib.dates as mdates
import pysrc.getSunPos.demo.demo_PlanetEphemerides as gp
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
import h5py

class SunMoonAngle:
    def __init__(self,sat:SatID,date='2020-02-02'):
        self.sat=sat
        self.date=date
        self.LOS=self.ConfigIF2LOS(date)
        self.Angletime=self.LOS[:,0].T
        self.QSA=self.ConfigSCF2SRF()
        pass




    def ConfigIF2LOS(self,date):

        file='../result/product/GRACE_FO/RL04/L1B/%s/IF2LOS_%s_HUST01_%s.npy'%(date,date,self.sat.name)
        LOS=np.load(file)

        return LOS


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

        return QSA

        pass

    def ConfigPos_SunMoon(self):
        sun=[]
        moon=[]
        for i in trange(len(self.Angletime[::60])):
        # for i in trange(100):
            itrs_sun,itrs_moon=gp.demo(self.Angletime[60*i])
            sun.append(itrs_sun)
            moon.append(itrs_moon)
            pass
        sun=np.array(sun)
        moon=np.array(moon)

        Sun=sun / np.linalg.norm(sun, axis=1, keepdims=True)
        Moon=moon / np.linalg.norm(moon, axis=1, keepdims=True)

        return Sun,Moon

        pass

    @staticmethod
    def angle_between(v1, v2):
        # 计算两个向量的点积
        dot_product = np.dot(v1, v2)

        # 计算两个向量的模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 计算夹角的余弦值
        cos_angle = dot_product / (norm_v1 * norm_v2)

        # 使用arccos函数计算夹角（弧度制）
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # 将弧度转换为角度
        angle_deg = np.degrees(angle_rad)

        return angle_deg



    def LOSz_vec(self):
        LOS=self.LOS
        QSA=self.QSA
        LOSz_vec={}
        z=np.array([0,0,1])

        for key in SCAID:
            R = quaternion.as_rotation_matrix(QSA[self.sat][key.value - 1])
            z_vec=[]
            for i in trange(len(self.Angletime[::60])):
            # for i in trange(100):
                LOS_scf=LOS[60*i,1:].reshape(3,3).T @ R.T
                result=np.dot(LOS_scf,z)
                z_vec.append(result)
            LOSz_vec[key.name]=np.array(z_vec)
        return LOSz_vec


    def GetAngle(self):
        sun,moon=self.ConfigPos_SunMoon()
        Losz=self.LOSz_vec()
        Sunangle={}
        Moonangle={}
        for key in SCAID:
            sunangle=[]
            moonangle=[]
            for i in trange(len(Losz[key.name][:,0])):
                v1=sun[i]
                v2=moon[i]
                v3=Losz[key.name][i]
                sunangle.append(self.angle_between(v3,v1))
                moonangle.append(self.angle_between(v3,v2))

            Sunangle[key.name]=np.array(sunangle)
            Moonangle[key.name]=np.array(moonangle)

        return Sunangle,Moonangle


        pass



    def saveAnglefile(self,DataDir='../SunMoonAngle'):
        try:
            Date=datetime.datetime.strptime(self.date,'%Y-%m-%d')
            Sun,Moon=self.GetAngle()
            time=self.LOS[:,0].T
            time=time[::60]
            angle={}
            for key in SCAID:
                angle[key.name]=np.vstack((time,Sun[key.name],Moon[key.name]))
            '''record the result'''
            fileDir = os.path.join(DataDir, '%s/%s/' %(Date.strftime('%Y-%m'),Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'Angle_Sun-Moon_%s_%s.h5' % (self.date, self.sat.name)
            with h5py.File(fileDir + '/' + filename, 'w') as f:
                for key, value in angle.items():
                    if isinstance(value, dict):
                        group = f.create_group(key)
                        for nested_key, nested_value in value.items():
                            group.create_dataset(nested_key, data=nested_value)
                    else:
                        # 对于其他类型的数据，直接创建数据集
                        f.create_dataset(key, data=value)

            print('the angle data have saved on %s-%s' % (self.date, self.sat.name))
        except:
            print('the day of data is empty on %s'%self.date)

        pass

    def ConfigAnglefile(self):
        Date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
        file='../SunMoonAngle/%s/%s/Angle_Sun-Moon_%s_%s.h5'\
             %(Date.strftime('%Y-%m'),Date.strftime('%Y-%m-%d'),Date.strftime('%Y-%m-%d'),self.sat.name)
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

    def plot(self,id:SCAID):
        Angle=self.ConfigAnglefile()
        x=Angle[id.name][0]
        y1=Angle[id.name][1]
        y2=Angle[id.name][2]
        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time = []
        for sec_x in x:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time.append(dt)
        fig, ax = plt.subplots()
        ax.plot(time,y1,label='Sun',linestyle='--')
        ax.plot(time,y2,label='moon',linestyle='--')
        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('angle(deg)')
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.grid(True, axis='y', linestyle='dashed')

        plt.legend(loc="upper right")

        plt.show()



    def ConfigPos_Sun1(self):
        sun=[]
        print('get the position of sun')
        for i in trange(len(self.Angletime)):
            itrs_sun=gp.demo1(self.Angletime[i])
            sun.append(itrs_sun)
            pass
        sun=np.array(sun)

        Sun=sun / np.linalg.norm(sun, axis=1, keepdims=True)

        return Sun

        pass


    def LOSz_vec1(self):
        LOS=self.LOS
        QSA=self.QSA
        LOSz_vec={}
        z=np.array([0,0,1])

        print('get the vector of LOSz')
        for key in SCAID:
            R = quaternion.as_rotation_matrix(QSA[self.sat][key.value - 1])
            z_vec=[]
            for i in trange(len(self.Angletime)):
            # for i in trange(100):
                LOS_scf=LOS[i,1:].reshape(3,3).T @ R.T
                result=np.dot(LOS_scf,z)
                z_vec.append(result)
            LOSz_vec[key.name]=np.array(z_vec)
        return LOSz_vec


    def GetAngle1(self):
        sun=self.ConfigPos_Sun1()
        Losz=self.LOSz_vec1()
        Sunangle={}
        for key in SCAID:
            sunangle=[]
            moonangle=[]
            for i in trange(len(Losz[key.name][:,0])):
                v1=sun[i]
                v3=Losz[key.name][i]
                sunangle.append(self.angle_between(v3,v1))

            Sunangle[key.name]=np.vstack((self.Angletime, np.array(sunangle)))

        return Sunangle


        pass

    def savesunfile(self,DataDir='../SunAngle'):
        try:
            Date=datetime.datetime.strptime(self.date,'%Y-%m-%d')
            Sun=self.GetAngle1()
            print(Date)
            '''record the result'''
            fileDir = os.path.join(DataDir, '%s/%s/' %(Date.strftime('%Y-%m'),Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'Angle_Sun-Moon_%s_%s.h5' % (self.date, self.sat.name)
            with h5py.File(fileDir + '/' + filename, 'w') as f:
                for key, value in Sun.items():
                    if isinstance(value, dict):
                        group = f.create_group(key)
                        for nested_key, nested_value in value.items():
                            group.create_dataset(nested_key, data=nested_value)
                    else:
                        # 对于其他类型的数据，直接创建数据集
                        f.create_dataset(key, data=value)

            print('the angle data have saved on %s-%s' % (self.date, self.sat.name))
        except:
            print('the day of data is empty on %s'%self.date)

        pass




if __name__=='__main__':


    # daylist = GeoMathKit.dayListByDay('2021-08-18', '2021-09-30')
    # for day in daylist:
    #     try:
    #         a=SunMoonAngle(sat=SatID.D,date=day.strftime('%Y-%m-%d'))
    #         a.saveAnglefile(DataDir='../SunMoonAngle')
    #     except:
    #         print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))

    # sun,moon=a.ConfigPos_SunMoon()
    # Losz_vec=a.LOSz_vec()
    # a = SunMoonAngle(sat=SatID.C, date='2020-04-01')
    # Sunangle,Moonangle=a.GetAngle()
    # a.saveAnglefile(DataDir='../SunMoonAngle')
    # a.plot(id=SCAID.No_3)
    # a.savesunfile(DataDir='../SunAngle')


    daylist = GeoMathKit.dayListByDay('2020-05-31', '2020-06-30')
    for day in daylist:
        try:
            a=SunMoonAngle(sat=SatID.C,date=day.strftime('%Y-%m-%d'))
            a.savesunfile(DataDir='E:/data/SunAngle')
        except:
            print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))
    pass