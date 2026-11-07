import matplotlib.dates as mdates
import numpy as np
from pysrc.core.geometry import GeoMathKit
from pysrc.core.settings import SatID, SCAID, Mission, Coord, Model
import matplotlib.patches as patches
import os
import datetime
from pysrc.analysis import hfn_analysis as HFNAnalysis
from pysrc.fusion.multi_sca import SCAweighting
from pysrc.attitude.pointing import Pointing
from pysrc.analysis.hfn_analysis import configReadEuler3
from tqdm import trange
import quaternion
import copy
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
import matplotlib.pyplot as plt
import h5py
from math import atan2, asin
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
from scipy.stats import norm
from matplotlib.pyplot import MultipleLocator

config = {
    "font.family": 'serif',
    # "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

class plot():
    def __init__(self,sat: SatID, date,SCA1A, TIM1B, CLK1B):
        self.sat = sat
        self.date = date
        self.__SCA1A = SCA1A
        self.__TIM1B = TIM1B
        self.__CLK1B = CLK1B
        self._QSA=self.ConfigSCF2SRF()
        self._LOS=self.ConfigIF2LOS(date)
        self.__r = None


    def ConfigIF2LOS(self,date):

        file='../result/product/GRACE_FO/RL04/L1B/%s/IF2LOS_%s_HUST01_%s.npy'%(date,date,self.sat.name)
        LOS=np.load(file)

        return LOS

    def get_TV_r(self):
        std, day = configReadEuler3(sat=self.sat,begindate=self.date,enddate=self.date)
        self.__r=np.vstack((std['No_1'],std['No_2'],std['No_3']))

        return self

    def get_prior_r(self):

        self.__r = np.array([[3, 3, 30], [3, 3, 30], [3, 3, 30]])
        print(self.__r)

        return self
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

    @staticmethod
    def ConfigRange_rate(date='2020-2'):
        Path_KBR1='../KBR_result/%s/Normal'%date
        Path_KBR2='../KBR_result/%s/GSCF-2'%date
        Path_KBR3='../KBR_result/%s/GSCF-3'%date
        N1 = len(os.listdir(Path_KBR1))
        N2 = len(os.listdir(Path_KBR2))
        N3 = len(os.listdir(Path_KBR3))
        for i in np.arange(N1):
            if i==0:
                load_path = Path_KBR1+'/RangeRate_{}.hdf5'.format(i)
                with h5py.File(load_path,'r') as f1:
                    data1=f1[list(f1.keys())[1]]
                    data1 = np.array(data1) * 1e6
                    data2=f1[list(f1.keys())[2]]
                    data2 = np.array(data2)
                    RMS_Set1 = np.vstack((data2,data1))
            else:
                load_path = Path_KBR1+'/RangeRate_{}.hdf5'.format(i)
                with h5py.File(load_path,'r') as f1:
                    data1=f1[list(f1.keys())[1]]
                    data1 = np.array(data1) * 1e6
                    data2=f1[list(f1.keys())[2]]
                    data2 = np.array(data2)
                    rms1 = np.vstack((data2,data1))
                    RMS_Set1=np.hstack((RMS_Set1,rms1))

        for j in np.arange(N2):
            if j == 0:
                load_path = Path_KBR2+'/RangeRate_{}.hdf5'.format(j)
                with h5py.File(load_path,'r') as f1:
                    data1=f1[list(f1.keys())[1]]
                    data1 = np.array(data1) * 1e6
                    data2=f1[list(f1.keys())[2]]
                    data2 = np.array(data2)
                    RMS_Set2 = np.vstack((data2,data1))
            else:
                load_path = Path_KBR2+'/RangeRate_{}.hdf5'.format(j)
                with h5py.File(load_path,'r') as f1:
                    data1=f1[list(f1.keys())[1]]
                    data1 = np.array(data1) * 1e6
                    data2=f1[list(f1.keys())[2]]
                    data2 = np.array(data2)
                    rms1 = np.vstack((data2,data1))
                    RMS_Set2=np.hstack((RMS_Set2,rms1))

        for k in np.arange(N3):
            if k == 0:
                load_path = Path_KBR3+'/RangeRate_{}.hdf5'.format(k)
                with h5py.File(load_path,'r') as f1:
                    data1=f1[list(f1.keys())[1]]
                    data1 = np.array(data1) * 1e6
                    data2=f1[list(f1.keys())[2]]
                    data2 = np.array(data2)
                    RMS_Set3 = np.vstack((data2,data1))
            else:
                load_path = Path_KBR3+'/RangeRate_{}.hdf5'.format(k)
                with h5py.File(load_path,'r') as f1:
                    data1=f1[list(f1.keys())[1]]
                    data1 = np.array(data1) * 1e6
                    data2=f1[list(f1.keys())[2]]
                    data2 = np.array(data2)
                    rms1 = np.vstack((data2,data1))
                    RMS_Set3=np.hstack((RMS_Set3,rms1))

        return RMS_Set1,RMS_Set2,RMS_Set3
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
            # print(valid)
            xp = xp[valid]
            yp = yp[valid]
            # print(xp)
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
        #  this should be refined by dividing the arcs
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


            scaValue[0] += V_int
            scaValue[1] += V_frac * 1e6

            scaValue[0][scaValue[1] < 0] -= 1
            scaValue[1][scaValue[1] < 0] += 1e6

            sca[key.name] = scaValue
            pass


        '''Step 3: Flip signs'''
        for id in SCAID:
            '''judge by the vector of quaternion'''
            a = sca[id.name][2:6, :].T
            quality = sca[id.name][6, :]

            m = [a[0, :]]

            for i in range(len(a[:, 0]) - 1):
                if np.dot(a[i + 1, 1:], m[-1][1:]) < 0 and quality[i] <= 6:
                    m.append(-a[i + 1, :])
                else:
                    m.append(a[i + 1, :])

            a = np.array(m)

            sca[id.name][2:6, :] = a.T


        # '''Step 4: Unified time series'''
        # SCA_scf={}
        # for key in SCAID:
        #     SCA1A=sca[key.name][0:,sca[key.name][6,:]<=6]
        #     starttime = SCA1A[0][0]
        #     time1 = SCA1A[0] - starttime + SCA1A[1] * 1e-6
        #     time2=self._LOS[:,0].T-starttime
        #     qa=quaternion.as_quat_array(SCA1A[2:6].T)
        #     new = quaternion.squad(R_in=qa, t_in=time1, t_out=time2)
        #     #The first data of interpolation has a significant error, which needs to be noted
        #     qc = quaternion.as_float_array(new).T
        #     qual=np.ones((1,len(qc[0])))
        #     SCA_scf[key.name]=np.vstack((time2+starttime,qc,qual))
        #
        #     pass

        return sca



    def SCA_SCF2SRF(self,SCA):
        """
        SCF--->SRF
        """
        data = SCA.copy()
        satID=self.sat
        for key in SCAID:
            QSA=self.ConfigSCF2SRF()
            qb = quaternion.as_quat_array(SCA[key.name][2:6, :].T)
            if key.name=='No_1':
                qa = qb * QSA[satID][0]
            elif key.name=='No_2':
                qa = qb * QSA[satID][1]
            elif key.name=='No_3':
                qa = qb * QSA[satID][2]
            else:
                pass

            qc = quaternion.as_float_array(qa).T
            data[key.name][2:6,:] = -qc

        return data


    @staticmethod
    def calculate_std_with_outliers(data):
        """
        After removing outliers, calculate the standard deviation
        """
        res1 = data
        res2 = res1[:, np.abs(res1[-1]) < 3 * np.std(res1[-1])]
        while len(res1[0]) != len(res2[0]):
            res1 = res2[:, np.abs(res2[-1]) < 3 * np.std(res2[-1])]
            res2 = res1[:, np.abs(res1[-1]) < 3 * np.std(res1[-1])]

        res_std = np.std(res2[-1])



        return res2,res_std

    def tag_outliers(self,sca_R):
        """Mark the location of outliers in SCA data(change the quality label to 254)"""

        time2 = HFNAnalysis.outliers_index_time(sat=self.sat, begindate=self.date, enddate=self.date)
        for key in SCAID:
            time1=sca_R[key.name][0]+sca_R[key.name][1]* 1e-6
            time1_2d=np.vstack((time1,sca_R[key.name][6]))
            diff=np.abs(time1_2d[0][:,np.newaxis]-time2[key.name])
            within_range = (diff >= -0.5) & (diff <= 0.5)  # 判断是否落在区间内
            less_than_6 = sca_R[key.name][6] <= 6  # 判断第二行元素是否小于6
            within_and_less_than_6 = np.any(within_range.T & less_than_6, axis=0)  # 找到同时满足两个条件的列索引
            sca_R[key.name][6, within_and_less_than_6] = 254

        return sca_R

    def Complete_data(self,sca_R):
        """Utilize LOS data to supplement SCA data, facilitating subsequent interpolation"""

        sca_R=sca_R.T
        time1=sca_R[0,:]
        time2=self._LOS[:,0].T
        missing_indices = np.where(~np.isin(time2, time1))[0]
        max=np.max(time1)

        for index in missing_indices:
            missing_time = time2[index]
            if missing_time<max:
                a_index = np.where(sca_R[0] > missing_time)[0][0]
            else:
                a_index = len(sca_R[0])-1
            LOS=self._LOS[index,1:]
            b=quaternion.from_rotation_matrix(LOS.reshape(3, 3).T)
            b=quaternion.as_float_array(b)
            sca_R = np.insert(sca_R, a_index, [missing_time,0 ,b[0],b[1],b[2],b[3]], axis=1)

        pass

        return sca_R

    def q_combine3(self):
        """A quaternion combination algorithm based on least squares method incorporating the factor of sun angle"""
        sca_C=self.TimeTagTransformation()

        sca_R = self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))


        QSA = self._QSA
        a=SCAweighting(sca_R,QSA[self.sat],A=None,date=self.date,sat=self.sat)
        sca_new=a.combine_2t()
        sca_new=np.array(sca_new)
        starttime = sca_new[0][0]
        sca_new=self.Complete_data(sca_R=sca_new)
        sca_new=sca_new.T
        time1 = sca_new[:,0].T - starttime + sca_new[:,1].T * 1e-6
        time2=self._LOS[:,0].T-starttime
        qa=quaternion.as_quat_array(sca_new[:,2:6])
        new = quaternion.squad(R_in=qa, t_in=time1, t_out=time2)
        #The first data of interpolation has a significant error, which needs to be noted
        qc = quaternion.as_float_array(new).T
        sca_combine=np.vstack((self._LOS[:,0].T,qc)).T


        return sca_combine



    def q_combine2(self):
        """A quaternion combination algorithm based on least squares method incorporating the factor of outliers"""

        sca_C=self.TimeTagTransformation()

        SCA_R = self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))

        sca_R=self.tag_outliers(sca_R=copy.deepcopy(SCA_R))

        QSA = self._QSA
        a=SCAweighting(sca_R,QSA[self.sat],self.__r)
        sca_new=a.combine_2t()
        sca_new=np.array(sca_new)
        sca_new=self.Complete_data(sca_R=sca_new)
        sca_new=sca_new.T
        starttime = sca_new[0][0]
        time1 = sca_new[:,0].T - starttime + sca_new[:,1].T * 1e-6
        time2=self._LOS[:,0].T-starttime
        qa=quaternion.as_quat_array(sca_new[:,2:6])
        new = quaternion.squad(R_in=qa, t_in=time1, t_out=time2)
        #The first data of interpolation has a significant error, which needs to be noted
        qc = quaternion.as_float_array(new).T
        sca_combine=np.vstack((self._LOS[:,0].T,qc)).T


        return sca_combine

    def ADS_HS_time(self):
        """
        Time point for obtaining effective attitude data: 0/1/2/3
        """
        sca_C=self.TimeTagTransformation()

        SCA_R = self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))

        sca_R=self.tag_outliers(sca_R=copy.deepcopy(SCA_R))

        time={}
        time['number_0'] = np.array([])
        time['number_1'] = np.array([])
        time['number_2'] = np.array([])
        time['number_3'] = np.array([])

        for i in trange(len(sca_R[SCAID.No_1.name][1]),desc="Obtain the ADS_HS_time\n"):
            if sca_R[SCAID.No_1.name][6,i]==254 and sca_R[SCAID.No_2.name][6,i]==254 and sca_R[SCAID.No_3.name][6,i]==254:
                time['number_0']=np.append(time['number_0'], sca_R[SCAID.No_1.name][0, i]+sca_R[SCAID.No_1.name][1, i] * 1e-6)
            elif sca_R[SCAID.No_1.name][6,i]!=254 and sca_R[SCAID.No_1.name][6,i]<=6 \
                    and sca_R[SCAID.No_2.name][6,i]==254 and sca_R[SCAID.No_3.name][6,i]==254:
                time['number_1']=np.append(time['number_1'], sca_R[SCAID.No_1.name][0, i] + sca_R[SCAID.No_1.name][1, i] * 1e-6)
            elif sca_R[SCAID.No_1.name][6,i]==254 and sca_R[SCAID.No_2.name][6,i]!=254 \
                    and sca_R[SCAID.No_2.name][6,i]<=6 and sca_R[SCAID.No_3.name][6,i]==254:
                time['number_1']=np.append(time['number_1'], sca_R[SCAID.No_1.name][0, i] + sca_R[SCAID.No_1.name][1, i] * 1e-6)
            elif sca_R[SCAID.No_1.name][6,i]==254 and sca_R[SCAID.No_2.name][6,i]==254 \
                    and sca_R[SCAID.No_3.name][6,i]!=254 and sca_R[SCAID.No_3.name][6,i]<=6:
                time['number_1']=np.append(time['number_1'], sca_R[SCAID.No_1.name][0, i] + sca_R[SCAID.No_1.name][1, i] * 1e-6)
            elif sca_R[SCAID.No_1.name][6,i]!=254 and sca_R[SCAID.No_2.name][6,i]!=254 \
                    and sca_R[SCAID.No_3.name][6,i]==254 and sca_R[SCAID.No_1.name][6,i]<=6 and sca_R[SCAID.No_2.name][6,i]<=6:
                time['number_2']=np.append(time['number_2'], sca_R[SCAID.No_1.name][0, i] + sca_R[SCAID.No_1.name][1, i] * 1e-6)
            elif sca_R[SCAID.No_1.name][6,i]==254 and sca_R[SCAID.No_2.name][6,i]!=254 \
                    and sca_R[SCAID.No_3.name][6,i]!=254 and sca_R[SCAID.No_2.name][6,i]<=6 and sca_R[SCAID.No_3.name][6,i]<=6:
                time['number_2']=np.append(time['number_2'], sca_R[SCAID.No_1.name][0, i] + sca_R[SCAID.No_1.name][1, i] * 1e-6)
            elif sca_R[SCAID.No_1.name][6,i]!=254 and sca_R[SCAID.No_2.name][6,i]==254 \
                    and sca_R[SCAID.No_3.name][6,i]!=254 and sca_R[SCAID.No_1.name][6,i]<=6 and sca_R[SCAID.No_3.name][6,i]<=6:
                time['number_2']=np.append(time['number_2'], sca_R[SCAID.No_1.name][0, i] + sca_R[SCAID.No_1.name][1, i] * 1e-6)
            elif sca_R[SCAID.No_1.name][6,i]<=6 and sca_R[SCAID.No_2.name][6,i]<=6 and sca_R[SCAID.No_3.name][6,i]<=6:
                time['number_3']=np.append(time['number_3'], sca_R[SCAID.No_1.name][0, i] + sca_R[SCAID.No_1.name][1, i] * 1e-6)
            else:
                pass

        row_one = np.ones(time['number_1'].shape[0], dtype=time['number_1'].dtype)
        row_two = 2*np.ones(time['number_2'].shape[0], dtype=time['number_1'].dtype)
        row_three = 3*np.ones(time['number_3'].shape[0], dtype=time['number_1'].dtype)

        time['number_1']=np.vstack((time['number_1'],row_one))
        time['number_2'] = np.vstack((time['number_2'], row_two))
        time['number_3'] = np.vstack((time['number_3'], row_three))

        return time

        pass

    def q_combine_Normal(self,date='2020-04-08'):
        sca_C=self.TimeTagTransformation()
        sca_R = self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))
        QSA = self._QSA
        a=SCAweighting(SCA=sca_R,QSA=QSA[self.sat],A=self.__r)
        sca_new=a.combine_2t()
        q=np.array(sca_new)
        Date = datetime.datetime.strptime(date, '%Y-%m-%d')
        DataDir = 'D:/data'
        print(len(q[:, 0]) - 1)
        '''Notice: q:N*6'''
        fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
        isExists = os.path.exists(fileDir)
        if not isExists:
            os.makedirs(fileDir)
        filename = 'SCA1B_%s_%s_04.txt' % (self.date, self.sat.name)
        with open(fileDir + filename, 'w') as file:
            file.write('header: \n')
            file.write('num_records: %d \n' % (len(q[:, 0]) - 1))
            file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', 'Normal'))
            file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Pan Xudong'))
            file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'D202480136@hust.edu.cn'))
            file.write('# End of YAML header \n')

            for index in range(len(q[:, 0]) - 1):
                file.write('%d  %d %3s %3s   %20.16f  %20.16f  %20.16f %20.16f %3d\n' %
                           (q[index + 1, 0], q[index + 1, 1], self.sat.name, 'G', q[index + 1, 2],
                            q[index + 1, 3], q[index + 1, 4], q[index + 1, 5], q[index+1, 6]))

        print('the data have saved on %s-%s' % (self.date, self.sat.name))


        pass


    def q_combine1(self):
        """Quaternion Combination Algorithm Based on Least Squares Method"""

        sca_C=self.TimeTagTransformation()

        sca_R = self.SCA_SCF2SRF(SCA=copy.deepcopy(sca_C))

        QSA = self._QSA
        a=SCAweighting(SCA=sca_R,QSA=QSA[self.sat],A=self.__r)
        sca_new=a.combine_2t()
        sca_new=np.array(sca_new)
        starttime = sca_new[0][0]
        time1 = sca_new[:,0].T - starttime + sca_new[:,1].T * 1e-6
        time2=self._LOS[:,0].T-starttime
        qa=quaternion.as_quat_array(sca_new[:,2:6])
        new = quaternion.squad(R_in=qa, t_in=time1, t_out=time2)
        #The first data of interpolation has a significant error, which needs to be noted
        qc = quaternion.as_float_array(new).T
        sca_combine=np.vstack((self._LOS[:,0].T,qc)).T


        return sca_combine

    def Get_angle1(self,TV_r=False):
        """Obtain residual Eular angles through small angle approximation,
        where the residual is the quaternion of LOS divided by the quaternion of SCA.
         Notice:the quaternion combination method doesn't take into account the factor of outliers"""

        LOS=self._LOS
        if TV_r==True:
            sca_new=self.get_TV_r().q_combine1()
        else:
            sca_new=self.get_prior_r().q_combine1()
        Res=[]

        for i in trange(len(LOS[:, 0])):
            q=quaternion.as_quat_array(sca_new[:,1:5])
            R_T=quaternion.as_rotation_matrix(q[i]).T
            res = LOS[i, 1:].reshape(3, 3).T @ R_T
            a=quaternion.from_rotation_matrix(res)
            Res.append(a)

        Res=quaternion.as_float_array(Res)
        angle=np.vstack((LOS[:,0].T,Res[:,1:4].T*3600*2*180/np.pi))


        return angle


    def Get_angle2(self,TV_r=True):
        """Obtain residual Eular angles through small angle approximation,
        where the residual is the quaternion of LOS divided by the quaternion of SCA.
         Notice:the quaternion combination method takes into account the factor of outliers"""
        LOS=self._LOS
        if TV_r==True:
            sca_new=self.get_TV_r().q_combine2()
        else:
            sca_new=self.get_prior_r().q_combine2()
        Res=[]

        for i in trange(len(LOS[:, 0])):
            q=quaternion.as_quat_array(sca_new[:,1:5])
            R_T=quaternion.as_rotation_matrix(q[i]).T
            res = LOS[i, 1:].reshape(3, 3).T @ R_T
            a=quaternion.from_rotation_matrix(res)
            Res.append(a)

        Res=quaternion.as_float_array(Res)
        angle=np.vstack((LOS[:,0].T,Res[:,1:4].T*3600*2*180/np.pi))


        return angle


    def get_angle(self,A=np.array([[3,3,30],[3,3,30],[3,3,30]])):
        """Obtain residual Eular angles through residual rotation matrix,
        where the residual is the quaternion of LOS divided by the quaternion of SCA """

        sca_new=self.get_prior_r().q_combine1()

        roll1=[]
        pitch1=[]
        yaw1=[]


        for i in trange(len(sca_new[:, 0])):
            q=quaternion.as_quat_array(sca_new[:,1:5])
            R=quaternion.as_rotation_matrix(q[i])
            roll1.append(atan2(R[2][1], R[2][2]))
            pitch1.append(-asin(R[2][0]))
            yaw1.append( atan2(R[1][0], R[0][0]))



        Res1 = np.vstack((sca_new[:,0].T,roll1,pitch1,yaw1))



        return Res1

    @staticmethod
    def plotPSD(fs=1,data1=None,data2=None):
        """
        Draw PSD graphs of two datas and compare them
        """


        # time1 = self._LOS[:,0].T
        # angle1=np.interp(time1,data1[0],data1[1])

        angle1=data1[1]


        # wn = 2 * 0.02 / 1
        # order = 4  # filter order
        # b, a = signal.butter(order, wn, btype='high', fs=1)
        # angle1 = signal.lfilter(b, a, angle1)

        # time2 = self._LOS[:,0].T
        # angle2=np.interp(time2,data2[0],data2[1])

        angle2=data2[1]

        # wn = 2 * 0.02 / 1
        # order = 4  # filter order
        # b, a = signal.butter(order, wn, btype='high', fs=1)
        # angle2 = signal.lfilter(b, a, angle2)

        fig = plt.figure()
        fig2 = plt.figure()

        axx = fig2.add_subplot(1, 1, 1)

        psd1, freq1 = plt.psd(angle1, Fs=fs, NFFT=len(angle1))
        psd2, freq2 = plt.psd(angle2, Fs=fs, NFFT=len(angle2))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')

        psd1 = np.sqrt(2 * psd1)
        psd2 = np.sqrt(2 * psd2)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.plot(freq1, psd1, label='pre', lw=0.5,
                alpha=0.8)
        ax.plot(freq2, psd2, label='aft', lw=0.5,
                alpha=0.9)

        ax.legend(ncol=3)
        ax.grid(True, linestyle='dashed')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')

        # 设置x轴刻度范围，从0.01开始
        # start_freq = 0.01
        # ax.set_xlim(start_freq, freq1[-1])
        # ax.set_ylim(bottom=None, top=200)

        fig.align_ylabels()
        fig.align_xlabels()

        plt.show()

        pass


    @staticmethod
    def plotPSD_Threedata(fs=1,data1=None,data2=None, data3=None):
        """
        Draw PSD graphs of two datas and compare them
        """


        # time1 = self._LOS[:,0].T
        # angle1=np.interp(time1,data1[0],data1[1])

        angle1=data1[1]


        # wn = 2 * 0.02 / 1
        # order = 4  # filter order
        # b, a = signal.butter(order, wn, btype='high', fs=1)
        # angle1 = signal.lfilter(b, a, angle1)

        # time2 = self._LOS[:,0].T
        # angle2=np.interp(time2,data2[0],data2[1])

        angle2=data2[1]

        angle3=data3[1]

        # wn = 2 * 0.02 / 1
        # order = 4  # filter order
        # b, a = signal.butter(order, wn, btype='high', fs=1)
        # angle2 = signal.lfilter(b, a, angle2)

        fig = plt.figure(figsize=(8,8))
        fig2 = plt.figure()

        axx = fig2.add_subplot(1, 1, 1)

        psd1, freq1 = plt.psd(angle1, Fs=fs, NFFT=len(angle1))
        psd2, freq2 = plt.psd(angle2, Fs=fs, NFFT=len(angle2))
        psd3, freq3 = plt.psd(angle3, Fs=fs, NFFT=len(angle3))

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')

        psd1 = np.sqrt(2 * psd1)
        psd2 = np.sqrt(2 * psd2)
        psd3 = np.sqrt(2 * psd3)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.plot(freq1, psd1, label='Normal', lw=0.5,
                alpha=0.8)
        ax.plot(freq2, psd2, label='GSCF-2', lw=0.5,
                alpha=0.9)
        ax.plot(freq3, psd3, label='GSCF-3', lw=0.5,
                alpha=0.9)

        ax.legend(ncol=3)
        ax.grid(True, linestyle='dashed')
        font = {'size': 20}
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

        plt.show()
        pass

    @staticmethod
    def plotPSD_single(data1,coord:Coord):
        """
        Draw a PSD diagram of the data
        """


        # time1 = self._LOS[:,0].T
        # angle1=np.interp(time1,data1[0],data1[1])

        angle1=data1[1]


        # wn = 2 * 0.02 / 1
        # order = 4  # filter order
        # b, a = signal.butter(order, wn, btype='high', fs=1)
        # angle1 = signal.lfilter(b, a, angle1)

        # time2 = self._LOS[:,0].T
        # angle2=np.interp(time2,data2[0],data2[1])


        fig = plt.figure()
        fig2 = plt.figure()

        axx = fig2.add_subplot(1, 1, 1)


        psd1, freq1 = plt.psd(angle1, Fs=1, NFFT=len(angle1))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')

        psd1 = np.sqrt(2 * psd1)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.plot(freq1, psd1, label='diff-'+'%s'%(coord.name), lw=0.5,
                alpha=0.8)


        # for freq in np.arange(0.00017, 0.1, 0.00017):
        #     index = np.abs(freq1 - freq).argmin()
        #     ax.plot(freq1[index], psd1[index], 'ro', markersize=3)

        # 自定义矩形的宽度
        rectangle_width = 0.00005
        height = psd1.max()

        # 画出频率为 0.0001736 的倍数的点附近的条带状矩形
        for freq in np.arange(0.0001736, 0.0014, 0.0001736):
            # 计算矩形的位置和大小
            index = np.abs(freq1 - freq).argmin()
            x = freq1[index] - rectangle_width / 2
            width = rectangle_width
            # 绘制矩形
            rect = patches.Rectangle((x, 0), width, height, alpha=0.1, color='r')
            ax.add_patch(rect)

        ax.legend(ncol=3,fontsize='large')
        ax.grid(True, linestyle='dashed')
        # ax.set_xlabel('Frequency [Hz]')
        # ax.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)')

        # 设置x轴刻度范围，从0.01开始
        # start_freq = 0.01
        # ax.set_xlim(start_freq, freq1[-1])
        # ax.set_ylim(bottom=None, top=200)

        # ax.tick_params(axis='x', labelbottom=False)
        # ax.tick_params(axis='y', labelleft=False)

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

        spines = ax.spines

        spines['top'].set_linewidth(1.5)
        spines['bottom'].set_linewidth(1.5)
        spines['left'].set_linewidth(1.5)
        spines['right'].set_linewidth(1.5)

        fig.align_ylabels()
        fig.align_xlabels()

        fig.savefig('../paper_figure/Fig20.png', dpi=300)
        plt.show()

        pass


    def PlotEuler(self,res1,res2):
        '''Obtain a time series Eular angle graph in the form of dates:hours/days/years'''
        # res1x,res1x_std=self.calculate_std_with_outliers(res1[[0,1],:])
        # res1y,res1y_std=self.calculate_std_with_outliers(res1[[0,2],:])
        # res1z,res1z_std=self.calculate_std_with_outliers(res1[[0,3],:])
        #
        # print([res1x_std,res1y_std,res1z_std])
        #
        # res2x,res2x_std=self.calculate_std_with_outliers(res2[[0,1],:])
        # res2y,res2y_std=self.calculate_std_with_outliers(res2[[0,2],:])
        # res2z,res2z_std=self.calculate_std_with_outliers(res2[[0,3],:])
        #
        # print([res2x_std, res2y_std, res2z_std])

        res1x=res1[[0,1],:]
        res1y=res1[[0,2],:]
        res1z=res1[[0,3],:]


        res2x=res2[[0,1],:]
        res2y=res2[[0,2],:]
        res2z=res2[[0,3],:]


        seconds_x1=res1x[0,:]
        x1=res1x[1,:]
        seconds_x2=res2x[0,:]
        x2=res2x[1,:]
        deltax=x2-x1

        label_x1='roll-pre'
        label_x2 = 'roll-aft'
        label_deltax='Difference_x'

        seconds_y1=res1y[0,:]
        y1=res1y[1,:]
        seconds_y2=res2y[0,:]
        y2=res2y[1,:]
        deltay=y2-y1
        label_y1='pitch-pre'
        label_y2 = 'pitch-aft'
        label_deltay = 'Difference_y'

        seconds_z1=res1z[0,:]
        z1=res1z[1,:]
        seconds_z2=res2z[0,:]
        z2=res2z[1,:]
        deltaz=z2-z1
        # avg_z=Res[scaID.name][6,:]
        label_z1='yaw-pre'
        label_z2 = 'yaw-aft'
        label_deltaz = 'Difference_z'

        base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
        # 转换为日期并打印
        time_x1 = []
        time_y1 = []
        time_z1 = []
        time_x2 = []
        time_y2 = []
        time_z2 = []
        for sec_x in seconds_x1:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x1.append(dt)

        for sec_y in seconds_y1:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y1.append(dt)

        for sec_z in seconds_z1:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z1.append(dt)

        for sec_x in seconds_x2:
            dt = base_time + datetime.timedelta(seconds=sec_x)
            time_x2.append(dt)

        for sec_y in seconds_y2:
            dt = base_time + datetime.timedelta(seconds=sec_y)
            time_y2.append(dt)

        for sec_z in seconds_z2:
            dt = base_time + datetime.timedelta(seconds=sec_z)
            time_z2.append(dt)

        fig, ax = plt.subplots()

        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)
        # ax.scatter(time_x1, x1, s=0.1, color='red', label=label_x1, alpha=0.5)
        # ax.scatter(time_x2, x2, s=0.1, color='blue', label=label_x2, alpha=0.5)
        ax.scatter(time_x2, deltax, s=0.1, color='green', label=label_deltax, alpha=0.5)


        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        plt.ylim(-20, 20)
        # 设置横坐标的日期格式为 "%y-%m-%d"
        # date_format = mdates.DateFormatter('%y-%m-%d')
        # ax.xaxis.set_major_formatter(date_format)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.grid(True, axis='y', linestyle='dashed')
        plt.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

        spines = ax.spines

        spines['top'].set_linewidth(1.5)
        spines['bottom'].set_linewidth(1.5)
        spines['left'].set_linewidth(1.5)
        spines['right'].set_linewidth(1.5)

        fig.align_ylabels()
        fig.align_xlabels()

        fig.savefig('../paper_figure/figure6-1.png', dpi=300)

        plt.show()


        fig, ax = plt.subplots()

        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)
        # ax.scatter(time_y1, y1, s=0.1, color='red', label=label_y1, alpha=0.5)
        # ax.scatter(time_y2, y2, s=0.1, color='blue', label=label_y2, alpha=0.5)
        ax.scatter(time_y2, deltay, s=0.1, color='green', label=label_deltay, alpha=0.5)


        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        plt.ylim(-20, 20)
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.grid(True, axis='y', linestyle='dashed')
        plt.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

        spines = ax.spines

        spines['top'].set_linewidth(1.5)
        spines['bottom'].set_linewidth(1.5)
        spines['left'].set_linewidth(1.5)
        spines['right'].set_linewidth(1.5)

        fig.align_ylabels()
        fig.align_xlabels()

        fig.savefig('../paper_figure/figure6-3.png', dpi=300)

        plt.show()


        fig, ax = plt.subplots()
        # ax.plot(time, a)
        color_x = (31 / 255, 119 / 255, 180 / 255)
        color_y = (229 / 255, 114 / 255, 12 / 255)
        color_z = (35 / 255, 142 / 255, 35 / 255)
        # ax.scatter(time_z1, z1, s=0.1, color='red', label=label_z1, alpha=0.5)
        # ax.scatter(time_z2, z2, s=0.1, color='blue', label=label_z2, alpha=0.5)
        ax.scatter(time_z2, deltaz, s=0.1, color='green', label=label_deltaz, alpha=0.5)

        ax.set_xlabel('Time(hours)')
        ax.set_ylabel('res_angle(arcsec)')
        plt.ylim(-20, 20)
        # 设置横坐标的日期格式为 "%y-%m-%d"
        date_format = mdates.DateFormatter('%y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.grid(True, axis='y', linestyle='dashed')
        plt.legend(loc='upper right')

        ax.tick_params(direction='in', top=True, bottom=True, left=True)
        ax.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
        ax.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

        spines = ax.spines

        spines['top'].set_linewidth(1.5)
        spines['bottom'].set_linewidth(1.5)
        spines['left'].set_linewidth(1.5)
        spines['right'].set_linewidth(1.5)

        fig.align_ylabels()
        fig.align_xlabels()

        fig.savefig('../paper_figure/figure6-5.png', dpi=300)




        plt.show()
        pass

        #plot the diffience of prior weight and analytical weight of PSD figure
        date1 = np.vstack((seconds_x1, deltax))
        self.plotPSD_single(date1,coord=Coord.x)
        date2 = np.vstack((seconds_y1, deltay))
        self.plotPSD_single(date2,coord=Coord.y)
        date3 = np.vstack((seconds_z1, deltaz))
        self.plotPSD_single(date3,coord=Coord.z)

        pass

    def SavingFusionProducts_h5(self, model: Model):
        """
        :param
        1.model: 0/1/2 ;'0' for q_combine1 (apriority), '1' for q_combine1 (time-varying) and '2' for q_combine2
        2.r: The matrix of star cameras containing covariance information

        :role: saving the fusion products of star cameras in h5 format
        """
        DataDir='E:/data/FusionProduct/GRACE_FO/RL04'
        Date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
        if model.value==0:
            q=self.get_prior_r().q_combine1()
            '''Notice: q:N*5'''
            fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'SCA1B_HUST_%s_%s_%s.h5' % (self.date, self.sat.name, model.name)
            with h5py.File(fileDir + '/' + filename, 'w') as f:
                if isinstance(q, np.ndarray):
                    f.create_dataset(fileDir + '/' + filename, data=q)
                else:
                    for key, value in q.items():
                        if isinstance(value, dict):
                            group = f.create_group(key)
                            for nested_key, nested_value in value.items():
                                group.create_dataset(nested_key, data=nested_value)
                        else:
                            # 对于其他类型的数据，直接创建数据集
                            f.create_dataset(key, data=value)

            print('the data have saved on %s-%s_%s' % (self.date, self.sat.name, model.name))

        elif model.value==1:
            q=self.get_TV_r().q_combine1()
            '''Notice: q:N*5'''
            fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'SCA1B_HUST_%s_%s_%s.h5' % (self.date, self.sat.name, model.name)
            with h5py.File(fileDir + '/' + filename, 'w') as f:
                if isinstance(q, np.ndarray):
                    f.create_dataset(fileDir + '/' + filename, data=q)
                else:
                    for key, value in q.items():
                        if isinstance(value, dict):
                            group = f.create_group(key)
                            for nested_key, nested_value in value.items():
                                group.create_dataset(nested_key, data=nested_value)
                        else:
                            # 对于其他类型的数据，直接创建数据集
                            f.create_dataset(key, data=value)

            print('the data have saved on %s-%s_%s' % (self.date, self.sat.name, model.name))

        elif model.value==2:
            q=self.get_TV_r().q_combine2()
            '''Notice: q:N*5'''
            fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'SCA1B_HUST_%s_%s_%s.h5' % (self.date, self.sat.name, model.name)
            with h5py.File(fileDir + '/' + filename, 'w') as f:
                if isinstance(q, np.ndarray):
                    f.create_dataset(fileDir + '/' + filename, data=q)
                else:
                    for key, value in q.items():
                        if isinstance(value, dict):
                            group = f.create_group(key)
                            for nested_key, nested_value in value.items():
                                group.create_dataset(nested_key, data=nested_value)
                        else:
                            # 对于其他类型的数据，直接创建数据集
                            f.create_dataset(key, data=value)

            print('the data have saved on %s-%s_%s' % (self.date, self.sat.name, model.name))

        else:
            pass

        pass


    def SavingFusionProducts_txt(self, model: Model):
        """
        :param
        1.model: 0/1/2 ;'0' for q_combine1 (apriority), '1' for q_combine1 (time-varying) and '2' for q_combine2
        2.r: The matrix of star cameras containing covariance information

        :role: saving the fusion products of star cameras in h5 format
        """
        # DataDir='E:/data/FusionProduct/GRACE_FO/RL04'
        Date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
        if model.value==0:
            print('Normal')
            DataDir = 'D:/data/FusionProduct/GRACE_FO/RL04/Normal'
            q=self.get_prior_r().q_combine1()
            print(len(q[:,0])-1)
            '''Notice: q:N*5'''
            fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'SCA1B_%s_%s_04.txt' % (self.date, self.sat.name)
            with open(fileDir + filename, 'w') as file:
                file.write('header: \n')
                file.write('num_records: %d \n' % (len(q[:,0])-1))
                file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
                file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
                file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', model.name))
                file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Pan Xudong'))
                file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'M202270125@hust.edu.cn'))
                file.write('# End of YAML header \n')

                for index in range(len(q[:,0])-1):
                    file.write('%d  %3s %3s   %20.16f  %20.16f  %20.16f %20.16f\n' %
                               (q[index+1,0], self.sat.name, 'G', q[index+1,1],
                                q[index+1,2], q[index+1,3], q[index+1,4]))

            print('the data have saved on %s-%s_%s' % (self.date, self.sat.name, model.name))

        elif model.value==1:
            print('GSCF-1')
            DataDir = 'D:/data/FusionProduct/GRACE_FO/RL04/GSCF-1'
            q = self.get_TV_r().q_combine1()
            '''Notice: q:N*5'''
            print(len(q[:, 0]) - 1)
            fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'SCA1B_%s_%s_04.txt' % (self.date, self.sat.name)
            with open(fileDir + filename, 'w') as file:
                file.write('header: \n')
                file.write('num_records: %d \n' % (len(q[:, 0]) - 1))
                file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
                file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
                file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', model.name))
                file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Pan Xudong'))
                file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'M202270125@hust.edu.cn'))
                file.write('# End of YAML header \n')

                for index in range(len(q[:,0])-1):
                    file.write('%d  %3s %3s   %20.16f  %20.16f  %20.16f %20.16f\n' %
                               (q[index+1,0], self.sat.name, 'G', q[index+1,1],
                                q[index+1,2], q[index+1,3], q[index+1,4]))

            print('the data have saved on %s-%s_%s' % (self.date, self.sat.name, model.name))

        elif model.value==2:
            print('GSCF-2')
            DataDir = 'D:/data/FusionProduct/GRACE_FO/RL04/GSCF-2'
            q = self.get_TV_r().q_combine2()
            '''Notice: q:N*5'''
            print(len(q[:, 0]) - 1)
            fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'SCA1B_%s_%s_04.txt' % (self.date, self.sat.name)
            with open(fileDir + filename, 'w') as file:
                file.write('header: \n')
                file.write('num_records: %d \n' % (len(q[:, 0]) - 1))
                file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
                file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
                file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', model.name))
                file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Pan Xudong'))
                file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'M202270125@hust.edu.cn'))
                file.write('# End of YAML header \n')

                for index in range(len(q[:,0])-1):
                    file.write('%d  %3s %3s   %20.16f  %20.16f  %20.16f %20.16f\n' %
                               (q[index+1,0],  self.sat.name, 'G', q[index+1,1],
                                q[index+1,2], q[index+1,3], q[index+1,4]))

            print('the data have saved on %s-%s_%s' % (self.date, self.sat.name, model.name))

        elif model.value==3:
            print('GSCF-3')
            DataDir = 'D:/data/FusionProduct/GRACE_FO/RL04/GSCF-3'
            q = self.q_combine3()
            '''Notice: q:N*5'''
            print(len(q[:, 0]) - 1)
            fileDir = os.path.join(DataDir, '%s/%s/' % (Date.strftime('%Y-%m'), Date.strftime('%Y-%m-%d')))
            isExists = os.path.exists(fileDir)
            if not isExists:
                os.makedirs(fileDir)
            filename = 'SCA1B_%s_%s_04.txt' % (self.date, self.sat.name)
            with open(fileDir + filename, 'w') as file:
                file.write('header: \n')
                file.write('num_records: %d \n' % (len(q[:, 0]) - 1))
                file.write('%-31s%3s%-31s \n' % ('PRODUCT NAME', ':', 'SCA Level-1B'))
                file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY', ':', 'HUST-CGE'))
                file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION', ':', model.name))
                file.write('%-31s%3s%-31s \n' % ('Author', ':', 'Pan Xudong'))
                file.write('%-31s%3s%-31s \n' % ('Contact', ':', 'M202270125@hust.edu.cn'))
                file.write('# End of YAML header \n')

                for index in range(len(q[:, 0]) - 1):
                    file.write('%d  %3s %3s   %20.16f  %20.16f  %20.16f %20.16f\n' %
                               (q[index + 1, 0], self.sat.name, 'G', q[index + 1, 1],
                                q[index + 1, 2], q[index + 1, 3], q[index + 1, 4]))

            print('the data have saved on %s-%s_%s' % (self.date, self.sat.name, model.name))
            pass

        else:
            print('the input of argument (model) is wrong')



def demo():
    day='2020-02-12'
    date = day
    sat=SatID.C

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    sca1B = L1B.getSCA(sat)
    a = plot(sat=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

    # r1=np.array([[3,3,30],[3,3,30],[3,3,30]])
    # sca_new1 = a.q_combine1(r1)

    # q1=quaternion.as_quat_array(sca1B[1:,:].T)
    # q2=quaternion.as_quat_array(sca_new1[:,1:])
    #
    # q1_inv=np.array([q.inverse() for q in q1])
    #
    # Res_w=(1-np.abs(quaternion.as_float_array(q1_inv*q2)[:,0]))*2*3600*180/np.pi

    res1=a.Get_angle1(TV_r=True)


    # r2 = np.array([[2.92, 1.65, 28.73], [2.32, 1.76, 25.63], [2.45, 2.12, 23.38]])#'2020-04-01'
    # r2 = np.array([[3,3,30],[3,3,30],[3,3,30]])
    # r2=np.array([[2.52,1.42,25.19],[2.85,2.09,29.72],[2.18,1.89,21.67]])
    # sca_new2 = a.q_combine2(r2)
    res2=a.Get_angle2(TV_r=True)

    # outliers_index =HFNAnalysis.outliers_index(sat=sat,begindate=date,enddate=date)

    # index=np.vstack((outliers_index['No_1'],outliers_index['No_2'],outliers_index['No_3']))
    #
    # index=np.all(index,axis=0)

    data1 = res1
    data2 = res2


    # r1 = [3, 3, 30]
    # r2 = [5, 20, 50]
    #
    # res1=a.get_angle(r1)
    # res2=a.get_angle(r2)
    #
    # data1=res1[0:2]
    # data2 = res2[0:2]
    a.PlotEuler(data1,data2)



    a.plotPSD(data1[[0, 1], 1:], data2[[0, 1], 1:])
    # a.plotPSD(data1[[0, 2], 1:], data2[[0, 2], 1:])
    # a.plotPSD(data1[[0, 3], 1:], data2[[0, 3], 1:])

    pass


def plot_paper_Euler1(res1, number_time):
    """
    plot the residual angle with ADS_HS time
    """

    seconds = res1[0, :]


    label_x1 = 'roll-Normal/GSCF-1'
    label_x2 = 'roll-GSCF-1/GSCF-2'

    label_y1 = 'pitch-Normal/GSCF-1'
    label_y2 = 'pitch-GSCF-1/GSCF-2'


    label_z1 = 'yaw-Normal/GSCF-1'
    label_z2 = 'yaw-GSCF-1/GSCF-2'


    base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    # 转换为日期并打印
    time = []
    time1=[]
    time2=[]
    time3=[]

    for sec in seconds:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)


    for sec in number_time['number_1'][0]:
        dt = base_time + datetime.timedelta(seconds=sec)
        time1.append(dt)

    for sec in number_time['number_2'][0]:
        dt = base_time + datetime.timedelta(seconds=sec)
        time2.append(dt)

    for sec in number_time['number_3'][0]:
        dt = base_time + datetime.timedelta(seconds=sec)
        time3.append(dt)

    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    fig = plt.figure(figsize=(13.6, 10.4))
    fig1 = plt.figure()
    ax1 = fig.add_axes([0.085, 0.045, 0.5, 0.3])
    ax2 = fig.add_axes([0.085, 0.360, 0.5, 0.3])
    ax3 = fig.add_axes([0.085, 0.675,0.5, 0.3])

    ax4 = fig.add_axes([0.675, 0.045, 0.3, 0.3])
    ax5 = fig.add_axes([0.675, 0.360, 0.3, 0.3])
    ax6 = fig.add_axes([0.675, 0.675,0.3, 0.3])

    psd1, freq1 = plt.psd(res1[1], Fs=1, NFFT=len(res1[1]))
    psd2, freq2 = plt.psd(res1[2], Fs=1, NFFT=len(res1[1]))
    psd3, freq3 = plt.psd(res1[3], Fs=1, NFFT=len(res1[1]))

    ax1.scatter(time, res1[1], color=color_x, label=label_x2, s=0.1, alpha=0.5)
    ax2.scatter(time, res1[2], color=color_y, label=label_y2, s=0.1, alpha=0.5)
    ax3.scatter(time, res1[3], color=color_z, label=label_z2, s=0.1, alpha=0.5)

    ax4.plot(freq1, psd1, label=label_x2, lw=0.5, alpha=0.8, color='red')
    ax5.plot(freq2, psd2, label=label_y2, lw=0.5, alpha=0.8, color='red')
    ax6.plot(freq3, psd3, label=label_z2, lw=0.5, alpha=0.8, color='red')

    axx1 = ax1.twinx()
    axx2 = ax2.twinx()
    axx3 = ax3.twinx()

    axx1.scatter(time1, number_time['number_1'][1], color='red',s=0.5)
    axx1.scatter(time2, number_time['number_2'][1], color='red',s=0.3)
    axx1.scatter(time3, number_time['number_3'][1], color='red',s=0.1)

    axx2.scatter(time1, number_time['number_1'][1], color='red',s=0.5)
    axx2.scatter(time2, number_time['number_2'][1], color='red',s=0.3)
    axx2.scatter(time3, number_time['number_3'][1], color='red',s=0.1)

    axx3.scatter(time1, number_time['number_1'][1], color='red',s=0.5)
    axx3.scatter(time2, number_time['number_2'][1], color='red',s=0.3)
    axx3.scatter(time3, number_time['number_3'][1], color='red',s=0.1)

    axx1.set_ylim(-3.2, 3.2)
    axx2.set_ylim(-3.2, 3.2)
    axx3.set_ylim(-3.2, 3.2)

    axx1.tick_params(axis='y', direction='in', top=True, bottom=True, left=False, right=False, width=1.5, labelsize=20,
                     labelright=False)
    axx2.tick_params(axis='y', direction='in', top=True, bottom=True, left=False, right=False, width=1.5, labelsize=20,
                     labelright=False)
    axx3.tick_params(axis='y', direction='in', top=True, bottom=True, left=False, right=False, width=1.5, labelsize=20,
                     labelright=False)


    # 自定义矩形的宽度
    rectangle_width = 0.00005
    height = psd1.max()
    # 画出频率为 0.0001736 的倍数的点附近的条带状矩形
    for freq in np.arange(0.0001736, 0.0014, 0.0001736):
        # 计算矩形的位置和大小
        index = np.abs(freq1 - freq).argmin()
        x = freq1[index] - rectangle_width / 2
        width = rectangle_width
        # 绘制矩形
        rect = patches.Rectangle((x, 0), width, height, alpha=0.5, color='grey')
        ax4.add_patch(rect)

    # 自定义矩形的宽度
    rectangle_width = 0.00005
    height = psd2.max()
    # 画出频率为 0.0001736 的倍数的点附近的条带状矩形
    for freq in np.arange(0.0001736, 0.0014, 0.0001736):
        # 计算矩形的位置和大小
        index = np.abs(freq2 - freq).argmin()
        x = freq2[index] - rectangle_width / 2
        width = rectangle_width
        # 绘制矩形
        rect = patches.Rectangle((x, 0), width, height, alpha=0.5, color='grey')
        ax5.add_patch(rect)

    # 自定义矩形的宽度
    rectangle_width = 0.00005
    height = psd3.max()
    # 画出频率为 0.0001736 的倍数的点附近的条带状矩形
    for freq in np.arange(0.0001736, 0.0014, 0.0001736):
        # 计算矩形的位置和大小
        index = np.abs(freq3 - freq).argmin()
        x = freq3[index] - rectangle_width / 2
        width = rectangle_width
        # 绘制矩形
        rect = patches.Rectangle((x, 0), width, height, alpha=0.5, color='grey')
        ax6.add_patch(rect)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax1.tick_params(direction='in', top=True, bottom=True, left=True)
    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20)
    font={'size':20}
    ax1.set_xlabel('Time (hours)', fontdict=font)
    ax1.set_ylabel('Res_angle (arcsec)', fontdict=font)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    y_major_locator=MultipleLocator(10)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.text(0.04, 0.05, s='(e)', fontdict={'size': 25}, transform=ax1.transAxes)
    ax1.grid(True, axis='y', linestyle='dashed')

    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax2.tick_params(direction='in', top=True, bottom=True, left=True)
    ax2.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax2.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20,
                    labelbottom=False)
    font={'size':20}
    ax2.set_ylabel('Res_angle (arcsec)', fontdict=font)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    y_major_locator=MultipleLocator(10)
    ax2.yaxis.set_major_locator(y_major_locator)
    ax2.text(0.04, 0.05, s='(c)', fontdict={'size': 25}, transform=ax2.transAxes)
    ax2.grid(True, axis='y', linestyle='dashed')

    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax3.tick_params(direction='in', top=True, bottom=True, left=True)
    ax3.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax3.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20,
                    labelbottom=False)
    font={'size':20}
    ax3.set_ylabel('Res_angle (arcsec)', fontdict=font)
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    ax3.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    y_major_locator=MultipleLocator(10)
    ax3.yaxis.set_major_locator(y_major_locator)
    ax3.text(0.04, 0.05, s='(a)', fontdict={'size': 25}, transform=ax3.transAxes)
    ax3.grid(True, axis='y', linestyle='dashed')

    ax1.set_ylim(-24,24)
    ax2.set_ylim(-24, 24)
    ax3.set_ylim(-24,24)

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, linestyle='dashed')
    font = {'size': 20}
    ax4.set_xlabel('Frequency [Hz]', fontdict=font)
    ax4.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)',fontdict=font)
    ax4.tick_params(direction='in', top=True, bottom=True, left=True)
    ax4.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax4.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    for spine in ax4.spines.values():
        spine.set_linewidth(2)
    ax4.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    ax4.text(0.06, 0.05, s='(f)', fontdict={'size': 25}, transform=ax4.transAxes)

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.grid(True, linestyle='dashed')
    font = {'size': 20}
    ax5.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
    ax5.tick_params(direction='in', top=True, bottom=True, left=True)
    ax5.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax5.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20,
                    labelbottom=False)
    for spine in ax5.spines.values():
        spine.set_linewidth(2)
    ax5.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    ax5.text(0.06, 0.05, s='(d)', fontdict={'size': 25}, transform=ax5.transAxes)

    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.grid(True, linestyle='dashed')
    font = {'size': 20}
    ax6.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
    ax6.tick_params(direction='in', top=True, bottom=True, left=True)
    ax6.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax6.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20,
                    labelbottom=False)
    for spine in ax6.spines.values():
        spine.set_linewidth(2)
    ax6.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    ax6.text(0.06, 0.05, s='(b)', fontdict={'size': 25}, transform=ax6.transAxes)

    fig.savefig('../paper_figure/figure5.png', dpi=300)

    plt.show()

    pass


def plot_paper_Euler(res1,Label=None):
    """
    plot the residual angle without ADS_HS time
    """


    seconds = res1[0, :]


    label_x1 = 'roll-%s' % Label
    label_y1 = 'pitch-%s' % Label
    label_z1 = 'yaw-%s' % Label



    base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    # 转换为日期并打印
    time = []
    time1=[]
    time2=[]
    time3=[]

    for sec in seconds:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)

    color_x = (31 / 255, 119 / 255, 180 / 255)
    color_y = (229 / 255, 114 / 255, 12 / 255)
    color_z = (35 / 255, 142 / 255, 35 / 255)

    # fig = plt.figure(figsize=(13.6, 15.6))
    fig = plt.figure(figsize=(13.6, 10.4))
    fig1 = plt.figure()
    ax1 = fig.add_axes([0.085, 0.045, 0.5, 0.3])
    ax2 = fig.add_axes([0.085, 0.360, 0.5, 0.3])
    ax3 = fig.add_axes([0.085, 0.675,0.5, 0.3])

    ax4 = fig.add_axes([0.675, 0.045, 0.3, 0.3])
    ax5 = fig.add_axes([0.675, 0.360, 0.3, 0.3])
    ax6 = fig.add_axes([0.675, 0.675,0.3, 0.3])

    # ax1 = fig.add_axes([0.085, 0.145, 0.5, 0.3])
    # ax2 = fig.add_axes([0.085, 0.460, 0.5, 0.3])
    # ax3 = fig.add_axes([0.085, 0.775,0.5, 0.3])
    #
    # ax4 = fig.add_axes([0.675, 0.145, 0.3, 0.3])
    # ax5 = fig.add_axes([0.675, 0.460, 0.3, 0.3])
    # ax6 = fig.add_axes([0.675, 0.775,0.3, 0.3])

    psd1, freq1 = plt.psd(res1[1], Fs=1, NFFT=len(res1[1]))
    psd2, freq2 = plt.psd(res1[2], Fs=1, NFFT=len(res1[1]))
    psd3, freq3 = plt.psd(res1[3], Fs=1, NFFT=len(res1[1]))

    ax1.scatter(time, res1[1], color=color_x, label=label_x1, s=0.1, alpha=0.5)
    ax2.scatter(time, res1[2], color=color_y, label=label_y1, s=0.1, alpha=0.5)
    ax3.scatter(time, res1[3], color=color_z, label=label_z1, s=0.1, alpha=0.5)

    ax4.plot(freq1, psd1, label=label_x1, lw=0.5, alpha=0.8, color='red')
    ax5.plot(freq2, psd2, label=label_y1, lw=0.5, alpha=0.8, color='red')
    ax6.plot(freq3, psd3, label=label_z1, lw=0.5, alpha=0.8, color='red')

    # 自定义矩形的宽度
    rectangle_width = 0.00005
    height = psd1.max()
    # 画出频率为 0.0001736 的倍数的点附近的条带状矩形
    for freq in np.arange(0.0001736, 0.0014, 0.0001736):
        # 计算矩形的位置和大小
        index = np.abs(freq1 - freq).argmin()
        x = freq1[index] - rectangle_width / 2
        width = rectangle_width
        # 绘制矩形
        rect = patches.Rectangle((x, 0), width, height, alpha=0.5, color='grey')
        ax4.add_patch(rect)

    # 自定义矩形的宽度
    rectangle_width = 0.00005
    height = psd2.max()
    # 画出频率为 0.0001736 的倍数的点附近的条带状矩形
    for freq in np.arange(0.0001736, 0.0014, 0.0001736):
        # 计算矩形的位置和大小
        index = np.abs(freq2 - freq).argmin()
        x = freq2[index] - rectangle_width / 2
        width = rectangle_width
        # 绘制矩形
        rect = patches.Rectangle((x, 0), width, height, alpha=0.5, color='grey')
        ax5.add_patch(rect)

    # 自定义矩形的宽度
    rectangle_width = 0.00005
    height = psd3.max()
    # 画出频率为 0.0001736 的倍数的点附近的条带状矩形
    for freq in np.arange(0.0001736, 0.0014, 0.0001736):
        # 计算矩形的位置和大小
        index = np.abs(freq3 - freq).argmin()
        x = freq3[index] - rectangle_width / 2
        width = rectangle_width
        # 绘制矩形
        rect = patches.Rectangle((x, 0), width, height, alpha=0.5, color='grey')
        ax6.add_patch(rect)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax1.tick_params(direction='in', top=True, bottom=True, left=True)
    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20)
    font={'size':20}
    ax1.set_xlabel('Time (hours)', fontdict=font)
    ax1.set_ylabel('Res_angle (arcsec)', fontdict=font)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    y_major_locator=MultipleLocator(4)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.text(0.04, 0.05, s='(e)', fontdict={'size': 25}, transform=ax1.transAxes)
    ax1.grid(True, axis='y', linestyle='dashed')

    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax2.tick_params(direction='in', top=True, bottom=True, left=True)
    ax2.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax2.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20,
                    labelbottom=False)
    font={'size':20}
    ax2.set_ylabel('Res_angle (arcsec)', fontdict=font)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    y_major_locator=MultipleLocator(3)
    ax2.yaxis.set_major_locator(y_major_locator)
    ax2.text(0.04, 0.05, s='(c)', fontdict={'size': 25}, transform=ax2.transAxes)
    ax2.grid(True, axis='y', linestyle='dashed')

    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 设置间隔为 2 小时
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax3.tick_params(direction='in', top=True, bottom=True, left=True)
    ax3.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
    ax3.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, labelsize=20,
                    labelbottom=False)
    font={'size':20}
    ax3.set_ylabel('Res_angle (arcsec)', fontdict=font)
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    ax3.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    y_major_locator=MultipleLocator(3)
    ax3.yaxis.set_major_locator(y_major_locator)
    ax3.text(0.04, 0.05, s='(a)', fontdict={'size': 25}, transform=ax3.transAxes)
    ax3.grid(True, axis='y', linestyle='dashed')

    ax1.set_ylim(-12,8)
    ax2.set_ylim(-7, 7)
    ax3.set_ylim(-9,9)

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, linestyle='dashed')
    font = {'size': 20}
    ax4.set_xlabel('Frequency [Hz]', fontdict=font)
    ax4.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)',fontdict=font)
    ax4.tick_params(direction='in', top=True, bottom=True, left=True)
    ax4.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax4.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    for spine in ax4.spines.values():
        spine.set_linewidth(2)
    ax4.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    ax4.text(0.06, 0.05, s='(f)', fontdict={'size': 25}, transform=ax4.transAxes)

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.grid(True, linestyle='dashed')
    font = {'size': 20}
    ax5.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
    ax5.tick_params(direction='in', top=True, bottom=True, left=True)
    ax5.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax5.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20,
                    labelbottom=False)
    for spine in ax5.spines.values():
        spine.set_linewidth(2)
    ax5.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    ax5.text(0.06, 0.05, s='(d)', fontdict={'size': 25}, transform=ax5.transAxes)

    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.grid(True, linestyle='dashed')
    font = {'size': 20}
    ax6.set_ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
    ax6.tick_params(direction='in', top=True, bottom=True, left=True)
    ax6.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax6.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20,
                    labelbottom=False)
    for spine in ax6.spines.values():
        spine.set_linewidth(2)
    ax6.legend(loc='upper right', ncol=1, fontsize='x-large', framealpha=0.8)
    ax6.text(0.06, 0.05, s='(b)', fontdict={'size': 25}, transform=ax6.transAxes)

    fig.savefig('../paper_figure/figure5.png', dpi=300)

    plt.show()

    pass


def plot_PDF(Resx,Resy,Resz):
    fig=plt.figure(figsize=(23.8, 6.4))

    color_x = 'green'
    color_y = 'orange'
    color_z = 'blue'

    for key in Coord:

        ax1 = fig.add_axes([0.05+0.32*key.value, 0.1, 0.28, 0.8])
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

        ax1.hist(data, bins=100, density=True, alpha=0.7, color=color, label='HP-C-No_1-%s'%key.name)
        xmin=np.min(data)
        xmax=np.max(data)
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(data), np.std(data[1]))
        ax1.plot(x, p, 'grey', linewidth=5)
        font = {'size': 20}
        ax1.set_title('Probability Density Distribution', fontdict=font)
        if key.value==0:
            ax1.set_ylabel('Probability Density', fontdict=font)
        else:
            pass
        ax1.set_xlabel('Value', fontdict=font)
        ax1.legend(loc='upper right', ncol=1, fontsize=18, framealpha=0.8)
        ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, width=1.5, labelsize=20)
        ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
        plt.grid(True, linestyle='dashed')

        for spine in ax1.spines.values():
            spine.set_linewidth(2)

    fig.savefig('../paper_figure/figure9.png', dpi=300)
    plt.show()

    pass


def demo1():
    """
    Analyse the Pointing variation
    """

    day='2020-04-10'
    date = day
    sat=SatID.C

    label0 = 'Normal'
    label1 = 'GSCF-1'
    label2 = 'GSCF-2'
    label3 = 'GSCF-3'

    color0 = '#ED7C17'
    color1 = '#088C00'
    color2 = '#1868B2'
    # color3 = '#DE582B'
    color3 = '#C00000'


    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('../data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    # sca1B = L1B.getSCA(sat)
    a = plot(sat=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

    sca_new0 = a.get_prior_r().q_combine1()
    time0, roll0, pitch0, yaw0=Pointing(SCA_attitude=sca_new0,sat=SatID.C)\
        .loadLos(date=day,dataDir='../result/product/GRACE_FO/RL04/L1B').getRPY(sat=SatID.C)

    # r1=np.array([[3,3,30],[3,3,30],[3,3,30]])
    # r2 = np.array([[3.01, 1.63, 28.64], [2.10, 1.59, 23.25], [2.54, 2.20, 23.59]])#2020-04-10
    # sca_new2 = a.get_TV_r().q_combine1()
    sca_new1 = a.get_TV_r().q_combine1()
    time1, roll1, pitch1, yaw1=Pointing(SCA_attitude=sca_new1,sat=SatID.C)\
        .loadLos(date=day,dataDir='../result/product/GRACE_FO/RL04/L1B').getRPY(sat=SatID.C)

    sca_new2 = a.get_TV_r().q_combine2()
    time2, roll2, pitch2, yaw2 = Pointing(SCA_attitude=sca_new2, sat=SatID.C) \
        .loadLos(date=day, dataDir='../result/product/GRACE_FO/RL04/L1B').getRPY(sat=SatID.C)

    sca_new3 = a.q_combine3()
    time3, roll3, pitch3, yaw3=Pointing(SCA_attitude=sca_new3,sat=SatID.C)\
        .loadLos(date=day,dataDir='../result/product/GRACE_FO/RL04/L1B').getRPY(sat=SatID.C)


    roll0 *= 1000
    pitch0 *= 1000
    yaw0 *= 1000

    roll1 *= 1000
    pitch1 *= 1000
    yaw1 *= 1000

    roll2 *= 1000
    pitch2 *= 1000
    yaw2 *= 1000

    roll3 *= 1000
    pitch3 *= 1000
    yaw3 *= 1000

    stdx0=np.std(roll0[1:])
    stdy0=np.std(pitch0[1:])
    stdz0=np.std(yaw0[1:])

    stdx1=np.std(roll1[1:])
    stdy1=np.std(pitch1[1:])
    stdz1=np.std(yaw1[1:])

    stdx2=np.std(roll2[1:])
    stdy2=np.std(pitch2[1:])
    stdz2=np.std(yaw2[1:])

    stdx3=np.std(roll3[1:])
    stdy3=np.std(pitch3[1:])
    stdz3=np.std(yaw3[1:])

    meanx0 = np.mean(roll0[1:])
    meanx1 = np.mean(roll1[1:])
    meanx2 = np.mean(roll2[1:])
    meanx3 = np.mean(roll3[1:])

    meany0 = np.mean(pitch0[1:])
    meany1 = np.mean(pitch1[1:])
    meany2 = np.mean(pitch2[1:])
    meany3 = np.mean(pitch3[1:])

    meanz0 = np.mean(yaw0[1:])
    meanz1 = np.mean(yaw1[1:])
    meanz2 = np.mean(yaw2[1:])
    meanz3 = np.mean(yaw3[1:])

    stdx01=np.std(roll0[1:]-roll1[1:])
    stdy01=np.std(pitch0[1:]-pitch1[1:])
    stdz01=np.std(yaw0[1:]-yaw1[1:])
    meanx01=np.mean(roll0[1:]-roll1[1:])
    meany01=np.mean(pitch0[1:]-pitch1[1:])
    meanz01=np.mean(yaw0[1:]-yaw1[1:])


    stdx12=np.std(roll1[1:]-roll2[1:])
    stdy12=np.std(pitch0[1:]-pitch2[1:])
    stdz12=np.std(yaw1[1:]-yaw2[1:])
    meanx12=np.mean(roll1[1:]-roll2[1:])
    meany12=np.mean(pitch0[1:]-pitch2[1:])
    meanz12=np.mean(yaw1[1:]-yaw2[1:])






    base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
    # 转换为日期并打印
    time = []
    for sec in time0:
        dt = base_time + datetime.timedelta(seconds=sec)
        time.append(dt)

    time0=time
    time1=time
    time2=time
    time3=time

    fig=plt.figure(figsize=(9.6,10.4))
    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    ax1 = fig.add_subplot(321)
    ax1.plot(time0, roll0, label=label0, lw=0.8, c=color0)
    ax1.plot(time1, roll1, label=label1, lw=0.8, c=color1)
    ax1.plot(time1, roll0 - roll1, label='Diff', lw=0.8, c=(0.5, 0.5, 0.5), alpha=0.8, linestyle='--')

    font = {'size':20}
    ax1.set_ylabel('Roll [mrad]',fontdict=font)
    # ax1.legend(loc='upper right', ncol=3, fontsize='x-large', framealpha=1)
    ax1.set_ylim([-3, 3])
    # ax1.set_ylim([-5, 5])
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # 设置间隔为 interval 小时
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.grid(True, axis='y', linestyle='dashed')

    ax1.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False)
    ax1.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

    spines = ax1.spines

    spines['top'].set_linewidth(1.5)
    spines['bottom'].set_linewidth(1.5)
    spines['left'].set_linewidth(1.5)
    spines['right'].set_linewidth(1.5)

    ax2 = fig.add_subplot(322)
    ax2.plot(time3, roll3, label=label3, lw=0.8, c=color3)
    ax2.plot(time2, roll2, label=label2, lw=0.8, c=color2)
    # ax2.plot(time1, roll1, label=label1, lw=0.8, c=color1)
    ax2.plot(time0, roll0, label=label0, lw=0.8, c=color0)

    # ax2.plot(time2, roll1 - roll2, label='Diff', lw=0.8, c=(0.5, 0.5, 0.5), alpha=0.8, linestyle='--')

    font = {'size':20}
    # ax2.legend(loc='upper right', ncol=3, fontsize='x-large', framealpha=1)
    ax2.set_ylim([-3, 3])
    # ax1.set_ylim([-5, 5])
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # 设置间隔为 2 小时
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax2.grid(True, axis='y', linestyle='dashed')

    ax2.tick_params(direction='in', top=True, bottom=True, left=True, labelbottom=False)
    ax2.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax2.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

    spines = ax2.spines

    spines['top'].set_linewidth(1.5)
    spines['bottom'].set_linewidth(1.5)
    spines['left'].set_linewidth(1.5)
    spines['right'].set_linewidth(1.5)


    ax3 = fig.add_subplot(323)
    ax3.plot(time0, pitch0, label=label0, lw=0.8, c=color0)
    ax3.plot(time1, pitch1, label=label1, lw=0.8, c=color1)
    ax3.plot(time1, pitch0 - pitch1, label='Diff', lw=0.8, c=(0.5, 0.5, 0.5), alpha=0.8, linestyle='--')

    font={'size':20}
    ax3.set_ylabel('Pitch [mrad]', fontdict=font)
    # ax3.legend(loc='upper right', ncol=3, fontsize='x-large', framealpha=1)
    ax3.set_ylim([-0.4,0.1])
    # ax3.set_ylim([-1, 1])
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # 设置间隔为 2 小时
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.grid(True, axis='y', linestyle='dashed')

    ax3.tick_params(direction='in', top=True, bottom=True, left=True,labelbottom=False)
    ax3.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax3.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

    spines = ax3.spines

    spines['top'].set_linewidth(1.5)
    spines['bottom'].set_linewidth(1.5)
    spines['left'].set_linewidth(1.5)
    spines['right'].set_linewidth(1.5)

    ax4 = fig.add_subplot(324)
    ax4.plot(time3, pitch3, label=label3, lw=0.8, c=color3)
    ax4.plot(time2, pitch2, label=label2, lw=0.8, c=color2)
    # ax4.plot(time1, pitch1, label=label1, lw=0.8, c=color1)
    ax4.plot(time0, pitch0, label=label0, lw=0.8, c=color0)
    # ax4.plot(time2, pitch1 - pitch2, label='Diff', lw=0.8, c=(0.5, 0.5, 0.5), alpha=0.8, linestyle='--')

    font={'size':20}
    # ax4.legend(loc='upper right', ncol=3, fontsize='x-large', framealpha=1)
    # ax4.set_ylim([-0.6,0.6])
    ax4.set_ylim([-1, 1])
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # 设置间隔为 2 小时
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.grid(True, axis='y', linestyle='dashed')

    ax4.tick_params(direction='in', top=True, bottom=True, left=True,labelbottom=False)
    ax4.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax4.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

    spines = ax4.spines

    spines['top'].set_linewidth(1.5)
    spines['bottom'].set_linewidth(1.5)
    spines['left'].set_linewidth(1.5)
    spines['right'].set_linewidth(1.5)


    ax5 = fig.add_subplot(325)
    ax5.plot(time0, yaw0, label=label0, lw=0.8, c=color0)
    ax5.plot(time1, yaw1, label=label1, lw=0.8, c=color1)
    ax5.plot(time1, yaw0 - yaw1, label='Diff', lw=0.8, c=(0.5, 0.5, 0.5), alpha=0.8, linestyle='--')

    font={'size':20}
    ax5.set_xlabel('Time (h)', fontdict=font)
    ax5.set_ylabel('Yaw [mrad]', fontdict=font)
    # ax5.legend(loc='upper right', ncol=3, fontsize='x-large',framealpha=1)
    ax5.set_ylim([-0.4, 0.4])
    # ax5.set_ylim([-1, 1])
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax5.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # 设置间隔为 2 小时
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.grid(True, axis='y', linestyle='dashed')

    ax5.tick_params(direction='in', top=True, bottom=True, left=True)
    ax5.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax5.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

    spines = ax5.spines

    spines['top'].set_linewidth(1.5)
    spines['bottom'].set_linewidth(1.5)
    spines['left'].set_linewidth(1.5)
    spines['right'].set_linewidth(1.5)

    ax6 = fig.add_subplot(326)
    ax6.plot(time3, yaw3, label=label3, lw=0.8, c=color3)
    ax6.plot(time2, yaw2, label=label2, lw=0.8, c=color2)
    # ax6.plot(time1, yaw1, label=label1, lw=0.8, c=color1)
    ax6.plot(time0, yaw0, label=label0, lw=0.8, c=color0)
    # ax6.plot(time2, yaw1 - yaw2, label='Diff', lw=0.8, c=(0.5, 0.5, 0.5), alpha=0.8, linestyle='--')

    font={'size':20}
    ax6.set_xlabel('Time (h)', fontdict=font)
    # ax6.legend(loc='upper right', ncol=3, fontsize='x-large',framealpha=1)
    # ax6.set_ylim([-0.5, 0.5])
    ax6.set_ylim([-1, 1])
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax6.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # 设置间隔为 2 小时
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    # y_major_locator=MultipleLocator(0.2)
    # ax6.yaxis.set_major_locator(y_major_locator)

    plt.grid(True, axis='y', linestyle='dashed')

    ax6.tick_params(direction='in', top=True, bottom=True, left=True)
    ax6.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
    ax6.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)

    spines = ax6.spines

    spines['top'].set_linewidth(1.5)
    spines['bottom'].set_linewidth(1.5)
    spines['left'].set_linewidth(1.5)
    spines['right'].set_linewidth(1.5)

    fig.savefig('../paper_figure/figure8-3.png', dpi=300)
    plt.show()


    pass

def demo2(begindate='2020-02-01',enddate='2020-02-29',sat = SatID.C):
    """
    saving the Fusion Product of star cameras
    """
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    for day in daylist:
        try:
            date = day.strftime('%Y-%m-%d')
            L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1A')
            SCA1A = L1A.getSCA(sat)
            L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1B')
            TIM1B = L1B.getTIM(sat)
            CLK1B = L1B.getCLK(sat)
            a = plot(sat=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)
            # a.SavingFusionProducts_txt(model=Model.Normal)
            a.SavingFusionProducts_txt(model=Model.GSCF_1)
            # a.SavingFusionProducts_txt(model=Model.GSCF_2)
            # a.SavingFusionProducts_txt(model=Model.GSCF_3)
        except:
            print('the day of data is empty on %s'%day.strftime('%Y-%m-%d'))


def demo3():
    """
    Mutual comparison：Normal/GSCF-1/GSCF-2, plot the residual and PSD
    """
    day='2020-02-12'
    date = day
    sat=SatID.C

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    sca1B = L1B.getSCA(sat)
    a = plot(sat=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)

    # sca_new1 = a.get_prior_r().q_combine1()
    #
    sca_new2 = a.get_TV_r().q_combine1()

    sca_new3 = a.get_TV_r().q_combine2()

    # sca_new4 = a.q_combine3()

    # q1 = quaternion.as_quat_array(sca_new1[1:,1:5])
    q2 = quaternion.as_quat_array(sca_new2[1:,1:5])
    q3 = quaternion.as_quat_array(sca_new3[1:,1:5])
    # q4 = quaternion.as_quat_array(sca_new4[1:,1:5])

    # '''Normal/GSCF-1'''
    # label='Normal/GSCF-1'
    # for i in range(len(sca_new1[1:,0])):
    #     q1[i] = q1[i].inverse()
    # res1 = quaternion.as_float_array(q2*q1)
    # res1 = np.vstack((sca_new1[1:,0].T, res1[:,1:4].T*2*3600*180/np.pi))
    #
    # '''Normal/GSCF-2'''
    # label='Normal/GSCF-2'
    # for j in range(len(sca_new3[1:,0])):
    #     q3[j] = q3[j].inverse()
    # res2 = quaternion.as_float_array(q1*q3)
    # res2 = np.vstack((sca_new3[1:,0].T, res2[:,1:4].T*2*3600*180/np.pi))
    #
    '''GSCF-1/GSCF-2'''
    label='GSCF-1/GSCF-2'
    for k in range(len(sca_new2[1:,0])):
        q2[k] = q2[k].inverse()
    res3 = quaternion.as_float_array(q3*q2)
    res3 = np.vstack((sca_new2[1:, 0].T, res3[:, 1:4].T * 2 * 3600 * 180 / np.pi))

    # '''GSCF-2/GSCF-3'''
    # label='GSCF-2/GSCF-3'
    # for k1 in range(len(sca_new3[1:,0])):
    #     q3[k1] = q3[k1].inverse()
    # res4 = quaternion.as_float_array(q4*q3)
    # res4 = np.vstack((sca_new3[1:, 0].T, res4[:, 1:4].T * 2 * 3600 * 180 / np.pi))

    # number_time=a.ADS_HS_time()

    # Res3x = np.interp(number_time['number_1'][0], res3[0], res3[1])
    # Res3y = np.interp(number_time['number_2'][0], res3[0], res3[1])
    # Res3z = np.interp(number_time['number_3'][0], res3[0], res3[1])



    plot_paper_Euler(res3,Label=label)
    # plot_paper_Euler1(res3, number_time)
    # plot_PDF(Res3x,Res3y,Res3z)




    pass


def demo4():
    day = '2020-02-12'
    date = day
    sat = SatID.C

    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)

    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    sca1B = L1B.getSCA(sat)
    a = plot(sat=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)
    a.ADS_HS_time()
    pass

def demo5():
    RMS_Set1,RMS_Set2,RMS_Set3=plot.ConfigRange_rate(date='2020-5')
    # RMS_Set1[1]=RMS_Set2[1]-RMS_Set1[1]
    # plot.plotPSD(fs=5,data1=RMS_Set2,data2=RMS_Set3)
    # plot.plotPSD_single(data1=RMS_Set1,coord=Coord.x)
    plot.plotPSD_Threedata(fs=5,data1=RMS_Set1,data2=RMS_Set2,data3=RMS_Set3)
    pass


def demo6(date='2020-04-08',sat=SatID.C):
    L1A = GetInstrument_L1A(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1A')
    SCA1A = L1A.getSCA(sat)
    L1B = GetInstrument_L1B(mission=Mission.GRACE_FO, date=date).configDir('E:/data/GRACE_FO/RL04/L1B')
    TIM1B = L1B.getTIM(sat)
    CLK1B = L1B.getCLK(sat)
    a = plot(sat=sat, date=date, SCA1A=SCA1A, TIM1B=TIM1B, CLK1B=CLK1B)
    a.get_prior_r().q_combine_Normal(date=date)



if __name__ == '__main__':
    # demo()
    # demo1()
    # demo2(begindate='2020-02-08', enddate='2020-02-10', sat=SatID.C)
    # demo2(begindate='2020-02-01', enddate='2020-05-31', sat=SatID.D)
    # demo2(begindate='2020-04-01', enddate='2020-04-30', sat=SatID.D)


    # demo3()
    # demo4()
    # demo5()

    # demo6(date='2020-04-08',sat=SatID.C)
    pass