import numpy as np
from src.AOD.CRA_LICOM.Configure.GeoMathKit import GeoMathKit
import json
import os
from tqdm import tqdm
from src.AOD.CRA_LICOM.Configure.LoadSH import AOD_GFZ,AODtype
from src.AOD.CRA_LICOM.Configure.Format import FormatWrite,CnmSnm
import time as ti



class Config():
    def __init__(self,period):
        self.period = period
        self.GeoHeight = 'model'
        self.SavePath = 'H:/Paper3/paper_result/'
        self.Pathsp = None
        self.Pathupper = None
        self.PathMean = None
        self.PathMean_Pressure = None
        self.meansp = None
        self.meanup = None
        self.daylist = None
        self.interval = None
        self.TimeEpoch = ['00:00:00','03:00:00','06:00:00','09:00:00',
                          '12:00:00','15:00:00','18:00:00','21:00:00']
        self.Nmax = 180
        self.Grid = None
    def setGeoHeight(self,GeoHeight='model'):
        self.GeoHeight = GeoHeight
        return self
    def setSavePath(self,path):
        self.SavePath = path
        return self
    def setSpPath(self,path):
        self.Pathsp = path
        return self
    def setUpperPath(self,path):
        self.Pathupper = path
        return self
    def setMeanPath(self,path):
        self.PathMean = path
        return self
    def setMeanPath_Pressure(self,path):
        self.PathMean_Pressure = path
        return
    def setDuration(self,begin,end):
        self.daylist = GeoMathKit.dayListByDay(begin,end)
        return self
    def setInterval(self,interval=3):
        Time = []
        for i in np.arange(0, 24, int(interval)):
            i = str(i).rjust(2, '0')
            epoch = '{}:00:00'.format(i)
            Time.append(epoch)
        self.TimeEpoch = Time
        return self
    def setMaxDegree(self,Nmax=180):
        self.Nmax=Nmax
        return self
    def setGrid(self,grid=0.5):
        self.Grid = grid
        return self

class GetMeanField(Config):
    def __init__(self,period):
        super().__init__(period=period)
    def getMeanField(self,MeanType='sp'):
        begin_time = ti.time()
        print('MaxDegree : {}'.format(self.Nmax))
        print('Mean Type : {}'.format(MeanType))
        Load_Path = self.Pathsp
        mtye = 'sp'
        if MeanType == 'sp' or MeanType == 'SP' or MeanType == 'surface pressure':
            mtye = 'sp'
            Load_Path = self.Pathsp
        elif MeanType == 'up' or MeanType == 'upper' or MeanType == 'Upper Air' or MeanType == 'upp':
            mtye = 'upper'
            Load_Path = self.Pathupper
        Nmax = self.Nmax
        Save_Path = os.path.join(self.PathMean,'{}{}'.format(mtye,self.period))
        if not os.path.exists(Save_Path):
            os.makedirs(Save_Path)
        print('Load Path : {}'.format(Load_Path))
        print('Save Path : {}'.format(Save_Path))
        if not os.path.exists(Save_Path):
            os.makedirs(Save_Path)
        CS = []
        for date in tqdm(self.daylist):
            date = date.strftime('%Y-%m-%d')
            for time in self.TimeEpoch:
                FILE_CS = np.array(AOD_GFZ().load(Load_Path).setType(AODtype.ATM).setTime(date, time).getCS(Nmax))
                CS.append(FILE_CS)
        CS = np.array(CS)
        mean = np.mean(CS, axis=0)
        print("Cost time: %s ms" % ((ti.time() - begin_time) * 1000))
        np.save(Save_Path + '/mean.npy', mean)
class GetAD(Config):
    def __init__(self,period):
        super().__init__(period=period)
    def getAD(self):
        begin_time = ti.time()
        save_path = self.SavePath
        mean_path = self.PathMean + '/mean.npy'
        Nmax = self.Nmax
        print('SP Path is: {}'.format(self.Pathsp))
        print('Upper Path is: {}'.format(self.Pathupper))
        print('Mean Paths are:\n{}'.format(mean_path))
        print('Save Path is: {}'.format(save_path))
        print(f'MaxDegree is: {Nmax}')
        mean = np.load(mean_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        daylist = self.daylist
        TimeEpoch = self.TimeEpoch
        fm = FormatWrite().setRootDir(save_path)
        for date in tqdm(daylist, ncols=100):
            date = date.strftime('%Y-%m-%d')
            cs = CnmSnm(date=date, Nmax=Nmax)
            for time in TimeEpoch:
                CS_SP = np.array(AOD_GFZ().load(self.Pathsp).setType(AODtype.ATM).setTime(date, time).getCS(Nmax))
                CS_UP = np.array(AOD_GFZ().load(self.Pathupper).setType(AODtype.ATM).setTime(date, time).getCS(Nmax))
                De_CS = CS_UP + CS_SP - mean
                cs.add(Cnm=De_CS[0], Snm=De_CS[1], epoch=time, date=date, attribute=AODtype.ATM.name)
            fm.setCS(cs).AODstyle(date=date)
        print("Cost time: %s ms" % ((ti.time() - begin_time) * 1000))
        return self

    @staticmethod
    def DefaultConfig(isWrite=True):
        config = {'Function': 'AD',
                  'period': '2022',
                  'Path_SP': 'H:/Paper3/paper_result/sp_model/',
                  'Path_Upp':'H:/Paper3/paper_result/upper_pressure/',
                  'Path_Mean':'H:/Paper3/paper_result/mean_model/',
                  'Path_Mean_Pressure':'H:/Paper3/paper_result/mean_pressure/',
                  'Save_Path':'H:/Paper3/paper_result/5_Spatial/G0.5/',
                  'BeginDate': '2022-01-01',
                  'EndDate': '2022-12-31',
                  'interval': 3,
                  'Nmax': 180,
                  'GeoHeight': 'pressure',
                  'MeanType': 'sp',
                  'Grid': 0.5}
        if isWrite:
            with open('Demean.json', 'w') as f:
                f.write(json.dumps(config, indent=13))
        return config
def demo_json():
    # a = GetAD(period='2003')
    # a.DefaultConfig()
    with open('Demean.json','r') as f:
        config = json.load(f)
    if config['Function'] == 'Mean' or config['Function'] == 'MeanField':
        a = GetMeanField(period=config['period'])
        a.setSpPath(path=config['Path_SP'])
        a.setUpperPath(path=config['Path_Upp'])
        a.setMeanPath(path=config['Path_Mean'])
        beginDate = config['BeginDate']
        endDate = config['EndDate']
        a.setInterval(interval=config['interval'])
        a.setDuration(begin=beginDate,end=endDate)
        a.getMeanField(MeanType=config['MeanType'])
    elif config['Function'] == 'AD':
        a = GetAD(period=config['period'])
        a.setSpPath(path=config['Path_SP'])
        a.setUpperPath(path=config['Path_Upp'])
        a.setMeanPath(path=config['Path_Mean'])
        a.setSavePath(path=config['Save_Path'])
        a.setInterval(interval=config['interval'])
        beginDate = config['BeginDate']
        endDate = config['EndDate']
        a.setDuration(begin=beginDate,end=endDate)
        a.getAD()




def demo_GetMean():
    a = GetMeanField(period='2003-2022')
    a.setSpPath(path='H:/Paper3/paper_result/sp_model/')
    a.setUpperPath(path='H:/Paper3/paper_result/upper_model/')
    a.setMeanPath(path='H:/Paper3/paper_result/mean_model/')
    a.setDuration(begin='2003-01-01',end='2022-12-31')
    a.getMeanField(MeanType='sp')
def demo_GetAD():
    a = GetAD(period='2022')
    a.setSpPath(path='H:/Paper3/paper_result/1_Tide/sp_model_0310/')
    a.setUpperPath(path='H:/Paper3/paper_result/1_Tide/sp_model_0310/')
    a.setMeanPath(path='H:/Paper3/paper_result/1_Tide/mean_model_0310/')
    a.setSavePath(path='H:/Paper3/paper_result/')
    a.setDuration(begin='2022-01-12',end='2022-01-12')
    a.getAD()
if __name__ == '__main__':
    demo_json()
    # demo_GetMean()
    # demo_GetAD()