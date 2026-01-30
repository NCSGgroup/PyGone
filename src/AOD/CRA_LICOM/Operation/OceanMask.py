import numpy as np
import xarray as xr
from src.AOD.CRA_LICOM.Configure.Harmonic import SynthesisType,HarAnalysisType
from src.AOD.CRA_LICOM.Configure.GeoMathKit import GeoMathKit
from src.AOD.CRA_LICOM.Configure.DefaultConfig import Config
from src.AOD.CRA_LICOM.Configure.Format import FormatWrite,CnmSnm
from src.AOD.CRA_LICOM.Configure.LoadSH import AOD_GFZ,AODtype
import os
import time as ti
from tqdm import tqdm

class OceanMask(Config):
    def __init__(self):
        super(OceanMask, self).__init__()
        self.oceanmask_path = '../../data/Auxiliary/'
    def setOceaMask(self,path='../../data/Auxiliary/'):
        self.oceanmask_path = path
        return self
    def GetMask(self):
        Nmax = self.Nmax
        OceanMask = xr.open_dataset(self.oceanmask_path+'/OceanMask.nc')
        MaskFile = OceanMask['lsm'].values[0,:,:]
        PnmMat = GeoMathKit.getPnmMatrix(self.lat,Nmax,2)
        # print(type(MaskFile))
        # print(MaskFile.shape)
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)
        print(f'Save path is: {self.SavePath}')
        fm = FormatWrite().setRootDir(self.SavePath)
        begin_time = ti.time()

        for date in tqdm(self.daylist):
            date = date.strftime('%Y-%m-%d')
            cs_file = CnmSnm(date=date,Nmax=Nmax)
            for epoch in self.TimeEpoch:
                try:
                    CS = AOD_GFZ().load(self.Path).setType(AODtype.ATM).setTime(date,epoch).getCS(Nmax)

                except FileNotFoundError:
                    print(f'lack file on {date}')
                except Exception as e:
                    print(f'lack file on {date}-{epoch}')
                # CS = AOD_GFZ().load(self.Path).setType(AODtype.ATM).setTime(date, epoch).getCS(Nmax)
                CS = np.array(CS)
                Space = self.HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(CS[0]),Sqlm=GeoMathKit.CS_1dTo2d(CS[1]),
                                          PnmMat=PnmMat,lat=self.lat,lon=self.lon,Nmax=Nmax,kind=SynthesisType.Pressure)
                Space[MaskFile==0]=0
                C_mask,S_mask = self.HM.analysis(Nmax=Nmax,Gqij=Space,lat=self.lat,lon=self.lon,
                                           PnmMat=PnmMat,kind=HarAnalysisType.Pressure)
                cs_file.add(Cnm=C_mask,Snm=S_mask,
                            epoch=epoch,date=date,attribute=AODtype.ATM.name)
            fm.setCS(cs_file).AODstyle(date=date)
        print(f'Cost time: {ti.time()-begin_time} s')

if __name__ == '__main__':
    a = OceanMask()
    a.setLovePath(path='../../data/Auxiliary/')
    a.setLatLon(grid=0.5)
    a.setMaxdegree(Nmax=180)
    a.setPath(path='../../result/upper/')
    a.setDuration(BeginDate='2005-01-01',EndDate='2023-12-31')
    a.setInterval(interval=6)
    a.setSavePath(path='../../result/upper_mask/')
    a.GetMask()