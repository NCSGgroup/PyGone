import numpy as np
from src.AOD.CRA_LICOM.Configure.GeoMathKit import GeoMathKit
from src.AOD.CRA_LICOM.Configure.Format import CnmSnm, FormatWrite
from src.AOD.CRA_LICOM.Configure.DefaultConfig import Config
from src.AOD.CRA_LICOM.Configure.LoadSH import AOD_GFZ,AODtype
from src.AOD.CRA_LICOM.Configure.Harmonic import SynthesisType,HarAnalysisType
import xarray as xr
import time as ti
from tqdm import tqdm


class ReWrite(Config):
    def __init__(self):
        super().__init__()
        self.mask_path = '../../data/Auxiliary/'
    def setMask_path(self,path):
        self.mask_path = path
        return self
    def Update(self):
        Nmax = self.Nmax
        mask = xr.open_dataset(self.mask_path+'/mask_720X361.nc')
        ocean_mask = mask['mask'].values
        ocean_mask = np.nan_to_num(ocean_mask,nan=1)
        land_mask = 1 - ocean_mask
        PnmMat = GeoMathKit.getPnmMatrix(self.lat,Nmax,2)
        fm = FormatWrite().setRootDir(self.SavePath)
        begin_time = ti.time()
        for date in tqdm(self.daylist):
            date = date.strftime('%Y-%m-%d')
            cs_file = CnmSnm(date=date,Nmax=180)
            for epoch in self.TimeEpoch:
                CS_ATM = np.array(AOD_GFZ().load('../../result/ATM/').setType(AODtype.ATM).setTime(date,epoch).getCS(Nmax))
                cs_file.add(Cnm=CS_ATM[0],Snm=CS_ATM[1],epoch=epoch,date=date,
                            attribute=AODtype.ATM.name)
                CS_OCN = np.array(AOD_GFZ().load('../../result/Bai_GLO/').setType(AODtype.OCN).setTime(date,epoch).getCS(Nmax))
                cs_file.add(Cnm=CS_OCN[0], Snm=CS_OCN[1], epoch=epoch, date=date,
                            attribute=AODtype.OCN.name)
                CS_GLO = CS_ATM+CS_OCN
                cs_file.add(Cnm=CS_GLO[0], Snm=CS_GLO[1], epoch=epoch, date=date,
                            attribute=AODtype.GLO.name)
                upperair = np.array(AOD_GFZ().load('../../result/upper_demean/').setType(AODtype.ATM).setTime(date, epoch).getCS(Nmax))
                CS_OBA = CS_GLO-upperair
                Space = self.HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(CS_OBA[0]),Sqlm=GeoMathKit.CS_1dTo2d(CS_OBA[1]),
                                          PnmMat=PnmMat,lat=self.lat,lon=self.lon,Nmax=Nmax,kind=SynthesisType.Pressure)
                Space[land_mask==0] = 0
                C_mask,S_mask = self.HM.analysis(Nmax=Nmax,Gqij=Space,lat=self.lat,lon=self.lon,
                                                 PnmMat=PnmMat,kind=HarAnalysisType.Pressure)
                cs_file.add(Cnm=C_mask, Snm=S_mask, epoch=epoch, date=date,
                            attribute=AODtype.OBA.name)
            fm.setCS(cs_file).CRALICOMstyle()
        print(f'Cost time: {ti.time()-begin_time} s')








if __name__ == '__main__':
    a = ReWrite()
    a.setLatLon(grid=0.5)
    a.setMaxdegree(Nmax=180)
    a.setSavePath(path='../../result/Licom_AOD/')
    a.setDuration(BeginDate='2024-03-01',EndDate='2024-04-30')
    a.setInterval(interval=6)
    # a.setPath(path=['../GLO/','../ATM/','../upper_demean/'])
    a.setLovePath(path='D:/Cheung/Data/Auxiliary/')
    a.Update()


