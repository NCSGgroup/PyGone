import os
from src.AOD.CRA_LICOM.Configure.DefaultConfig import Config
import numpy as np
import shutil
class OsFile(Config):
    def __init__(self):
        super().__init__()

    def GetReName(self,variable='SHM',format='grib'):
        for date in self.daylist:
            date = date.strftime('%Y-%m-%d')
            for epoch in self.TimeEpoch:
                day_str = str(int(date.split('-')[2])-1).rjust(2,'0')
                time = epoch.split(':')[0]
                sstn = date.split('-')[0]+date.split('-')[1]+date.split('-')[2]
                ssto = date.split('-')[0]+date.split('-')[1]+day_str
                old_name = '{}-{}{}.{}'.format(variable,ssto,time,format)
                new_name = '{}-{}{}.{}'.format(variable,sstn,time,format)
                majorpath = os.path.join(self.Path,sstn)
                old_file_path = os.path.join(majorpath,old_name)
                new_file_path = os.path.join(majorpath,new_name)
                print('old:',old_file_path)
                print('new:',new_file_path)
                if not os.path.exists(old_file_path):
                    print('No need to change!\n')
                    continue
                else:
                    os.rename(old_file_path, new_file_path)
                    print(f'successfully from {ssto + time} to {sstn + time}')
                    print('\n')
                # try:
                #     os.rename(old_file_path,new_file_path)
                #     print(f'successfully from {ssto+time} to {sstn+time}')
                # except FileNotFoundError:
                #     print(f'File {ssto+time} not exist.')
                # except Exception as e:
                #     print()
    def GetDelete(self,variable='TEM',format='grib',deltype='old'):
        for date in self.daylist:
            date = date.strftime('%Y-%m-%d')
            for epoch in self.TimeEpoch:
                day_str = str(int(date.split('-')[2]) - 1).rjust(2, '0')
                time = epoch.split(':')[0]
                sstn = date.split('-')[0] + date.split('-')[1] + date.split('-')[2]
                ssto = date.split('-')[0] + date.split('-')[1] + day_str
                old_name = '{}-{}{}.{}'.format(variable, ssto, time, format)
                new_name = '{}-{}{}.{}'.format(variable, sstn, time, format)
                majorpath = os.path.join(self.Path, sstn)
                old_file_path = os.path.join(majorpath, old_name)
                new_file_path = os.path.join(majorpath, new_name)
                # print('old:', old_file_path)
                # print('new:', new_file_path)
                if deltype == 'old' or deltype == 'Old':
                    del_file_path = old_file_path
                elif deltype == 'new' or deltype == 'New':
                    del_file_path = new_file_path
                if not os.path.exists(del_file_path):
                    print('No need to delete.\n')
                    continue
                else:
                    os.remove(del_file_path)
                    print(f'successfully delete {del_file_path}.\n')

    def GetShift(self,filename,format):
        source_file = '{}.{}'.format(filename,format)
        print('Source_Path is:',self.Path)
        print('Save_Path is:',self.SavePath)
        source_directory = os.path.join(self.Path,source_file)
        destination_directory = os.path.join(self.SavePath,source_file)
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)
        if os.path.isfile(source_directory):
            shutil.move(source_directory,destination_directory)
        pass


def demo1():
    a = OsFile()
    a.setPath(path='H:/ERA5/model level/2021/')
    a.setDuration(BeginDate='2021-01-01', EndDate='2022-01-01')
    a.setInterval(interval=3)
    a.GetReName(variable='TEM', format='grib')
def demo2(year='2010',var='GPT'):
    b = OsFile()
    b.setPath(path='I:/ERA5/model level/{}/'.format(year))
    b.setDuration(BeginDate='{}-01-01'.format(year), EndDate='{}-12-31'.format(year))
    b.setInterval(interval=3)
    b.GetDelete(variable=var, format='grib.9093e.idx',deltype='new')

def demo3():
    years = np.arange(2020, 2023)
    for year in years:
        year = str(year)
        demo2(year=year, var='sp')
        demo2(year=year, var='GPT')
        demo2(year=year,var='TEM')
        demo2(year=year,var='SHM')


if __name__ == '__main__':
    demo3()
