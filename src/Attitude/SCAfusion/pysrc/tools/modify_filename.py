#!/user/bin/env python3
# -*- coding: utf-8 -*-
##python批量更换后缀名
import os
import sys
from pysrc.core.geometry import GeoMathKit

def modify_filename(choice,pa,hz):
    choice=int(choice)
    pa=pa
    hz=hz
    path0=r"%s"%pa
    path1=r"%s"%pa+'\\'
    sys.path.append(path1)
    # print(sys.path)
    # 列出当前目录下所有的文件
    files = os.listdir(path0)
    # files = os.listdir('.')
    #print('files',files)


    for filename in files:
      """
      批量修改后缀名，windows系统中文件    1.txt---> 1.html
      """
      portion = os.path.splitext(filename)
      # 如果后缀是.txt
      if choice==1:
        newname = hz + filename
      elif choice==2:
        newname = portion[0] + hz
      elif choice==3:
        newname = portion[0] + hz + portion[1]
      else:
        break

      filenamedir=path1 +filename
      newnamedir=path1+newname
      os.rename(filenamedir,newnamedir)
      print("文件《%s》修改成功"%filename)
    if choice==4:
      x=input("文件中需替换的字符：")
      for filename in files:
        newname=filename.replace(x,hz)
        filenamedir = path1 + filename
        newnamedir = path1 + newname
        os.rename(filenamedir, newnamedir)
        print("文件《%s》修改成功" % filename)



if __name__=='__main__':
    begindate='2020-02-16'
    enddate='2020-05-31'
    daylist = GeoMathKit.dayListByDay(begindate, enddate)
    choice=3
    hz='_04'
    for day in daylist:
        try:
            pa1 = 'D:\data\FusionProduct\GRACE_FO\RL04\GSCF-2\%s\%s'%(day.strftime('%Y-%m'),day.strftime('%Y-%m-%d'))
            pa2 = 'D:\data\FusionProduct\GRACE_FO\RL04\GSCF-3\%s\%s' % (day.strftime('%Y-%m'), day.strftime('%Y-%m-%d'))
            modify_filename(choice, pa1, hz)
            modify_filename(choice, pa2, hz)
        except:
            print('the file is non-existent on %s'%day.strftime('%Y-%m-%d'))

    pass
