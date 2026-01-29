"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/8/4
@Description:
"""

import sys

sys.path.append('../')

from GeoMathKit import GeoMathKit

import os
import wget
import json
from Setting import Level, Mission


class DataCollection:

    def __init__(self, config: dict):

        self.Misson = Mission[config['Mission']]
        self.Level = Level[config['Level']]
        self._LocalDir = os.path.join(config['Local directory'], self.Level.name)
        self._RemoteDir = config['Remote site']

        self.dataspan = [config['Data begin'], config['Data end']]

        self.byDay(*self.dataspan)

    def downLoadFile(self, LocalFile, RemoteFile):
        """
        single file download
        :param LocalFile:
        :param RemoteFile:
        :return:
        """
        wget.download(RemoteFile, LocalFile)
        return True

    def byDay(self, begin, end):
        """
        usage: begin='2002-01-01', end='2002-02-03'
        :param begin:
        :param end:
        :return:
        """
        daylist = GeoMathKit.dayListByDay(begin, end)

        for day in daylist:

            localdir = os.path.join(self._LocalDir, day.strftime("%Y-%m"))
            isExists = os.path.exists(localdir)
            if not isExists:
                print('Month for %s' % day.strftime("%Y-%m"))
                os.makedirs(localdir)

            outname, RemotePath = None, None
            if self.Misson is Mission.GRACE_FO:
                outname = 'gracefo_%s_%s_RL04.ascii.noLRI.tgz' % (self.Level.name[1:3], day.strftime("%Y-%m-%d"))
                RemotePath = os.path.join(self._RemoteDir, day.strftime("%Y"), outname)
            else:
                pass

            print('\n' + outname)

            self.downLoadFile(LocalFile=localdir,
                              RemoteFile=RemotePath)
            GeoMathKit.un_tar(os.path.join(localdir, outname))

            pass

        pass

    @staticmethod
    def defaultConfig():
        config = {}

        config['Local directory'] = './RL06/'
        config['Remote site'] = 'ftp://anonymous@isdcftp.gfz-potsdam.de/grace-fo/Level-1B/GFZ/AOD/RL06/'
        config['Data begin'] = '2021-01'
        config['Data end'] = '2021-12'

        with open('Config.json', 'w') as f:
            f.write(json.dumps(config))
        pass


def demo(config='../setting/config.download.json'):
    """
    download official RL06
    :return:
    """

    with open(config) as ff:
        configJob = json.load(ff)
    dc = DataCollection(config=configJob)
    pass


if __name__ == '__main__':
    # demo(config='../setting/GRACE_FO_L1A.download.json')
    demo(config='../setting/GRACE_FO_L1B.download.json')
