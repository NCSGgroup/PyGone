from pysrc.getSunPos.src.EOP import EOP
from pysrc.getSunPos.src.Frame import Frame
from pysrc.getSunPos.Preference.Pre_Frame import FrameConfig
from pysrc.getSunPos.src.PlanetEphemerides import PlanetEphemerides
from pysrc.getSunPos.src.EnumType import TimeFormat
import json
import quaternion


def demo(time):
    '''load json'''
    frame_config = FrameConfig()
    frame_dict = json.load(open('D:\Git\gitcode\Attitude\satellite-attitude-determination\pysrc\getSunPos\setting\FrameConfig.json', 'r'))
    frame_config.__dict__ = frame_dict
    '''init'''
    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)
    planetEphemerides = PlanetEphemerides().configure(FrameConfig=frame_config).setPlanets()
    '''set time'''
    frame = fr.setTime(time, TimeFormat.GPS_second)
    planetEphemerides.setTime(frame).getPosByFrame()
    '''get pos'''
    pos_sun = planetEphemerides.getPosPlanets()['Sun']
    pos_moon = planetEphemerides.getPosPlanets()['Moon']
    # pos_earth=planetEphemerides.getPosPlanets()['Earth']
    #
    # sun=pos_sun-pos_earth
    # moon=pos_moon-pos_earth

    itrs_sun = frame.PosGCRS2ITRS(pos_sun, frame.getRotationMatrix)
    itrs_moon = frame.PosGCRS2ITRS(pos_moon, frame.getRotationMatrix)

    gcrs_sun = frame.PosITRS2GCRS(itrs_sun, frame.getRotationMatrix)
    gcrs_moon = frame.PosITRS2GCRS(itrs_moon, frame.getRotationMatrix)
    # print(itrs_sun)
    # print(itrs_moon)

    return gcrs_sun, gcrs_moon


def demo1(time):
    '''load json'''
    frame_config = FrameConfig()
    frame_dict = json.load(open('D:\Git\gitcode\Attitude\satellite-attitude-determination\pysrc\getSunPos\setting\FrameConfig.json', 'r'))
    frame_config.__dict__ = frame_dict
    '''init'''
    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)
    planetEphemerides = PlanetEphemerides().configure(FrameConfig=frame_config).setPlanets()
    '''set time'''
    frame = fr.setTime(time, TimeFormat.GPS_second)
    planetEphemerides.setTime(frame).getPosByFrame()
    '''get pos'''
    pos_sun = planetEphemerides.getPosPlanets()['Sun']
    # pos_earth=planetEphemerides.getPosPlanets()['Earth']
    #
    # sun=pos_sun-pos_earth
    # moon=pos_moon-pos_earth

    itrs_sun = frame.PosGCRS2ITRS(pos_sun, frame.getRotationMatrix)

    gcrs_sun = frame.PosITRS2GCRS(itrs_sun, frame.getRotationMatrix)
    # print(itrs_sun)
    # print(itrs_moon)

    return gcrs_sun


if __name__ == '__main__':
    gcrs_sun,gcrs_moon=demo(51222.5)
    pass