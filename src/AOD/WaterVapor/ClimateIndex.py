"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/11/30 下午12:05
@Description:
"""
from GeoMathKit import GeoMathKit
import numpy as np


class Nino34:

    def __init__(self):
        self.__nino = None
        self.__load()
        pass

    def __load(self):
        res = []

        with open('../data/Auxiliary/nina34.data') as f:
            content = f.readlines()
            pass
        for item in content:
            value = item.split()
            if len(value) == 0: continue
            for i in range(1, len(value)):
                res.append([int(value[0]) + (i - 1) / 12, float(value[i])])

        self.__nino = np.array(res)

        pass

    def setDate(self, begin, end):
        """
        Year-month
        :param begin: '2016-05'
        :param end: '2019-08'
        :return:
        """
        Months = GeoMathKit.monthListByMonth(begin=begin, end=end)

        ind = (self.__nino[:, 0] >= Months[0].year + (Months[0].month - 1) / 12) * \
              (self.__nino[:, 0] <= Months[-1].year + (Months[-1].month - 1) / 12)

        return self.__nino[ind, :]


def demo1():
    import matplotlib.pyplot as plt
    ni = Nino34()
    res = ni.setDate('1960-01', '2020-08')

    plt.style.use(['science', 'grid', 'ieee'])
    ax = plt.subplot(1, 1, 1)

    x = res[:, 0]
    y1 = res[:, 1]
    y1 = y1 -np.mean(y1)
    y2 = np.zeros(len(x))
    plt.plot(x, y1, label='SSTs', color='black', lw=1)
    plt.plot(x, y2, color='black', lw=1)
    ax.fill_between(x, y1, y2, where=(y1 > y2), color='red', alpha=0.3,
                     interpolate=True)
    ax.fill_between(x, y1, y2, where=(y1 <= y2), color='blue', alpha=0.3,
                     interpolate=True)
    plt.legend(fontsize=10, ncol=3)
    # plt.xlim(1979, 2019)
    plt.title('El-nino 3.4 Index')

    plt.show()
    pass


if __name__ == '__main__':
    demo1()
