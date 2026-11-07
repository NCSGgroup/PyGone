# import datetime
#
# # 定义一系列秒数
# seconds = [599659201, 599659224, 599659278]
# base_time = datetime.datetime(2000, 1, 1)
# # 转换为日期并打印
# for sec in seconds:
#     dt = base_time + datetime.timedelta(seconds=sec)
#     print(dt)
#
# date_time = datetime.datetime(2023, 9, 28)
# timestamp = (date_time - datetime.datetime(1970, 1, 1,12, 0, 0)).total_seconds()
# pass
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 生成一些示例数据
# x = np.linspace(0.1, 100, 100)
# y = np.exp(x)
#
# # 创建一个对数坐标的图形
# plt.figure()
# plt.plot(x, y)
#
# # 设置 x 轴和 y 轴的对数刻度
# plt.xscale('log')
# plt.yscale('log')
#
# # 获取当前坐标轴对象
# ax = plt.gca()
#
# # 设置 x 轴和 y 轴刻度朝向坐标轴内部
# ax.tick_params(axis='x', direction='in')
# ax.tick_params(axis='y', direction='in')
#
#
#
# # 显示图形
# plt.show()
#
# pass
import matplotlib.font_manager
import numpy as np

# import datetime
#
# base_time = datetime.datetime(2000, 1, 1, 12, 0, 0)
#
# sec = 636589001
# date = base_time+datetime.timedelta(seconds=sec)
# pass

# import matplotlib.pyplot as plt
#
# # 创建一个新的图形
# fig = plt.figure()
#
# # 添加子图1
# ax1 = fig.add_axes([0.1, 0.1, 0.35, 0.35])
# ax1.plot([0, 1], [0, 1], color='blue')
#
# ax1.text(0.25,-0.30,s='(a) figure1',fontdict={'size':15})
#
#
#
# # 显示图形
# plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
#
# # 生成一组随机数据，这里以正态分布为例
# data = np.random.normal(loc=0, scale=1, size=1000)
#
# # 绘制直方图
# plt.hist(data, bins=30, density=True, alpha=0.7, color='blue')
#
# # 绘制概率密度曲线
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, np.mean(data), np.std(data))
# plt.plot(x, p, 'k', linewidth=2)
#
# plt.title('Probability Density Distribution')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义x、y散点坐标——你自己的数据
# x = [10, 20, 30, 40, 50, 60, 70, 80]
# x = np.array(x)
# print('x is :\n', x)
# num = [174, 236, 305, 334, 349, 351, 342, 323]
# y = np.array(num)
# print('y is :\n', y)
#
# # 用n次多项式拟合，此处n = 3，设置为3-7都比较合理，视案例复杂度以及需求而定
# f1 = np.polyfit(x, y, 3)
# print('f1 is :\n', f1)
#
# # 拟合y值
# yvals = np.polyval(f1, x)
# print('yvals is :\n', yvals)
#
# # 绘图
# plot1 = plt.plot(x, y, 's', label='original values')
# plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4)  # 指定legend的位置右下角
# plt.title('polyfitting')
# plt.show()


# import numpy as np
#
# # 原始的一维数组
# original_array = np.array([2, 3, 4, 5])
#
# # 创建一个全是1的一维数组
# row_of_ones = np.ones(original_array.shape[0], dtype=int)
#
# # 将全是1的数组添加到原始数组的末尾
# new_array = np.append(original_array, row_of_ones, axis=0)
#
# print(new_array)


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# # 假设一天中每隔两个小时的气温（℃）数据
# x = range(2, 26, 2)
# y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]
#
# # 绘制折线图
# plt.plot(x, y)
# plt.xlabel("时间（小时）")
# plt.ylabel("气温（℃）")
# plt.title("一天中每隔两个小时的气温变化")
# plt.grid(True)  # 添加网格线
#
# # 保存图形（可选）
# plt.savefig("temperature_plot.png")
#
# # 显示图形
# plt.show()
#
# # 创建一个空白的图形
# fig, ax = plt.subplots()
#
# def update(frame):
#     # 清除之前的图形
#     ax.clear()
#
#     # 绘制当前帧的数据
#     plt.plot(x[:frame + 1], y[:frame + 1])
#     plt.xlabel("时间（小时）")
#     plt.ylabel("气温（℃）")
#     plt.title("一天中每隔两个小时的气温变化")
#
# # 创建动画
# ani = FuncAnimation(fig, update, frames=len(x), repeat=False)
#
# # 保存动画（可选）
# ani.save("temperature_animation.gif", writer="imagemagick", fps=2)
#
# # 显示动画
# plt.show()
# a=0
# if a!=0:
#     b=1
# else:
#     b=2
#
# print(b)

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成示例信号
# fs = 1000                  # 采样频率 (Hz)
# t = np.arange(0, 1, 1/fs)  # 时间向量
# f1 = 50                  # 信号中的第一个频率 (Hz)
# f2 = 120                   # 信号中的第二个频率 (Hz)
# signal = 2*np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)  # 原始信号
#
# # 对信号进行傅里叶变换
# Y = np.fft.fft(signal)
#
# # 计算频率轴
# N = len(signal)
# frequencies = np.fft.fftfreq(N, 1/fs)
#
# # 仅使用前半部分频谱，因为 FFT 结果是对称的
# Y_half = Y[:N//2]
# frequencies_half = frequencies[:N//2]
#
# # 找到幅值最大的频率成分
# idx = np.argmax(np.abs(Y_half))
#
# # 创建频域滤波器，只保留最大频率成分
# filter = np.zeros_like(Y)
# filter[idx] = Y[idx]             # 保留最大频率成分
# filter[N-idx] = Y[N-idx]         # 对称频率成分
#
# # 进行反向傅里叶变换
# extracted_signal = np.fft.ifft(filter)
#
# # 绘制结果
# plt.figure(figsize=(12, 8))
#
# plt.subplot(3, 1, 1)
# plt.plot(t, signal)
# plt.title('原始信号')
# plt.xlabel('时间 (秒)')
# plt.ylabel('幅值')
#
# plt.subplot(3, 1, 2)
# plt.plot(frequencies_half, np.abs(Y_half) / (N/2))
# plt.title('傅里叶变换后的频谱（归一化）')
# plt.xlabel('频率 (Hz)')
# plt.ylabel('幅值')
#
# plt.subplot(3, 1, 3)
# plt.plot(t, extracted_signal.real)  # 反向变换得到的信号并进行归一化
# plt.title('提取出的最大频率信号（归一化）')
# plt.xlabel('时间 (秒)')
# plt.ylabel('幅值')
#
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.font_manager import FontProperties
# plt.rcParams['font.family'] = 'Arial Unicode MS'
# font=FontProperties(fname='SimHei.ttf',size=14)
# # font_path='/path/to/simhei.ttf'
# # font=matplotlib.font_manager.FontProperties(fname=font_path)
# # 生成一个包含两个频率成分的模拟信号
# fs = 1000  # 采样频率为 1000 Hz
# t = np.linspace(0, 1, fs, endpoint=False)  # 生成时间序列，1秒内的采样点数为 fs
# data1 = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)  # 包含频率为 50 Hz 和 120 Hz 的信号
#
#
# # 绘制频谱图
# plt.figure(figsize=(8, 4))
# plt.plot(t, data1)
# plt.xlabel('频率 (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Magnitude Spectrum')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# segments = [
#     [(0, 5), (1, 2), (2, 4), (3, 5), (4, 6), (5, 7), (6, 9)],
#     [(9, 6), (10, 3), (11, 7)],
#     [(18, 4), (19, 2), (20, 0), (21, 1), (22, 2)],
#     [(40, 4), (41, 5), (42, 5)]
# ]
# segment_index = 1  # 想要提取的连续段的索引
# desired_segment = segments[segment_index]
#
# # 提取第二个连续段中每个元组的数据组成二维数组
# extracted_data = np.array([list(pair) for pair in desired_segment])
#
#
# print("第二个连续段中每个元组的数据的二维数组:")
# for row in extracted_data:
#     print(row)


import numpy as np
import quaternion

a1=np.array([0.1123787229532157,-0.2805042773497927,-0.9523351063083009,-0.0417877770242927])
a2=np.array([0.1123921347612161,-0.2807420669473304,-0.9522596836227688,-0.0418735577258719])

a=quaternion.as_quat_array(a1)
b=quaternion.as_quat_array(a2)

bb=a.inverse() * b
bb=quaternion.as_float_array(bb)
aa=bb*2*3600*180/np.pi
pass




















