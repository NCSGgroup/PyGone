import os
os.environ['GMT_LIBRARY_PATH'] = 'D:/App/GMT/gmt6/bin'



import datetime

import numpy as np
import pandas as pd
import pygmt
import xarray as xr

# pygmt.config(PS_CONVERT="D:/App/GMT/Ghostscript/gs10.03.0/bingswin64c.exe")

# fig = pygmt.Figure()
# fig.basemap(region="g", projection="W15c", frame=True)
# fig.coast(land="#666666", water="skyblue")
# fig.show()

# import pygmt
#
# # 创建图形对象
# fig = pygmt.Figure()
#
# # 定义 color palette table (CPT)
# cmap = 'seis'
# pygmt.makecpt(cmap=cmap, series=[-5000, 5000])
#
# # 加载示例数据，这里使用地球高程数据
# data = pygmt.datasets.load_earth_relief()
#
# # 绘制网格图并应用 CPT
# fig.grdimage(grid=data, cmap=cmap, projection="X6i", frame=True)
#
# # 添加 colorbar
# fig.colorbar(cmap=cmap, frame=['a0.2',"af+l'Elevation (m)'"])
#
# # 显示图形
# fig.show()

# import pygmt
#
# # 创建图形对象
# fig = pygmt.Figure()
#
# # 绘制折线图
# fig.plot(
#     x=[0, 1, 2, 3, 4, 5],
#     y=[0, 1, 2, 3, 4, 5],
#     pen="2p,blue",
#     label="GSCF-2"
# )
#
# # 设置地图的范围、标题和格网
# fig.basemap(
#     region=[-1, 6, -1, 6],                # X 和 Y 轴的范围
#     projection="X10c/8c",                 # 使用 X 投影，宽度 10 cm，高度 8 cm
#     frame=[
#         "+t(a)",                          # 设置标题
#         "+xaf+lDay",                      # X 轴标签
#         "+yaf+lKBRR Residual RMS (um/s)", # Y 轴标签
#         "+y+ggray",                       # Y 轴网格线颜色
#         "+x+gwhite",                      # X 轴网格线颜色
#         "+y+ggray+lw0.5p+ldashed"        # Y 轴虚线样式
#     ]
# )
#
# # 添加图例
# fig.legend(
#     position="JTR+o0.5c",
#     box="+gwhite"
# )
#
# # 显示图形
# fig.show()
#
#
#
# import pygmt
#
# # Set up same sample data
# x = [2.2, 3.3, -3.1, -3.7, -0.1]
# y = [1.8, -1.2, -0.9, -4.5, 4.5]
#
# # Create new Figure instance
# fig = pygmt.Figure()
#
# # -----------------------------------------------------------------------------
# # Left: record order
# fig.basemap(region=[-5, 5, -5, 5], projection="X6c", frame=["WSne", "af"])
#
# # Connect data points based on the record order [Default connection=None]
# fig.plot(x=x, y=y, pen="1.5p,dodgerblue")
# # Plot data points
# fig.plot(x=x, y=y, style="c0.2c", fill="green3", pen="1.5p")
#
# fig.shift_origin(xshift="w+0.5c")
#
# # -----------------------------------------------------------------------------
# # Middle: network
# fig.basemap(region=[-5, 5, -5, 5], projection="X6c", frame=["wSne", "af"])
#
# # Connect data points as network
# fig.plot(x=x, y=y, pen="1.5p,dodgerblue", connection="n")
# # Plot data points
# fig.plot(x=x, y=y, style="c0.2c", fill="green3", pen="1.5p")
#
# fig.shift_origin(xshift="w+0.5c")
#
# # -----------------------------------------------------------------------------
# # Right: reference point
# fig.basemap(region=[-5, 5, -5, 5], projection="X6c", frame=["wSne", "af"])
#
# # Connect data points with the reference point (0,0)
# fig.plot(x=x, y=y, pen="1.5p,dodgerblue", connection="p0/0")
# # Plot data points
# fig.plot(x=x, y=y, style="c0.2c", fill="green3", pen="1.5p")
# # Plot reference point
# fig.plot(x=0, y=0, style="s0.3c", fill="gold", pen="1.5p")
#
# fig.show()


# import pygmt
#
# # 创建一个 PyGMT Figure 对象
# fig = pygmt.Figure()
#
# # 使用 subplots 创建 1x2 的子图布局
# # 参数设置：nrows=1, ncols=2 表示 1 行 2 列的子图布局
# fig.subplot(nrows=1, ncols=2, figsize=("6i", "3i"))
#
# # 在第一个子图中绘制内容
# with fig.subplot(panel=0):
#     fig.plot(
#         x=[0, 1],
#         y=[0, 1],
#         pen="2p,blue"
#     )
#     fig.text(x=0.5, y=0.5, text="Subplot 1", font="12p,Helvetica-Bold,black", justify="CM")
#
# # 在第二个子图中绘制内容
# with fig.subplot(panel=1):
#     fig.plot(
#         x=[0, 1],
#         y=[1, 0],
#         pen="2p,red"
#     )
#     fig.text(x=0.5, y=0.5, text="Subplot 2", font="12p,Helvetica-Bold,black", justify="CM")
#
# # 显示图形
# fig.show()

# fig = pygmt.Figure()
# fig.coast(
#     region=[-69, -68, 43.75, 44.75],
#     shorelines=True,
#     land="lightgreen",
#     water="lightblue",
#     projection="M20c",
#     frame=["a", "+tMaine"],
# )
# fig.show()











