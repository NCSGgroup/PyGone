import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pysrc.analysis.hfn_correlation import CorrelationAnalysis
from pysrc.core.settings import SatID, SCAID
import matplotlib as mpl
from matplotlib import cm

# # way 1
# array1 = np.random.rand(7,9)
# print(array1.shape)
#
# y_ticks = ['03','06','09','12','15','18','21']
# x_ticks = []
# ax = sns.heatmap(array1, yticklabels=y_ticks, cbar_kws={'label': 'XXX [xxx]'})
# #
# ax.set_title('XXX')
# ax.set_xlabel('Value')
# ax.set_ylabel('Time')
# plt.show()


#way 2
a=CorrelationAnalysis(date='2020-03-04',sat=SatID.C)
A = a.Analysis2(win_size=2,scaID=SCAID.No_3).T
sqrt_A = np.sqrt(np.abs(A))

plt.imshow(sqrt_A)
cbar = plt.colorbar(plt.imshow(sqrt_A, cmap='viridis', clim=(0, 30), interpolation='hamming'), shrink=0.8)
cbar.set_label('standard deviation')
cbar.set_ticks([6,12,18,24])

x_labels = [ '2h',  '4h',  '6h',  '8h',  '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h']
y_labels = ['R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33']

plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels) # rotation用于旋转标签，避免重叠
plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

plt.xlabel('Time(h)')
plt.ylabel('The unit of matrix')

plt.savefig('../paper_figure/Fig6.png', dpi=300)


plt.show()





