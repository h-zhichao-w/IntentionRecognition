"""
__title__    = test.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2022/2/27 23:50
__Software__ = Pycharm
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓    ┏┓
            ┏┛┻━━━┛ ┻┓
            ┃         ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑  ┣┓
              ┃　永无BUG！ ┏┛
                ┗┓┓┏━┳┓┏┛
                 ┃┫┫  ┃┫┫
                 ┗┻┛  ┗┻┛
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

standard1 = pd.read_csv('standard/Drone0-plus-20220216-0.05.csv')
standard2 = pd.read_csv('standard/Drone0-plus-20220216--0.02.csv')
file = pd.read_csv('kp = 0.05/Drone0-plus-20220223.csv')

s1 = np.array(standard1.iloc[:, :2])
s2 = np.array(standard2.iloc[:, :2])
f = np.array(file.iloc[:,:2])

ratio1 = s1.shape[0] / f.shape[0]
ratio2 = s2.shape[0] / f.shape[0]
d1 = []
d2 = []
for i in range(f.shape[0]):
    j = int(i*ratio1)
    k = int(i*ratio2)
    differ1 = abs(s1[j] - f[i])
    differ2 = abs(s2[k] - f[i])
    d1.append(differ1[0])
    d2.append(differ2[0])

# plt.plot(ax, '-b')
# plt.plot(abs(s1[:,0] - f[:s1.shape[0],0]), '-r')
# plt.plot(abs(s2[:,0] - f[:s2.shape[0],0]), '-g')
# plt.plot(f[:,0],'-b')
plt.plot(d1, '-r')
plt.plot(d2, '-g')
plt.show()
