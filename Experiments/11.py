"""
__title__    = 11.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2022/2/10 22:19
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
import numpy as np
import NumericalAnalysis as na
import pandas as pd
import matplotlib.pyplot as plt

D = pd.read_csv('Drone2-plus-20220216.csv')
Data = D.iloc[ : , 2]
data = np.array(Data)

length = len(data)
t = np.zeros(length)

for i in range(length):
    t[i] = i * 1

omega = np.zeros(length - 1)
for i in range(length - 1):
    ipt = na.piecewiselinear_interploation(t, data, t[i] + 0.001)
    omega[i] = (ipt - data[i]) / 0.001           # 角速度 - 方向角插值差商

with open('Drone2-omega-0.0.csv', 'a') as f:
    f.write('omega_2\n')
    for w in omega:
        f.write(str(w) + '\n')